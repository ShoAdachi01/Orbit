/**
 * Background Reconstruction (Section C1 from PRD)
 * Generates background splat with confidence-weighted completion
 */

import { CameraPose, Matrix4x4 } from '../schemas/types';

export interface ReconstructionConfig {
  /** Maximum prior weight for hole filling (default: 0.2) */
  maxPriorWeight: number;
  /** Minimum hole confidence to apply prior */
  minHoleConfidence: number;
  /** Number of Gaussians for splat */
  numGaussians: number;
}

const DEFAULT_CONFIG: ReconstructionConfig = {
  maxPriorWeight: 0.2,
  minHoleConfidence: 0.3,
  numGaussians: 100000,
};

export interface FrameData {
  rgb: Uint8Array;
  depth: Float32Array;
  mask: Uint8Array;
  width: number;
  height: number;
  pose: CameraPose;
  depthConfidence?: Float32Array;
}

export interface GaussianSplat {
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  count: number;
}

export interface HoleInfo {
  pixels: number[];
  confidence: number;
  depthEstimate: number;
  hasTrackSupport: boolean;
}

export class BackgroundReconstruction {
  private config: ReconstructionConfig;

  constructor(config: Partial<ReconstructionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Generate background splat from masked frames
   * Rule: Always trust real pixels (weight = 1.0)
   * Prior weight = clamp(0, 0.2, hole_conf * depth_conf * track_support_conf)
   */
  async reconstruct(
    frames: FrameData[],
    tracks?: Array<{ frameIndex: number; x: number; y: number; confidence: number }[]>,
    backgroundPrior?: { data: Uint8Array; width: number; height: number }
  ): Promise<GaussianSplat> {
    // Step 1: Collect visible background points from all frames
    const pointCloud = this.collectBackgroundPoints(frames);

    // Step 2: Identify holes (disoccluded regions)
    const holes = this.identifyHoles(frames, tracks);

    // Step 3: Fill holes with prior (if available and confident)
    if (backgroundPrior && holes.length > 0) {
      this.fillHolesWithPrior(pointCloud, holes, backgroundPrior, frames);
    }

    // Step 4: Convert point cloud to Gaussian splat
    const splat = this.pointCloudToGaussians(pointCloud);

    return splat;
  }

  /**
   * Collect background points from masked frames
   */
  private collectBackgroundPoints(frames: FrameData[]): PointCloud {
    const points: Point3D[] = [];

    for (const frame of frames) {
      const { rgb, depth, mask, width, height, pose, depthConfidence } = frame;

      // Get intrinsics from pose (assuming standard projection)
      const fx = width;
      const fy = width;
      const cx = width / 2;
      const cy = height / 2;

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;

          // Skip subject pixels
          if (mask[idx] > 127) continue;

          const d = depth[idx];
          if (!isFinite(d) || d <= 0) continue;

          // Unproject to 3D
          const xCam = (x - cx) * d / fx;
          const yCam = (y - cy) * d / fy;
          const zCam = d;

          // Transform to world space
          const worldPos = this.transformPoint(
            [xCam, yCam, zCam],
            pose.transform
          );

          // Get color
          const rgbIdx = idx * 3;
          const color = [
            rgb[rgbIdx] / 255,
            rgb[rgbIdx + 1] / 255,
            rgb[rgbIdx + 2] / 255,
          ];

          // Confidence from depth if available
          const confidence = depthConfidence ? depthConfidence[idx] : 1.0;

          points.push({
            position: worldPos,
            color: color as [number, number, number],
            confidence,
            weight: 1.0, // Real pixels have weight 1.0
          });
        }
      }
    }

    return { points };
  }

  /**
   * Identify hole regions (disoccluded areas)
   */
  private identifyHoles(
    frames: FrameData[],
    tracks?: Array<{ frameIndex: number; x: number; y: number; confidence: number }[]>
  ): HoleInfo[] {
    const holes: HoleInfo[] = [];

    // For each frame, identify pixels that are:
    // - Not visible in any frame (always behind subject)
    // - Have no track support
    // - Have uncertain depth

    const aggregateVisibility = new Map<string, {
      seenCount: number;
      totalFrames: number;
      avgDepth: number;
      hasTrack: boolean;
    }>();

    // Build visibility map
    for (let fi = 0; fi < frames.length; fi++) {
      const { mask, depth, width, height } = frames[fi];

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const key = `${x},${y}`;

          if (!aggregateVisibility.has(key)) {
            aggregateVisibility.set(key, {
              seenCount: 0,
              totalFrames: frames.length,
              avgDepth: 0,
              hasTrack: false,
            });
          }

          const info = aggregateVisibility.get(key)!;

          // Check if background visible
          if (mask[idx] <= 127 && depth[idx] > 0) {
            info.seenCount++;
            info.avgDepth += depth[idx];
          }
        }
      }
    }

    // Check track support
    if (tracks) {
      for (const track of tracks) {
        for (const point of track) {
          const key = `${Math.round(point.x)},${Math.round(point.y)}`;
          const info = aggregateVisibility.get(key);
          if (info) {
            info.hasTrack = true;
          }
        }
      }
    }

    // Find holes (low visibility, no tracks)
    for (const [key, info] of aggregateVisibility) {
      const visibilityRatio = info.seenCount / info.totalFrames;

      if (visibilityRatio < 0.3 && !info.hasTrack) {
        const [x, y] = key.split(',').map(Number);
        const pixelIdx = y * frames[0].width + x;

        holes.push({
          pixels: [pixelIdx],
          confidence: 1 - visibilityRatio,
          depthEstimate: info.seenCount > 0 ? info.avgDepth / info.seenCount : 0,
          hasTrackSupport: info.hasTrack,
        });
      }
    }

    return holes;
  }

  /**
   * Fill holes with prior using confidence-weighted fusion
   */
  private fillHolesWithPrior(
    pointCloud: PointCloud,
    holes: HoleInfo[],
    prior: { data: Uint8Array; width: number; height: number },
    frames: FrameData[]
  ): void {
    const referenceFrame = frames[0];
    const { width, height } = referenceFrame;

    for (const hole of holes) {
      // Compute prior weight based on confidence
      // weight = clamp(0, maxPriorWeight, hole_conf * depth_conf * track_support_conf)
      let priorWeight = hole.confidence;

      // Reduce weight if no depth estimate
      if (hole.depthEstimate <= 0) {
        priorWeight *= 0.5;
      }

      // Reduce weight if no track support
      if (!hole.hasTrackSupport) {
        priorWeight *= 0.8;
      }

      // Clamp to max
      priorWeight = Math.min(this.config.maxPriorWeight, priorWeight);

      // Skip if confidence too low
      if (priorWeight < this.config.minHoleConfidence * this.config.maxPriorWeight) {
        continue;
      }

      // Add prior points for each hole pixel
      for (const pixelIdx of hole.pixels) {
        const x = pixelIdx % width;
        const y = Math.floor(pixelIdx / width);

        // Get color from prior
        const priorIdx = pixelIdx * 3;
        const color: [number, number, number] = [
          prior.data[priorIdx] / 255,
          prior.data[priorIdx + 1] / 255,
          prior.data[priorIdx + 2] / 255,
        ];

        // Estimate depth from surrounding pixels or hole estimate
        const d = hole.depthEstimate > 0 ? hole.depthEstimate : 5.0;

        // Unproject
        const fx = width;
        const fy = width;
        const cx = width / 2;
        const cy = height / 2;

        const xCam = (x - cx) * d / fx;
        const yCam = (y - cy) * d / fy;
        const zCam = d;

        // Transform to world (using first frame pose as reference)
        const worldPos = this.transformPoint(
          [xCam, yCam, zCam],
          referenceFrame.pose.transform
        );

        pointCloud.points.push({
          position: worldPos,
          color,
          confidence: hole.confidence,
          weight: priorWeight,
        });
      }
    }
  }

  /**
   * Convert point cloud to Gaussian splat representation
   */
  private pointCloudToGaussians(pointCloud: PointCloud): GaussianSplat {
    const { points } = pointCloud;

    // Subsample if needed
    const maxPoints = this.config.numGaussians;
    const sampledPoints = points.length > maxPoints
      ? this.subsamplePoints(points, maxPoints)
      : points;

    const count = sampledPoints.length;

    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    const rotations = new Float32Array(count * 4);
    const colors = new Float32Array(count * 4);
    const opacities = new Float32Array(count);

    // Estimate scale from point density
    const avgScale = this.estimatePointScale(sampledPoints);

    for (let i = 0; i < count; i++) {
      const point = sampledPoints[i];

      // Position
      positions[i * 3 + 0] = point.position[0];
      positions[i * 3 + 1] = point.position[1];
      positions[i * 3 + 2] = point.position[2];

      // Scale (uniform spherical)
      scales[i * 3 + 0] = avgScale;
      scales[i * 3 + 1] = avgScale;
      scales[i * 3 + 2] = avgScale;

      // Rotation (identity quaternion)
      rotations[i * 4 + 0] = 0;
      rotations[i * 4 + 1] = 0;
      rotations[i * 4 + 2] = 0;
      rotations[i * 4 + 3] = 1;

      // Color with alpha
      colors[i * 4 + 0] = point.color[0];
      colors[i * 4 + 1] = point.color[1];
      colors[i * 4 + 2] = point.color[2];
      colors[i * 4 + 3] = 1.0;

      // Opacity (weighted by confidence and prior weight)
      opacities[i] = point.weight * point.confidence;
    }

    return { positions, scales, rotations, colors, opacities, count };
  }

  /**
   * Transform point from camera to world space
   */
  private transformPoint(point: [number, number, number], transform: Matrix4x4): [number, number, number] {
    const [x, y, z] = point;
    return [
      transform[0] * x + transform[4] * y + transform[8] * z + transform[12],
      transform[1] * x + transform[5] * y + transform[9] * z + transform[13],
      transform[2] * x + transform[6] * y + transform[10] * z + transform[14],
    ];
  }

  /**
   * Subsample points using weighted random selection
   */
  private subsamplePoints(points: Point3D[], targetCount: number): Point3D[] {
    // Weight by confidence for importance sampling
    const weights = points.map(p => p.weight * p.confidence);
    const totalWeight = weights.reduce((a, b) => a + b, 0);

    const selected: Point3D[] = [];
    const selectedIndices = new Set<number>();

    while (selected.length < targetCount && selectedIndices.size < points.length) {
      // Weighted random selection
      let r = Math.random() * totalWeight;
      for (let i = 0; i < points.length; i++) {
        if (selectedIndices.has(i)) continue;
        r -= weights[i];
        if (r <= 0) {
          selected.push(points[i]);
          selectedIndices.add(i);
          break;
        }
      }
    }

    return selected;
  }

  /**
   * Estimate appropriate Gaussian scale from point density
   */
  private estimatePointScale(points: Point3D[]): number {
    if (points.length < 2) return 0.01;

    // Sample nearest neighbor distances
    const sampleSize = Math.min(1000, points.length);
    const distances: number[] = [];

    for (let i = 0; i < sampleSize; i++) {
      const idx = Math.floor(Math.random() * points.length);
      const p = points[idx];

      let minDist = Infinity;
      for (let j = 0; j < points.length; j++) {
        if (j === idx) continue;
        const q = points[j];
        const dx = p.position[0] - q.position[0];
        const dy = p.position[1] - q.position[1];
        const dz = p.position[2] - q.position[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        minDist = Math.min(minDist, dist);
      }

      if (isFinite(minDist)) {
        distances.push(minDist);
      }
    }

    if (distances.length === 0) return 0.01;

    // Use median distance scaled up slightly
    distances.sort((a, b) => a - b);
    const median = distances[Math.floor(distances.length / 2)];

    return median * 1.5;
  }
}

interface Point3D {
  position: [number, number, number];
  color: [number, number, number];
  confidence: number;
  weight: number;
}

interface PointCloud {
  points: Point3D[];
}
