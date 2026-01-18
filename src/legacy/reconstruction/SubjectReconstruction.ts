/**
 * Subject 4D Reconstruction (Section C2 from PRD)
 * Generates subject splat with temporal dynamics
 */

import { CameraPose, Matrix4x4 } from '../schemas/types';
import { GaussianSplat } from './BackgroundReconstruction';

export interface SubjectConfig {
  /** Number of Gaussians per frame */
  gaussiansPerFrame: number;
  /** Use 2.5D billboard mode */
  use2DMode: boolean;
  /** Depth prior weight */
  depthWeight: number;
  /** Track confidence threshold */
  minTrackConfidence: number;
}

const DEFAULT_CONFIG: SubjectConfig = {
  gaussiansPerFrame: 50000,
  use2DMode: false,
  depthWeight: 0.5,
  minTrackConfidence: 0.6,
};

export interface SubjectFrameData {
  rgb: Uint8Array;
  depth: Float32Array;
  mask: Uint8Array;
  width: number;
  height: number;
  pose: CameraPose;
  depthConfidence?: Float32Array;
  frameIndex: number;
}

export interface Track {
  id: number;
  points: Array<{
    frameIndex: number;
    x: number;
    y: number;
    confidence: number;
  }>;
}

export interface Subject4DSplat {
  /** Per-frame splats */
  frameSplats: Map<number, GaussianSplat>;
  /** Canonical (reference) frame index */
  canonicalFrame: number;
  /** Track correspondences for deformation */
  trackCorrespondences: Map<number, number[]>; // track_id -> gaussian_ids
}

export class SubjectReconstruction {
  private config: SubjectConfig;

  constructor(config: Partial<SubjectConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Reconstruct 4D subject from masked frames with tracking
   */
  async reconstruct(
    frames: SubjectFrameData[],
    tracks: Track[]
  ): Promise<Subject4DSplat> {
    if (this.config.use2DMode) {
      return this.reconstruct2D(frames);
    }

    return this.reconstruct4D(frames, tracks);
  }

  /**
   * Full 4D reconstruction using SoM + depth + tracks
   */
  private async reconstruct4D(
    frames: SubjectFrameData[],
    tracks: Track[]
  ): Promise<Subject4DSplat> {
    // Step 1: Select canonical (reference) frame
    const canonicalFrame = this.selectCanonicalFrame(frames);

    // Step 2: Build initial splat from canonical frame
    const canonicalSplat = this.frameToSplat(frames[canonicalFrame]);

    // Step 3: Establish track-to-gaussian correspondences
    const trackCorrespondences = this.buildTrackCorrespondences(
      canonicalSplat,
      tracks,
      frames[canonicalFrame]
    );

    // Step 4: Deform canonical splat to each frame using tracks
    const frameSplats = new Map<number, GaussianSplat>();
    frameSplats.set(canonicalFrame, canonicalSplat);

    for (let i = 0; i < frames.length; i++) {
      if (i === canonicalFrame) continue;

      const deformedSplat = this.deformSplat(
        canonicalSplat,
        trackCorrespondences,
        tracks,
        canonicalFrame,
        i,
        frames[canonicalFrame],
        frames[i]
      );

      frameSplats.set(i, deformedSplat);
    }

    return {
      frameSplats,
      canonicalFrame,
      trackCorrespondences,
    };
  }

  /**
   * 2.5D billboard reconstruction (fallback mode)
   */
  private async reconstruct2D(frames: SubjectFrameData[]): Promise<Subject4DSplat> {
    const frameSplats = new Map<number, GaussianSplat>();
    const canonicalFrame = this.selectCanonicalFrame(frames);

    // For 2.5D mode, treat each frame independently as a depth-layered billboard
    for (let i = 0; i < frames.length; i++) {
      const splat = this.frameToBillboard(frames[i]);
      frameSplats.set(i, splat);
    }

    return {
      frameSplats,
      canonicalFrame,
      trackCorrespondences: new Map(),
    };
  }

  /**
   * Select canonical frame (best quality, centered pose)
   */
  private selectCanonicalFrame(frames: SubjectFrameData[]): number {
    let bestScore = -Infinity;
    let bestIndex = 0;

    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];

      // Score based on:
      // - Subject visibility (mask coverage)
      // - Depth confidence
      // - Centered in timeline
      const maskCoverage = this.computeMaskCoverage(frame.mask);
      const depthScore = frame.depthConfidence
        ? this.computeAvgConfidence(frame.depthConfidence, frame.mask)
        : 0.5;
      const timeScore = 1 - Math.abs(i - frames.length / 2) / (frames.length / 2);

      const score = maskCoverage * 0.4 + depthScore * 0.4 + timeScore * 0.2;

      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }

    return bestIndex;
  }

  /**
   * Convert single frame to Gaussian splat
   */
  private frameToSplat(frame: SubjectFrameData): GaussianSplat {
    const { rgb, depth, mask, width, height, pose, depthConfidence } = frame;

    const points: Array<{
      position: [number, number, number];
      color: [number, number, number];
      confidence: number;
    }> = [];

    const fx = width;
    const fy = width;
    const cx = width / 2;
    const cy = height / 2;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;

        // Only include subject pixels
        if (mask[idx] <= 127) continue;

        const d = depth[idx];
        if (!isFinite(d) || d <= 0) continue;

        // Unproject to 3D (in camera space for subject)
        const xCam = (x - cx) * d / fx;
        const yCam = (y - cy) * d / fy;
        const zCam = d;

        // Get color
        const rgbIdx = idx * 3;
        const color: [number, number, number] = [
          rgb[rgbIdx] / 255,
          rgb[rgbIdx + 1] / 255,
          rgb[rgbIdx + 2] / 255,
        ];

        const confidence = depthConfidence ? depthConfidence[idx] : 1.0;

        points.push({
          position: [xCam, yCam, zCam],
          color,
          confidence,
        });
      }
    }

    // Subsample to target count
    const targetCount = Math.min(this.config.gaussiansPerFrame, points.length);
    const sampled = this.subsampleByConfidence(points, targetCount);

    return this.pointsToSplat(sampled);
  }

  /**
   * Convert frame to 2.5D billboard splat
   */
  private frameToBillboard(frame: SubjectFrameData): GaussianSplat {
    const { rgb, depth, mask, width, height } = frame;

    const points: Array<{
      position: [number, number, number];
      color: [number, number, number];
      confidence: number;
    }> = [];

    // Compute bounding box of subject
    let minX = width, maxX = 0, minY = height, maxY = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y * width + x] > 127) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }

    // Create billboard points with small depth variation
    const subjectWidth = maxX - minX;
    const subjectHeight = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Compute median depth for subject
    const subjectDepths: number[] = [];
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const idx = y * width + x;
        if (mask[idx] > 127 && depth[idx] > 0) {
          subjectDepths.push(depth[idx]);
        }
      }
    }
    subjectDepths.sort((a, b) => a - b);
    const medianDepth = subjectDepths.length > 0
      ? subjectDepths[Math.floor(subjectDepths.length / 2)]
      : 3.0;

    // Create points on billboard plane with depth offset
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const idx = y * width + x;

        if (mask[idx] <= 127) continue;

        // Normalized position on billboard
        const u = (x - centerX) / width;
        const v = (y - centerY) / height;

        // Small depth offset from depth map
        const depthOffset = (depth[idx] - medianDepth) * 0.1;

        const color: [number, number, number] = [
          rgb[idx * 3] / 255,
          rgb[idx * 3 + 1] / 255,
          rgb[idx * 3 + 2] / 255,
        ];

        points.push({
          position: [u, v, medianDepth + depthOffset],
          color,
          confidence: 1.0,
        });
      }
    }

    const targetCount = Math.min(this.config.gaussiansPerFrame, points.length);
    const sampled = this.subsampleByConfidence(points, targetCount);

    return this.pointsToSplat(sampled);
  }

  /**
   * Build correspondences between tracks and Gaussians
   */
  private buildTrackCorrespondences(
    splat: GaussianSplat,
    tracks: Track[],
    frame: SubjectFrameData
  ): Map<number, number[]> {
    const correspondences = new Map<number, number[]>();
    const { width, height } = frame;

    // Project Gaussians to 2D
    const gaussianPositions2D: Array<{ x: number; y: number }> = [];
    const fx = width;
    const cx = width / 2;
    const cy = height / 2;

    for (let i = 0; i < splat.count; i++) {
      const x = splat.positions[i * 3];
      const y = splat.positions[i * 3 + 1];
      const z = splat.positions[i * 3 + 2];

      // Project to 2D
      const px = fx * x / z + cx;
      const py = fx * y / z + cy;

      gaussianPositions2D.push({ x: px, y: py });
    }

    // For each track, find nearby Gaussians
    for (const track of tracks) {
      // Find track point at canonical frame
      const trackPoint = track.points.find(p => p.frameIndex === frame.frameIndex);
      if (!trackPoint || trackPoint.confidence < this.config.minTrackConfidence) {
        continue;
      }

      // Find Gaussians within radius
      const radius = 5; // pixels
      const nearbyGaussians: number[] = [];

      for (let gi = 0; gi < gaussianPositions2D.length; gi++) {
        const gPos = gaussianPositions2D[gi];
        const dx = gPos.x - trackPoint.x;
        const dy = gPos.y - trackPoint.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < radius) {
          nearbyGaussians.push(gi);
        }
      }

      if (nearbyGaussians.length > 0) {
        correspondences.set(track.id, nearbyGaussians);
      }
    }

    return correspondences;
  }

  /**
   * Deform splat from canonical frame to target frame using tracks
   */
  private deformSplat(
    canonicalSplat: GaussianSplat,
    correspondences: Map<number, number[]>,
    tracks: Track[],
    canonicalFrameIdx: number,
    targetFrameIdx: number,
    canonicalFrame: SubjectFrameData,
    targetFrame: SubjectFrameData
  ): GaussianSplat {
    // Copy canonical splat
    const deformed: GaussianSplat = {
      positions: new Float32Array(canonicalSplat.positions),
      scales: new Float32Array(canonicalSplat.scales),
      rotations: new Float32Array(canonicalSplat.rotations),
      colors: new Float32Array(canonicalSplat.colors),
      opacities: new Float32Array(canonicalSplat.opacities),
      count: canonicalSplat.count,
    };

    // Compute deformation field from tracks
    const deformationField = new Map<number, { dx: number; dy: number; dz: number }>();

    for (const track of tracks) {
      const canonicalPoint = track.points.find(p => p.frameIndex === canonicalFrameIdx);
      const targetPoint = track.points.find(p => p.frameIndex === targetFrameIdx);

      if (!canonicalPoint || !targetPoint) continue;
      if (canonicalPoint.confidence < this.config.minTrackConfidence) continue;
      if (targetPoint.confidence < this.config.minTrackConfidence) continue;

      const gaussianIds = correspondences.get(track.id);
      if (!gaussianIds) continue;

      // 2D displacement
      const dx2d = targetPoint.x - canonicalPoint.x;
      const dy2d = targetPoint.y - canonicalPoint.y;

      // Convert to 3D displacement (approximate)
      const { width } = canonicalFrame;
      const scale = 0.001; // Scale factor

      for (const gi of gaussianIds) {
        const z = canonicalSplat.positions[gi * 3 + 2];
        deformationField.set(gi, {
          dx: dx2d * scale * z,
          dy: dy2d * scale * z,
          dz: 0, // Could estimate from depth change
        });
      }
    }

    // Apply deformation
    for (const [gi, deform] of deformationField) {
      deformed.positions[gi * 3 + 0] += deform.dx;
      deformed.positions[gi * 3 + 1] += deform.dy;
      deformed.positions[gi * 3 + 2] += deform.dz;
    }

    // Interpolate untracked Gaussians
    this.interpolateUntrackedGaussians(deformed, deformationField);

    return deformed;
  }

  /**
   * Interpolate deformation for Gaussians without direct track correspondence
   */
  private interpolateUntrackedGaussians(
    splat: GaussianSplat,
    deformationField: Map<number, { dx: number; dy: number; dz: number }>
  ): void {
    // Simple nearest-neighbor interpolation
    const trackedIndices = Array.from(deformationField.keys());

    if (trackedIndices.length === 0) return;

    for (let gi = 0; gi < splat.count; gi++) {
      if (deformationField.has(gi)) continue;

      // Find nearest tracked Gaussian
      let minDist = Infinity;
      let nearestIdx = trackedIndices[0];

      const px = splat.positions[gi * 3];
      const py = splat.positions[gi * 3 + 1];
      const pz = splat.positions[gi * 3 + 2];

      for (const ti of trackedIndices) {
        const tx = splat.positions[ti * 3];
        const ty = splat.positions[ti * 3 + 1];
        const tz = splat.positions[ti * 3 + 2];

        const dist = Math.sqrt(
          (px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2
        );

        if (dist < minDist) {
          minDist = dist;
          nearestIdx = ti;
        }
      }

      // Apply same deformation (with distance falloff)
      const deform = deformationField.get(nearestIdx)!;
      const weight = Math.exp(-minDist * 10);

      splat.positions[gi * 3 + 0] += deform.dx * weight;
      splat.positions[gi * 3 + 1] += deform.dy * weight;
      splat.positions[gi * 3 + 2] += deform.dz * weight;
    }
  }

  /**
   * Convert points to Gaussian splat
   */
  private pointsToSplat(points: Array<{
    position: [number, number, number];
    color: [number, number, number];
    confidence: number;
  }>): GaussianSplat {
    const count = points.length;

    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    const rotations = new Float32Array(count * 4);
    const colors = new Float32Array(count * 4);
    const opacities = new Float32Array(count);

    const avgScale = 0.005; // Small scale for subject detail

    for (let i = 0; i < count; i++) {
      const point = points[i];

      positions[i * 3 + 0] = point.position[0];
      positions[i * 3 + 1] = point.position[1];
      positions[i * 3 + 2] = point.position[2];

      scales[i * 3 + 0] = avgScale;
      scales[i * 3 + 1] = avgScale;
      scales[i * 3 + 2] = avgScale;

      rotations[i * 4 + 0] = 0;
      rotations[i * 4 + 1] = 0;
      rotations[i * 4 + 2] = 0;
      rotations[i * 4 + 3] = 1;

      colors[i * 4 + 0] = point.color[0];
      colors[i * 4 + 1] = point.color[1];
      colors[i * 4 + 2] = point.color[2];
      colors[i * 4 + 3] = 1.0;

      opacities[i] = point.confidence;
    }

    return { positions, scales, rotations, colors, opacities, count };
  }

  /**
   * Subsample points by confidence (importance sampling)
   */
  private subsampleByConfidence(
    points: Array<{
      position: [number, number, number];
      color: [number, number, number];
      confidence: number;
    }>,
    targetCount: number
  ): typeof points {
    if (points.length <= targetCount) return points;

    // Weighted sampling by confidence
    const totalConfidence = points.reduce((sum, p) => sum + p.confidence, 0);
    const selected: typeof points = [];
    const selectedSet = new Set<number>();

    while (selected.length < targetCount) {
      let r = Math.random() * totalConfidence;
      for (let i = 0; i < points.length; i++) {
        if (selectedSet.has(i)) continue;
        r -= points[i].confidence;
        if (r <= 0) {
          selected.push(points[i]);
          selectedSet.add(i);
          break;
        }
      }
    }

    return selected;
  }

  private computeMaskCoverage(mask: Uint8Array): number {
    let subjectPixels = 0;
    for (const val of mask) {
      if (val > 127) subjectPixels++;
    }
    return subjectPixels / mask.length;
  }

  private computeAvgConfidence(confidence: Float32Array, mask: Uint8Array): number {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < confidence.length; i++) {
      if (mask[i] > 127) {
        sum += confidence[i];
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  }
}
