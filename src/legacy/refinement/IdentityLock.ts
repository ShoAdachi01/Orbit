/**
 * Identity Lock System (Section D from PRD)
 * Ensures face/logo stability during refinement
 */

import { AnchorFrame, IdentityDriftResult } from '../schemas/types';

export interface ImageFrame {
  data: Uint8Array | Float32Array;
  width: number;
  height: number;
  channels: number;
}

export interface IdentityLockConfig {
  /** Number of anchor frames to select (default: 5) */
  numAnchors: number;
  /** Maximum allowed identity drift (default: 0.15) */
  maxDrift: number;
  /** Minimum sharpness score for anchor selection */
  minSharpness: number;
  /** Minimum face/logo visibility score */
  minIdentityVisibility: number;
}

const DEFAULT_CONFIG: IdentityLockConfig = {
  numAnchors: 5,
  maxDrift: 0.15,
  minSharpness: 0.5,
  minIdentityVisibility: 0.3,
};

export class IdentityLock {
  private config: IdentityLockConfig;
  private anchors: AnchorFrame[] = [];

  constructor(config: Partial<IdentityLockConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Select K anchor frames from video
   * Criteria: high sharpness, face/logo visible, minimal motion blur
   */
  async selectAnchors(
    frames: ImageFrame[],
    masks: Array<{ data: Uint8Array; width: number; height: number }>,
    faceDetector?: (frame: ImageFrame) => Promise<{ detected: boolean; embedding?: Float32Array; region?: { x: number; y: number; width: number; height: number } }>
  ): Promise<AnchorFrame[]> {
    const candidates: Array<{
      frameIndex: number;
      sharpness: number;
      motionBlur: number;
      identityVisibility: number;
      faceEmbedding?: Float32Array;
      cropRegion?: { x: number; y: number; width: number; height: number };
    }> = [];

    // Analyze each frame
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const mask = masks[i];

      // Compute sharpness (Laplacian variance)
      const sharpness = this.computeSharpness(frame, mask);

      // Compute motion blur estimate
      const motionBlur = this.estimateMotionBlur(frame, frames[i - 1], frames[i + 1]);

      // Detect face/identity features if detector provided
      let identityVisibility = 0;
      let faceEmbedding: Float32Array | undefined;
      let cropRegion: { x: number; y: number; width: number; height: number } | undefined;

      if (faceDetector) {
        const faceResult = await faceDetector(frame);
        if (faceResult.detected) {
          identityVisibility = 1.0;
          faceEmbedding = faceResult.embedding;
          cropRegion = faceResult.region;
        }
      } else {
        // Estimate from mask - larger subject area = higher visibility
        const subjectArea = this.computeSubjectArea(mask);
        identityVisibility = Math.min(1, subjectArea * 2);
      }

      // Filter by quality thresholds
      if (
        sharpness >= this.config.minSharpness &&
        identityVisibility >= this.config.minIdentityVisibility
      ) {
        candidates.push({
          frameIndex: i,
          sharpness,
          motionBlur,
          identityVisibility,
          faceEmbedding,
          cropRegion,
        });
      }
    }

    // Sort by quality score (sharpness * visibility / (1 + motionBlur))
    candidates.sort((a, b) => {
      const scoreA = (a.sharpness * a.identityVisibility) / (1 + a.motionBlur);
      const scoreB = (b.sharpness * b.identityVisibility) / (1 + b.motionBlur);
      return scoreB - scoreA;
    });

    // Select distributed anchors (not all from same region of video)
    this.anchors = this.selectDistributedAnchors(candidates, frames.length);

    return this.anchors;
  }

  /**
   * Detect identity drift in refined output vs anchors
   */
  async detectDrift(
    refinedFrames: ImageFrame[],
    embedder?: (frame: ImageFrame, region?: { x: number; y: number; width: number; height: number }) => Promise<Float32Array>
  ): Promise<IdentityDriftResult> {
    if (this.anchors.length === 0) {
      return {
        maxDrift: 0,
        meanDrift: 0,
        driftExceeded: false,
        highDriftFrames: [],
      };
    }

    const drifts: number[] = [];
    const highDriftFrames: number[] = [];

    for (const anchor of this.anchors) {
      if (!anchor.faceEmbedding) continue;

      const refinedFrame = refinedFrames[anchor.frameIndex];
      if (!refinedFrame) continue;

      let refinedEmbedding: Float32Array;

      if (embedder) {
        refinedEmbedding = await embedder(refinedFrame, anchor.cropRegion);
      } else {
        // Fallback: simple pixel-level comparison in crop region
        refinedEmbedding = this.extractSimpleEmbedding(refinedFrame, anchor.cropRegion);
      }

      const drift = this.computeEmbeddingDistance(anchor.faceEmbedding, refinedEmbedding);
      drifts.push(drift);

      if (drift > this.config.maxDrift) {
        highDriftFrames.push(anchor.frameIndex);
      }
    }

    if (drifts.length === 0) {
      return {
        maxDrift: 0,
        meanDrift: 0,
        driftExceeded: false,
        highDriftFrames: [],
      };
    }

    const maxDrift = Math.max(...drifts);
    const meanDrift = drifts.reduce((a, b) => a + b, 0) / drifts.length;

    return {
      maxDrift,
      meanDrift,
      driftExceeded: maxDrift > this.config.maxDrift,
      highDriftFrames,
    };
  }

  /**
   * Apply identity conditioning for refinement
   * Returns conditioning info for the refinement model
   */
  getConditioningInfo(): {
    anchorFrameIds: number[];
    anchorEmbeddings: Float32Array[];
    anchorCrops: Array<{ x: number; y: number; width: number; height: number }>;
  } {
    const anchorFrameIds: number[] = [];
    const anchorEmbeddings: Float32Array[] = [];
    const anchorCrops: Array<{ x: number; y: number; width: number; height: number }> = [];

    for (const anchor of this.anchors) {
      anchorFrameIds.push(anchor.frameIndex);
      if (anchor.faceEmbedding) {
        anchorEmbeddings.push(anchor.faceEmbedding);
      }
      if (anchor.cropRegion) {
        anchorCrops.push(anchor.cropRegion);
      }
    }

    return { anchorFrameIds, anchorEmbeddings, anchorCrops };
  }

  /**
   * Compute image sharpness using Laplacian variance
   */
  private computeSharpness(
    frame: ImageFrame,
    mask: { data: Uint8Array; width: number; height: number }
  ): number {
    const { data, width, height, channels } = frame;

    // Convert to grayscale if needed
    const gray = this.toGrayscale(data, width, height, channels);

    // Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
    let variance = 0;
    let count = 0;
    const laplacianValues: number[] = [];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;

        // Only compute for subject region
        if (mask.data[idx] <= 127) continue;

        const center = gray[idx];
        const top = gray[idx - width];
        const bottom = gray[idx + width];
        const left = gray[idx - 1];
        const right = gray[idx + 1];

        const laplacian = Math.abs(4 * center - top - bottom - left - right);
        laplacianValues.push(laplacian);
        count++;
      }
    }

    if (count === 0) return 0;

    // Compute variance of Laplacian values
    const mean = laplacianValues.reduce((a, b) => a + b, 0) / count;
    variance = laplacianValues.reduce((sum, v) => sum + (v - mean) * (v - mean), 0) / count;

    // Normalize to 0-1 range
    return Math.min(1, variance / 500);
  }

  /**
   * Estimate motion blur from frame differences
   */
  private estimateMotionBlur(
    frame: ImageFrame,
    prevFrame?: ImageFrame,
    nextFrame?: ImageFrame
  ): number {
    if (!prevFrame && !nextFrame) return 0;

    const gray = this.toGrayscale(frame.data, frame.width, frame.height, frame.channels);

    let totalDiff = 0;
    let count = 0;

    if (prevFrame) {
      const prevGray = this.toGrayscale(
        prevFrame.data,
        prevFrame.width,
        prevFrame.height,
        prevFrame.channels
      );

      for (let i = 0; i < gray.length; i++) {
        totalDiff += Math.abs(gray[i] - prevGray[i]);
        count++;
      }
    }

    if (nextFrame) {
      const nextGray = this.toGrayscale(
        nextFrame.data,
        nextFrame.width,
        nextFrame.height,
        nextFrame.channels
      );

      for (let i = 0; i < gray.length; i++) {
        totalDiff += Math.abs(gray[i] - nextGray[i]);
        count++;
      }
    }

    if (count === 0) return 0;

    // Higher difference suggests more motion, which correlates with blur
    const avgDiff = totalDiff / count;
    return Math.min(1, avgDiff / 50);
  }

  /**
   * Compute subject area from mask
   */
  private computeSubjectArea(mask: { data: Uint8Array; width: number; height: number }): number {
    let subjectPixels = 0;
    for (let i = 0; i < mask.data.length; i++) {
      if (mask.data[i] > 127) subjectPixels++;
    }
    return subjectPixels / mask.data.length;
  }

  /**
   * Select anchors distributed across video timeline
   */
  private selectDistributedAnchors(
    candidates: Array<{
      frameIndex: number;
      sharpness: number;
      motionBlur: number;
      identityVisibility: number;
      faceEmbedding?: Float32Array;
      cropRegion?: { x: number; y: number; width: number; height: number };
    }>,
    totalFrames: number
  ): AnchorFrame[] {
    if (candidates.length === 0) return [];

    const selected: AnchorFrame[] = [];
    const segmentSize = Math.ceil(totalFrames / this.config.numAnchors);

    // Select best candidate from each segment
    for (let segment = 0; segment < this.config.numAnchors; segment++) {
      const segmentStart = segment * segmentSize;
      const segmentEnd = Math.min((segment + 1) * segmentSize, totalFrames);

      // Find best candidate in this segment
      const segmentCandidates = candidates.filter(
        (c) => c.frameIndex >= segmentStart && c.frameIndex < segmentEnd
      );

      if (segmentCandidates.length > 0) {
        const best = segmentCandidates[0]; // Already sorted by quality
        selected.push({
          frameIndex: best.frameIndex,
          timestamp: best.frameIndex / 30, // Assume 30fps
          sharpness: best.sharpness,
          identityVisibility: best.identityVisibility,
          motionBlur: best.motionBlur,
          faceEmbedding: best.faceEmbedding,
          cropRegion: best.cropRegion,
        });
      }
    }

    return selected;
  }

  /**
   * Convert image data to grayscale
   */
  private toGrayscale(
    data: Uint8Array | Float32Array,
    width: number,
    height: number,
    channels: number
  ): Float32Array {
    const gray = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
      if (channels === 1) {
        gray[i] = data[i];
      } else if (channels >= 3) {
        const idx = i * channels;
        // Luminance formula
        gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      }
    }

    return gray;
  }

  /**
   * Extract simple embedding from crop region (fallback without ML)
   */
  private extractSimpleEmbedding(
    frame: ImageFrame,
    region?: { x: number; y: number; width: number; height: number }
  ): Float32Array {
    const { data, width, height, channels } = frame;

    // Default to center crop if no region specified
    const cropRegion = region || {
      x: Math.floor(width * 0.25),
      y: Math.floor(height * 0.25),
      width: Math.floor(width * 0.5),
      height: Math.floor(height * 0.5),
    };

    // Extract 8x8 average color grid as simple embedding
    const embedding = new Float32Array(64 * 3);
    const cellWidth = cropRegion.width / 8;
    const cellHeight = cropRegion.height / 8;

    for (let cy = 0; cy < 8; cy++) {
      for (let cx = 0; cx < 8; cx++) {
        const startX = Math.floor(cropRegion.x + cx * cellWidth);
        const startY = Math.floor(cropRegion.y + cy * cellHeight);
        const endX = Math.floor(cropRegion.x + (cx + 1) * cellWidth);
        const endY = Math.floor(cropRegion.y + (cy + 1) * cellHeight);

        let r = 0, g = 0, b = 0, count = 0;

        for (let y = startY; y < endY && y < height; y++) {
          for (let x = startX; x < endX && x < width; x++) {
            const idx = (y * width + x) * channels;
            r += data[idx];
            g += data[idx + 1] || 0;
            b += data[idx + 2] || 0;
            count++;
          }
        }

        const embIdx = (cy * 8 + cx) * 3;
        embedding[embIdx] = count > 0 ? r / count / 255 : 0;
        embedding[embIdx + 1] = count > 0 ? g / count / 255 : 0;
        embedding[embIdx + 2] = count > 0 ? b / count / 255 : 0;
      }
    }

    return embedding;
  }

  /**
   * Compute cosine distance between embeddings
   */
  private computeEmbeddingDistance(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      // Use shorter length for comparison
      const minLen = Math.min(a.length, b.length);
      a = a.slice(0, minLen);
      b = new Float32Array(b.slice(0, minLen));
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) return 1;

    const cosineSimilarity = dotProduct / (normA * normB);

    // Convert to distance (0 = identical, 1 = orthogonal, 2 = opposite)
    return 1 - cosineSimilarity;
  }
}
