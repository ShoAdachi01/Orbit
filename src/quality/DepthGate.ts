/**
 * Depth Gate (B4 from PRD)
 * Validates depth estimation confidence from DepthCrafter
 */

import { DepthQualityMetrics } from '../schemas/types';

export interface DepthFrame {
  data: Float32Array;
  width: number;
  height: number;
  confidence?: Float32Array;
}

export interface DepthGateConfig {
  /** Global depth weight (default: 0.5) */
  globalDepthWeight: number;
  /** Minimum temporal consistency (default: 0.7) */
  minTemporalConsistency: number;
  /** Minimum edge stability (default: 0.6) */
  minEdgeStability: number;
}

const DEFAULT_CONFIG: DepthGateConfig = {
  globalDepthWeight: 0.5,
  minTemporalConsistency: 0.7,
  minEdgeStability: 0.6,
};

export class DepthGate {
  private config: DepthGateConfig;

  constructor(config: Partial<DepthGateConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze depth sequence and compute quality metrics
   */
  analyze(
    depthFrames: DepthFrame[],
    maskFrames?: Array<{ data: Uint8Array; width: number; height: number }>
  ): DepthQualityMetrics {
    if (depthFrames.length === 0) {
      return this.createEmptyMetrics();
    }

    const frameConfidences = this.computeFrameConfidences(depthFrames);
    const temporalConsistency = this.computeTemporalConsistency(depthFrames);
    const edgeStability = maskFrames
      ? this.computeEdgeStability(depthFrames, maskFrames)
      : 1.0;

    return {
      frameConfidences,
      temporalConsistency,
      edgeStability,
    };
  }

  /**
   * Compute effective depth weight based on confidence
   */
  computeDepthWeight(metrics: DepthQualityMetrics): number {
    const avgConfidence =
      metrics.frameConfidences.reduce((a, b) => a + b, 0) /
      Math.max(1, metrics.frameConfidences.length);

    return Math.min(
      1,
      avgConfidence *
        metrics.temporalConsistency *
        metrics.edgeStability *
        this.config.globalDepthWeight
    );
  }

  /**
   * Check if depth quality is sufficient for reconstruction
   */
  checkGate(metrics: DepthQualityMetrics): {
    passed: boolean;
    depthWeight: number;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;

    if (metrics.temporalConsistency < this.config.minTemporalConsistency) {
      passed = false;
      reasons.push(
        `Temporal consistency ${metrics.temporalConsistency.toFixed(2)} below threshold ${this.config.minTemporalConsistency}`
      );
    }

    if (metrics.edgeStability < this.config.minEdgeStability) {
      passed = false;
      reasons.push(
        `Edge stability ${metrics.edgeStability.toFixed(2)} below threshold ${this.config.minEdgeStability}`
      );
    }

    const depthWeight = this.computeDepthWeight(metrics);

    if (depthWeight < 0.1) {
      reasons.push('Depth weight too low, will rely more on other priors');
    }

    return { passed, depthWeight, reasons };
  }

  /**
   * Compute per-frame confidence scores
   */
  private computeFrameConfidences(frames: DepthFrame[]): number[] {
    return frames.map((frame) => {
      if (frame.confidence) {
        // Use provided confidence map
        const sum = frame.confidence.reduce((a, b) => a + b, 0);
        return sum / frame.confidence.length;
      }

      // Estimate confidence from depth statistics
      return this.estimateConfidenceFromDepth(frame);
    });
  }

  /**
   * Estimate confidence from depth map statistics
   */
  private estimateConfidenceFromDepth(frame: DepthFrame): number {
    const { data, width, height } = frame;

    // Compute depth statistics
    let validCount = 0;
    let sum = 0;
    let sumSq = 0;
    let gradientSum = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const depth = data[idx];

        if (isFinite(depth) && depth > 0) {
          validCount++;
          sum += depth;
          sumSq += depth * depth;

          // Compute gradient magnitude
          if (x > 0 && y > 0) {
            const dx = Math.abs(depth - data[idx - 1]);
            const dy = Math.abs(depth - data[idx - width]);
            gradientSum += Math.sqrt(dx * dx + dy * dy);
          }
        }
      }
    }

    if (validCount < 100) return 0;

    const mean = sum / validCount;
    const variance = sumSq / validCount - mean * mean;
    const avgGradient = gradientSum / validCount;

    // Confidence based on:
    // - High valid pixel ratio
    // - Reasonable variance (not too flat, not too noisy)
    // - Smooth gradients
    const validRatio = validCount / (width * height);
    const varianceScore = variance > 0.01 && variance < 100 ? 1 : 0.5;
    const gradientScore = Math.max(0, 1 - avgGradient * 0.1);

    return validRatio * 0.4 + varianceScore * 0.3 + gradientScore * 0.3;
  }

  /**
   * Compute temporal consistency (depth change smoothness)
   */
  private computeTemporalConsistency(frames: DepthFrame[]): number {
    if (frames.length < 2) return 1;

    const consistencyScores: number[] = [];

    for (let i = 1; i < frames.length; i++) {
      const prev = frames[i - 1];
      const curr = frames[i];

      if (prev.width !== curr.width || prev.height !== curr.height) {
        consistencyScores.push(0);
        continue;
      }

      // Sample pixels for efficiency
      const sampleSize = Math.min(10000, prev.data.length);
      const step = Math.floor(prev.data.length / sampleSize);

      let totalDiff = 0;
      let validCount = 0;

      for (let j = 0; j < prev.data.length; j += step) {
        const d1 = prev.data[j];
        const d2 = curr.data[j];

        if (isFinite(d1) && isFinite(d2) && d1 > 0 && d2 > 0) {
          // Relative depth change
          const relDiff = Math.abs(d2 - d1) / Math.max(d1, d2);
          totalDiff += relDiff;
          validCount++;
        }
      }

      if (validCount === 0) {
        consistencyScores.push(0);
        continue;
      }

      const avgDiff = totalDiff / validCount;
      // Convert to consistency score (0-1)
      const consistency = Math.max(0, 1 - avgDiff * 5);
      consistencyScores.push(consistency);
    }

    return (
      consistencyScores.reduce((a, b) => a + b, 0) / consistencyScores.length
    );
  }

  /**
   * Compute edge stability around subject boundary
   */
  private computeEdgeStability(
    depthFrames: DepthFrame[],
    maskFrames: Array<{ data: Uint8Array; width: number; height: number }>
  ): number {
    if (depthFrames.length !== maskFrames.length) {
      console.warn('Depth and mask frame counts mismatch');
      return 1;
    }

    const stabilityScores: number[] = [];

    for (let i = 0; i < depthFrames.length; i++) {
      const depth = depthFrames[i];
      const mask = maskFrames[i];

      if (depth.width !== mask.width || depth.height !== mask.height) {
        stabilityScores.push(0);
        continue;
      }

      const score = this.computeFrameEdgeStability(depth, mask);
      stabilityScores.push(score);
    }

    return (
      stabilityScores.reduce((a, b) => a + b, 0) / stabilityScores.length
    );
  }

  /**
   * Compute edge stability for a single frame
   */
  private computeFrameEdgeStability(
    depth: DepthFrame,
    mask: { data: Uint8Array; width: number; height: number }
  ): number {
    const { width, height } = depth;
    const edgePixels: number[] = [];
    const neighborDepths: number[] = [];

    // Find edge pixels (boundary between mask and non-mask)
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;

        if (mask.data[idx] > 127) {
          // Check if on edge
          const neighbors = [
            mask.data[idx - 1],
            mask.data[idx + 1],
            mask.data[idx - width],
            mask.data[idx + width],
          ];

          if (neighbors.some((n) => n <= 127)) {
            // Edge pixel
            edgePixels.push(depth.data[idx]);

            // Get background neighbor depths
            const bgNeighborDepths: number[] = [];
            if (mask.data[idx - 1] <= 127)
              bgNeighborDepths.push(depth.data[idx - 1]);
            if (mask.data[idx + 1] <= 127)
              bgNeighborDepths.push(depth.data[idx + 1]);
            if (mask.data[idx - width] <= 127)
              bgNeighborDepths.push(depth.data[idx - width]);
            if (mask.data[idx + width] <= 127)
              bgNeighborDepths.push(depth.data[idx + width]);

            if (bgNeighborDepths.length > 0) {
              neighborDepths.push(
                bgNeighborDepths.reduce((a, b) => a + b, 0) /
                  bgNeighborDepths.length
              );
            }
          }
        }
      }
    }

    if (edgePixels.length === 0 || neighborDepths.length === 0) {
      return 1;
    }

    // Compute depth discontinuity at edges
    // A clear depth separation between subject and background is good
    const avgEdgeDepth =
      edgePixels.reduce((a, b) => a + b, 0) / edgePixels.length;
    const avgNeighborDepth =
      neighborDepths.reduce((a, b) => a + b, 0) / neighborDepths.length;

    const depthSeparation = Math.abs(avgEdgeDepth - avgNeighborDepth);

    // Compute edge smoothness
    const edgeVariance = this.computeVariance(edgePixels);
    const neighborVariance = this.computeVariance(neighborDepths);

    // Good stability: clear separation, low variance
    const separationScore = Math.min(1, depthSeparation * 2);
    const smoothnessScore = Math.max(
      0,
      1 - (edgeVariance + neighborVariance) * 0.5
    );

    return separationScore * 0.5 + smoothnessScore * 0.5;
  }

  private computeVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return (
      values.reduce((sum, v) => sum + (v - mean) * (v - mean), 0) /
      values.length
    );
  }

  private createEmptyMetrics(): DepthQualityMetrics {
    return {
      frameConfidences: [],
      temporalConsistency: 0,
      edgeStability: 0,
    };
  }
}
