/**
 * Pose Gate (B2 from PRD)
 * Validates camera pose estimation quality
 */

import { PoseQualityMetrics, CameraPose, Matrix4x4 } from '../schemas/types';

export interface PoseEstimate {
  pose: CameraPose;
  inliers: number;
  totalMatches: number;
  reprojectionErrors: number[];
  trackPositions: Array<{ x: number; y: number }>;
}

export interface PoseGateConfig {
  /** Minimum inlier ratio (default: 0.35) */
  minInlierRatio: number;
  /** Maximum median reprojection error in pixels (default: 2.0) */
  maxReprojectionError: number;
  /** Minimum quadrant coverage percentage (default: 0.60) */
  minQuadrantCoveragePercent: number;
  /** Maximum jitter score (default: 0.5) */
  maxJitterScore: number;
}

const DEFAULT_CONFIG: PoseGateConfig = {
  minInlierRatio: 0.35,
  maxReprojectionError: 2.0,
  minQuadrantCoveragePercent: 0.60,
  maxJitterScore: 0.5,
};

export class PoseGate {
  private config: PoseGateConfig;

  constructor(config: Partial<PoseGateConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze pose sequence and compute quality metrics
   */
  analyze(
    estimates: PoseEstimate[],
    imageWidth: number,
    imageHeight: number
  ): PoseQualityMetrics {
    if (estimates.length === 0) {
      return this.createEmptyMetrics();
    }

    const inlierRatio = this.computeAverageInlierRatio(estimates);
    const medianReprojectionError = this.computeMedianReprojectionError(estimates);
    const { quadrantCoverage, goodCoveragePercent } = this.computeQuadrantCoverage(
      estimates,
      imageWidth,
      imageHeight
    );
    const jitterScore = this.computeJitterScore(estimates);

    const score = this.computeOverallScore(
      inlierRatio,
      medianReprojectionError,
      goodCoveragePercent,
      jitterScore
    );

    return {
      score,
      inlierRatio,
      medianReprojectionError,
      quadrantCoverage,
      goodCoverageFramePercent: goodCoveragePercent,
      jitterScore,
    };
  }

  /**
   * Check if metrics pass the gate
   */
  checkGate(metrics: PoseQualityMetrics): {
    passed: boolean;
    fallbackMode: 'micro-parallax' | 'render-only' | null;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;
    let fallbackMode: 'micro-parallax' | 'render-only' | null = null;

    // Check inlier ratio
    if (metrics.inlierRatio < this.config.minInlierRatio) {
      passed = false;
      reasons.push(`Inlier ratio ${metrics.inlierRatio.toFixed(2)} below threshold ${this.config.minInlierRatio}`);
    }

    // Check reprojection error
    if (metrics.medianReprojectionError > this.config.maxReprojectionError) {
      passed = false;
      reasons.push(`Median reprojection error ${metrics.medianReprojectionError.toFixed(2)}px exceeds threshold ${this.config.maxReprojectionError}px`);
    }

    // Check quadrant coverage
    if (metrics.goodCoverageFramePercent < this.config.minQuadrantCoveragePercent) {
      passed = false;
      reasons.push(`Quadrant coverage ${(metrics.goodCoverageFramePercent * 100).toFixed(1)}% below threshold ${this.config.minQuadrantCoveragePercent * 100}%`);
    }

    // Check jitter
    if (metrics.jitterScore > this.config.maxJitterScore) {
      passed = false;
      reasons.push(`Jitter score ${metrics.jitterScore.toFixed(2)} exceeds threshold ${this.config.maxJitterScore}`);
    }

    // Determine fallback mode based on severity
    if (!passed) {
      if (metrics.score < 0.3) {
        fallbackMode = 'render-only';
        reasons.push('Pose quality too low for any parallax effect');
      } else {
        fallbackMode = 'micro-parallax';
        reasons.push('Reduced orbit bounds due to pose instability');
      }
    }

    return { passed, fallbackMode, reasons };
  }

  /**
   * Compute average inlier ratio across all frames
   */
  private computeAverageInlierRatio(estimates: PoseEstimate[]): number {
    const ratios = estimates.map(e => e.inliers / Math.max(1, e.totalMatches));
    return ratios.reduce((a, b) => a + b, 0) / ratios.length;
  }

  /**
   * Compute median reprojection error across all frames
   */
  private computeMedianReprojectionError(estimates: PoseEstimate[]): number {
    const allErrors: number[] = [];
    for (const estimate of estimates) {
      allErrors.push(...estimate.reprojectionErrors);
    }

    if (allErrors.length === 0) return 0;

    allErrors.sort((a, b) => a - b);
    const mid = Math.floor(allErrors.length / 2);

    return allErrors.length % 2 === 0
      ? (allErrors[mid - 1] + allErrors[mid]) / 2
      : allErrors[mid];
  }

  /**
   * Compute quadrant coverage metrics
   */
  private computeQuadrantCoverage(
    estimates: PoseEstimate[],
    imageWidth: number,
    imageHeight: number
  ): {
    quadrantCoverage: [boolean, boolean, boolean, boolean];
    goodCoveragePercent: number;
  } {
    const midX = imageWidth / 2;
    const midY = imageHeight / 2;

    let goodFrames = 0;

    // Overall quadrant coverage (any frame)
    const overallQuadrants = [false, false, false, false];

    for (const estimate of estimates) {
      const frameQuadrants = [false, false, false, false];

      for (const track of estimate.trackPositions) {
        const quadrant = (track.x >= midX ? 1 : 0) + (track.y >= midY ? 2 : 0);
        frameQuadrants[quadrant] = true;
        overallQuadrants[quadrant] = true;
      }

      // Count quadrants populated in this frame
      const populatedQuadrants = frameQuadrants.filter(Boolean).length;
      if (populatedQuadrants >= 3) {
        goodFrames++;
      }
    }

    return {
      quadrantCoverage: overallQuadrants as [boolean, boolean, boolean, boolean],
      goodCoveragePercent: goodFrames / estimates.length,
    };
  }

  /**
   * Compute pose jitter score (high-frequency energy in pose changes)
   */
  private computeJitterScore(estimates: PoseEstimate[]): number {
    if (estimates.length < 3) return 0;

    const rotationJitter: number[] = [];
    const translationJitter: number[] = [];

    for (let i = 2; i < estimates.length; i++) {
      const prev2 = estimates[i - 2].pose.transform;
      const prev1 = estimates[i - 1].pose.transform;
      const curr = estimates[i].pose.transform;

      // Compute second derivative (acceleration) as jitter metric
      const deltaR1 = this.rotationDelta(prev2, prev1);
      const deltaR2 = this.rotationDelta(prev1, curr);
      const deltaT1 = this.translationDelta(prev2, prev1);
      const deltaT2 = this.translationDelta(prev1, curr);

      rotationJitter.push(Math.abs(deltaR2 - deltaR1));
      translationJitter.push(Math.abs(deltaT2 - deltaT1));
    }

    // Normalize and combine
    const avgRotJitter = rotationJitter.reduce((a, b) => a + b, 0) / rotationJitter.length;
    const avgTransJitter = translationJitter.reduce((a, b) => a + b, 0) / translationJitter.length;

    // Scale to 0-1 range (empirical thresholds)
    const normalizedRotJitter = Math.min(1, avgRotJitter * 10);
    const normalizedTransJitter = Math.min(1, avgTransJitter * 100);

    return (normalizedRotJitter + normalizedTransJitter) / 2;
  }

  /**
   * Compute rotation delta between two poses (in radians)
   */
  private rotationDelta(m1: Matrix4x4, m2: Matrix4x4): number {
    // Extract rotation matrices (upper-left 3x3)
    const r1 = [m1[0], m1[1], m1[2], m1[4], m1[5], m1[6], m1[8], m1[9], m1[10]];
    const r2 = [m2[0], m2[1], m2[2], m2[4], m2[5], m2[6], m2[8], m2[9], m2[10]];

    // R_delta = R2 * R1^T
    const trace = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2] +
                  r1[3] * r2[3] + r1[4] * r2[4] + r1[5] * r2[5] +
                  r1[6] * r2[6] + r1[7] * r2[7] + r1[8] * r2[8];

    // angle = acos((trace - 1) / 2)
    const cosAngle = (trace - 1) / 2;
    return Math.acos(Math.max(-1, Math.min(1, cosAngle)));
  }

  /**
   * Compute translation delta between two poses (in meters)
   */
  private translationDelta(m1: Matrix4x4, m2: Matrix4x4): number {
    const dx = m2[12] - m1[12];
    const dy = m2[13] - m1[13];
    const dz = m2[14] - m1[14];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Compute overall quality score
   */
  private computeOverallScore(
    inlierRatio: number,
    reprojectionError: number,
    coveragePercent: number,
    jitterScore: number
  ): number {
    // Normalize each metric to 0-1 (higher is better)
    const inlierScore = Math.min(1, inlierRatio / this.config.minInlierRatio);
    const reprojScore = Math.max(0, 1 - reprojectionError / (this.config.maxReprojectionError * 2));
    const coverageScore = coveragePercent / this.config.minQuadrantCoveragePercent;
    const stabilityScore = Math.max(0, 1 - jitterScore);

    // Weighted average
    return (
      inlierScore * 0.25 +
      reprojScore * 0.25 +
      coverageScore * 0.25 +
      stabilityScore * 0.25
    );
  }

  private createEmptyMetrics(): PoseQualityMetrics {
    return {
      score: 0,
      inlierRatio: 0,
      medianReprojectionError: Infinity,
      quadrantCoverage: [false, false, false, false],
      goodCoverageFramePercent: 0,
      jitterScore: 1,
    };
  }
}
