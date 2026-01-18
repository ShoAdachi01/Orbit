/**
 * Segmentation Gate (B1 from PRD)
 * Validates mask quality and computes metrics
 */

import { MaskQualityMetrics } from '../schemas/types';

export interface MaskFrame {
  data: Uint8Array;
  width: number;
  height: number;
}

export interface SegmentationGateConfig {
  /** Subject coverage threshold (default: 0.65) */
  maxSubjectCoverage: number;
  /** Frame percentage threshold for high coverage (default: 0.30) */
  highCoverageFramePercent: number;
  /** Edge jitter threshold */
  maxEdgeJitter: number;
  /** Leak score threshold */
  maxLeakScore: number;
}

const DEFAULT_CONFIG: SegmentationGateConfig = {
  maxSubjectCoverage: 0.65,
  highCoverageFramePercent: 0.30,
  maxEdgeJitter: 0.5,
  maxLeakScore: 0.3,
};

export class SegmentationGate {
  private config: SegmentationGateConfig;

  constructor(config: Partial<SegmentationGateConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze mask sequence and compute quality metrics
   */
  analyze(masks: MaskFrame[]): MaskQualityMetrics {
    if (masks.length === 0) {
      return this.createEmptyMetrics();
    }

    const subjectAreaRatios = masks.map(mask => this.computeSubjectAreaRatio(mask));
    const edgeJitter = this.computeEdgeJitter(masks);
    const leakScore = this.computeLeakScore(masks);

    const highCoverageCount = subjectAreaRatios.filter(
      ratio => ratio > this.config.maxSubjectCoverage
    ).length;

    const score = this.computeOverallScore(
      subjectAreaRatios,
      edgeJitter,
      leakScore,
      highCoverageCount
    );

    return {
      score,
      subjectAreaRatios,
      edgeJitter,
      leakScore,
      highCoverageFrameCount: highCoverageCount,
      totalFrames: masks.length,
    };
  }

  /**
   * Check if metrics pass the gate
   */
  checkGate(metrics: MaskQualityMetrics): {
    passed: boolean;
    poseRisk: boolean;
    fallbackCandidate: 'micro-parallax' | '2.5d-subject' | null;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;
    let poseRisk = false;
    let fallbackCandidate: 'micro-parallax' | '2.5d-subject' | null = null;

    // Check high coverage frames
    const highCoveragePercent = metrics.highCoverageFrameCount / metrics.totalFrames;
    if (highCoveragePercent > this.config.highCoverageFramePercent) {
      poseRisk = true;
      reasons.push(`Subject covers >65% frame in ${(highCoveragePercent * 100).toFixed(1)}% of frames (threshold: ${this.config.highCoverageFramePercent * 100}%)`);
    }

    // Check edge jitter
    if (metrics.edgeJitter > this.config.maxEdgeJitter) {
      fallbackCandidate = fallbackCandidate || 'micro-parallax';
      reasons.push(`Edge jitter ${metrics.edgeJitter.toFixed(2)} exceeds threshold ${this.config.maxEdgeJitter}`);
    }

    // Check leak score
    if (metrics.leakScore > this.config.maxLeakScore) {
      passed = false;
      fallbackCandidate = '2.5d-subject';
      reasons.push(`Leak score ${metrics.leakScore.toFixed(2)} exceeds threshold ${this.config.maxLeakScore}`);
    }

    // Check overall score
    if (metrics.score < 0.5) {
      passed = false;
      reasons.push(`Overall mask score ${metrics.score.toFixed(2)} below 0.5`);
    }

    return { passed, poseRisk, fallbackCandidate, reasons };
  }

  /**
   * Compute subject area ratio for a single frame
   */
  private computeSubjectAreaRatio(mask: MaskFrame): number {
    const totalPixels = mask.width * mask.height;
    let subjectPixels = 0;

    for (let i = 0; i < mask.data.length; i++) {
      if (mask.data[i] > 127) {
        subjectPixels++;
      }
    }

    return subjectPixels / totalPixels;
  }

  /**
   * Compute edge jitter across mask sequence
   * Measures mask boundary displacement over time
   */
  private computeEdgeJitter(masks: MaskFrame[]): number {
    if (masks.length < 2) return 0;

    let totalJitter = 0;
    let comparisonCount = 0;

    for (let i = 1; i < masks.length; i++) {
      const prevEdge = this.extractEdge(masks[i - 1]);
      const currEdge = this.extractEdge(masks[i]);

      const displacement = this.computeEdgeDisplacement(prevEdge, currEdge);
      totalJitter += displacement;
      comparisonCount++;
    }

    // Normalize by frame dimensions
    const avgDimension = (masks[0].width + masks[0].height) / 2;
    return totalJitter / (comparisonCount * avgDimension);
  }

  /**
   * Compute leak score (high-frequency fragments outside main component)
   */
  private computeLeakScore(masks: MaskFrame[]): number {
    let totalLeakScore = 0;

    for (const mask of masks) {
      const components = this.findConnectedComponents(mask);

      if (components.length <= 1) continue;

      // Sort by size
      components.sort((a, b) => b.pixelCount - a.pixelCount);

      // Main component is largest
      const mainComponent = components[0];
      const totalMaskPixels = components.reduce((sum, c) => sum + c.pixelCount, 0);

      // Leak = pixels not in main component
      const leakPixels = totalMaskPixels - mainComponent.pixelCount;

      // Score based on leak percentage and fragment count
      const leakPercent = leakPixels / totalMaskPixels;
      const fragmentPenalty = Math.min(1, (components.length - 1) * 0.1);

      totalLeakScore += leakPercent * (1 + fragmentPenalty);
    }

    return totalLeakScore / masks.length;
  }

  /**
   * Compute overall quality score
   */
  private computeOverallScore(
    subjectAreaRatios: number[],
    edgeJitter: number,
    leakScore: number,
    highCoverageCount: number
  ): number {
    // Component scores (0-1, higher is better)
    const avgCoverage = subjectAreaRatios.reduce((a, b) => a + b, 0) / subjectAreaRatios.length;
    const coverageScore = avgCoverage > 0.1 && avgCoverage < 0.5 ? 1 : Math.max(0, 1 - Math.abs(avgCoverage - 0.3) * 2);

    const jitterScore = Math.max(0, 1 - edgeJitter * 2);
    const leakageScore = Math.max(0, 1 - leakScore * 3);

    const highCoveragePercent = highCoverageCount / subjectAreaRatios.length;
    const coverageConsistencyScore = Math.max(0, 1 - highCoveragePercent * 2);

    // Weighted average
    return (
      coverageScore * 0.2 +
      jitterScore * 0.3 +
      leakageScore * 0.3 +
      coverageConsistencyScore * 0.2
    );
  }

  /**
   * Extract edge pixels from mask
   */
  private extractEdge(mask: MaskFrame): Set<number> {
    const edge = new Set<number>();
    const { width, height, data } = mask;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (data[idx] > 127) {
          // Check 4-neighborhood
          const neighbors = [
            data[idx - 1],      // left
            data[idx + 1],      // right
            data[idx - width],  // top
            data[idx + width],  // bottom
          ];

          // Edge if any neighbor is background
          if (neighbors.some(n => n <= 127)) {
            edge.add(idx);
          }
        }
      }
    }

    return edge;
  }

  /**
   * Compute edge displacement between two frames
   */
  private computeEdgeDisplacement(prevEdge: Set<number>, currEdge: Set<number>): number {
    // Simple approximation: count pixels that changed edge status
    const added = [...currEdge].filter(p => !prevEdge.has(p)).length;
    const removed = [...prevEdge].filter(p => !currEdge.has(p)).length;
    return added + removed;
  }

  /**
   * Find connected components in mask
   */
  private findConnectedComponents(mask: MaskFrame): { pixelCount: number }[] {
    const { width, height, data } = mask;
    const visited = new Set<number>();
    const components: { pixelCount: number }[] = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;

        if (data[idx] > 127 && !visited.has(idx)) {
          // BFS to find connected component
          const component = this.bfsComponent(mask, idx, visited);
          components.push({ pixelCount: component.size });
        }
      }
    }

    return components;
  }

  /**
   * BFS to find connected component starting from seed
   */
  private bfsComponent(mask: MaskFrame, seed: number, visited: Set<number>): Set<number> {
    const component = new Set<number>();
    const queue = [seed];
    const { width, height, data } = mask;

    while (queue.length > 0) {
      const idx = queue.shift()!;

      if (visited.has(idx)) continue;
      if (data[idx] <= 127) continue;

      visited.add(idx);
      component.add(idx);

      const x = idx % width;
      const y = Math.floor(idx / width);

      // Add 4-neighbors
      if (x > 0) queue.push(idx - 1);
      if (x < width - 1) queue.push(idx + 1);
      if (y > 0) queue.push(idx - width);
      if (y < height - 1) queue.push(idx + width);
    }

    return component;
  }

  private createEmptyMetrics(): MaskQualityMetrics {
    return {
      score: 0,
      subjectAreaRatios: [],
      edgeJitter: 0,
      leakScore: 0,
      highCoverageFrameCount: 0,
      totalFrames: 0,
    };
  }
}
