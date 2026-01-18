/**
 * Track Gate (B3 from PRD)
 * Validates TAPIR foreground tracking quality
 */

import { TrackQualityMetrics } from '../schemas/types';

export interface Track {
  id: number;
  points: Array<{
    frameIndex: number;
    x: number;
    y: number;
    confidence: number;
  }>;
}

export interface TrackGateConfig {
  /** Minimum confidence threshold (default: 0.6) */
  minConfidence: number;
  /** Minimum lifespan in frames (default: 15) */
  minLifespan: number;
  /** Maximum residual vs local motion model */
  maxMotionResidual: number;
  /** Minimum number of tracks to keep (default: 50) */
  minTracksRequired: number;
}

const DEFAULT_CONFIG: TrackGateConfig = {
  minConfidence: 0.6,
  minLifespan: 15,
  maxMotionResidual: 10.0,
  minTracksRequired: 50,
};

export class TrackGate {
  private config: TrackGateConfig;

  constructor(config: Partial<TrackGateConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Filter tracks and compute quality metrics
   */
  analyze(tracks: Track[]): {
    metrics: TrackQualityMetrics;
    filteredTracks: Track[];
  } {
    if (tracks.length === 0) {
      return {
        metrics: this.createEmptyMetrics(),
        filteredTracks: [],
      };
    }

    const filteredTracks: Track[] = [];
    const discardedReasons: Map<string, number> = new Map();

    for (const track of tracks) {
      const discard = this.shouldDiscardTrack(track);

      if (discard.discard) {
        discardedReasons.set(
          discard.reason,
          (discardedReasons.get(discard.reason) || 0) + 1
        );
      } else {
        filteredTracks.push(track);
      }
    }

    const lifespans = filteredTracks.map(t => t.points.length);
    const confidences = filteredTracks.flatMap(t => t.points.map(p => p.confidence));

    const medianLifespan = this.median(lifespans);
    const medianConfidence = this.median(confidences);

    const metrics: TrackQualityMetrics = {
      numTracksKept: filteredTracks.length,
      numTracksDiscarded: tracks.length - filteredTracks.length,
      medianLifespan,
      medianConfidence,
    };

    return { metrics, filteredTracks };
  }

  /**
   * Check if metrics pass the gate
   */
  checkGate(metrics: TrackQualityMetrics): {
    passed: boolean;
    fallbackCandidate: '2.5d-subject' | null;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;
    let fallbackCandidate: '2.5d-subject' | null = null;

    if (metrics.numTracksKept < this.config.minTracksRequired) {
      passed = false;
      fallbackCandidate = '2.5d-subject';
      reasons.push(
        `Only ${metrics.numTracksKept} tracks kept (minimum: ${this.config.minTracksRequired})`
      );
    }

    if (metrics.medianLifespan < this.config.minLifespan) {
      passed = false;
      fallbackCandidate = '2.5d-subject';
      reasons.push(
        `Median lifespan ${metrics.medianLifespan} frames below threshold ${this.config.minLifespan}`
      );
    }

    if (metrics.medianConfidence < this.config.minConfidence) {
      passed = false;
      fallbackCandidate = '2.5d-subject';
      reasons.push(
        `Median confidence ${metrics.medianConfidence.toFixed(2)} below threshold ${this.config.minConfidence}`
      );
    }

    return { passed, fallbackCandidate, reasons };
  }

  /**
   * Determine if a track should be discarded
   */
  private shouldDiscardTrack(track: Track): { discard: boolean; reason: string } {
    // Check lifespan
    if (track.points.length < this.config.minLifespan) {
      return { discard: true, reason: 'short_lifespan' };
    }

    // Check average confidence
    const avgConfidence =
      track.points.reduce((sum, p) => sum + p.confidence, 0) / track.points.length;
    if (avgConfidence < this.config.minConfidence) {
      return { discard: true, reason: 'low_confidence' };
    }

    // Check motion consistency
    const motionResidual = this.computeMotionResidual(track);
    if (motionResidual > this.config.maxMotionResidual) {
      return { discard: true, reason: 'inconsistent_motion' };
    }

    return { discard: false, reason: '' };
  }

  /**
   * Compute motion residual for a track
   * Measures deviation from a simple motion model
   */
  private computeMotionResidual(track: Track): number {
    if (track.points.length < 3) return 0;

    const residuals: number[] = [];

    for (let i = 2; i < track.points.length; i++) {
      const p0 = track.points[i - 2];
      const p1 = track.points[i - 1];
      const p2 = track.points[i];

      // Predict p2 using constant velocity from p0, p1
      const predictedX = p1.x + (p1.x - p0.x);
      const predictedY = p1.y + (p1.y - p0.y);

      const dx = p2.x - predictedX;
      const dy = p2.y - predictedY;

      residuals.push(Math.sqrt(dx * dx + dy * dy));
    }

    return this.median(residuals);
  }

  /**
   * Group tracks by spatial proximity for local motion analysis
   */
  groupTracksBySpatialProximity(
    tracks: Track[],
    gridSize: number = 50
  ): Map<string, Track[]> {
    const groups = new Map<string, Track[]>();

    for (const track of tracks) {
      if (track.points.length === 0) continue;

      // Use first point for grouping
      const firstPoint = track.points[0];
      const gridX = Math.floor(firstPoint.x / gridSize);
      const gridY = Math.floor(firstPoint.y / gridSize);
      const key = `${gridX},${gridY}`;

      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(track);
    }

    return groups;
  }

  /**
   * Compute local motion consistency within track groups
   */
  computeLocalMotionConsistency(groups: Map<string, Track[]>): number {
    const consistencyScores: number[] = [];

    for (const [_, groupTracks] of groups) {
      if (groupTracks.length < 2) continue;

      // Compare motion vectors within group
      const motionVectors = groupTracks.map(track => this.computeAverageMotion(track));

      // Compute variance of motion vectors
      const avgMotion = {
        x: motionVectors.reduce((s, m) => s + m.x, 0) / motionVectors.length,
        y: motionVectors.reduce((s, m) => s + m.y, 0) / motionVectors.length,
      };

      const variance = motionVectors.reduce((sum, m) => {
        const dx = m.x - avgMotion.x;
        const dy = m.y - avgMotion.y;
        return sum + dx * dx + dy * dy;
      }, 0) / motionVectors.length;

      // Lower variance = higher consistency
      const consistency = 1 / (1 + Math.sqrt(variance));
      consistencyScores.push(consistency);
    }

    return consistencyScores.length > 0
      ? consistencyScores.reduce((a, b) => a + b, 0) / consistencyScores.length
      : 1;
  }

  /**
   * Compute average motion vector for a track
   */
  private computeAverageMotion(track: Track): { x: number; y: number } {
    if (track.points.length < 2) return { x: 0, y: 0 };

    let totalDx = 0;
    let totalDy = 0;

    for (let i = 1; i < track.points.length; i++) {
      totalDx += track.points[i].x - track.points[i - 1].x;
      totalDy += track.points[i].y - track.points[i - 1].y;
    }

    const count = track.points.length - 1;
    return { x: totalDx / count, y: totalDy / count };
  }

  private median(values: number[]): number {
    if (values.length === 0) return 0;

    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  private createEmptyMetrics(): TrackQualityMetrics {
    return {
      numTracksKept: 0,
      numTracksDiscarded: 0,
      medianLifespan: 0,
      medianConfidence: 0,
    };
  }
}
