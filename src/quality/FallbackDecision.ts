/**
 * Fallback Mode Decision System (Section E from PRD)
 * Determines the appropriate rendering mode based on quality gate results
 */

import {
  OrbitMode,
  OrbitBounds,
  MaskQualityMetrics,
  PoseQualityMetrics,
  TrackQualityMetrics,
  DepthQualityMetrics,
  DEFAULT_ORBIT_BOUNDS,
} from '../schemas/types';

import { SegmentationGate } from './SegmentationGate';
import { PoseGate } from './PoseGate';
import { TrackGate } from './TrackGate';
import { DepthGate } from './DepthGate';

export interface FallbackDecisionResult {
  mode: OrbitMode;
  bounds: OrbitBounds;
  reasons: string[];
  gateResults: {
    segmentation: { passed: boolean; reasons: string[] };
    pose: { passed: boolean; reasons: string[] };
    track: { passed: boolean; reasons: string[] };
    depth: { passed: boolean; reasons: string[] };
  };
}

export interface QualityMetrics {
  mask: MaskQualityMetrics;
  pose: PoseQualityMetrics;
  track: TrackQualityMetrics;
  depth: DepthQualityMetrics;
}

/**
 * Mode bounds configurations
 */
const MODE_BOUNDS: Record<OrbitMode, OrbitBounds> = {
  'full-orbit': {
    maxYaw: 20,
    maxPitch: 10,
    maxRoll: 3,
    maxTranslation: 0.10,
    maxTranslationDepthPercent: 2,
    clampToParallax: true,
  },
  'micro-parallax': {
    maxYaw: 8,
    maxPitch: 4,
    maxRoll: 1,
    maxTranslation: 0.03,
    maxTranslationDepthPercent: 1,
    clampToParallax: true,
  },
  '2.5d-subject': {
    maxYaw: 15,
    maxPitch: 8,
    maxRoll: 2,
    maxTranslation: 0.05,
    maxTranslationDepthPercent: 1.5,
    clampToParallax: true,
  },
  'render-only': {
    maxYaw: 0,
    maxPitch: 0,
    maxRoll: 0,
    maxTranslation: 0,
    maxTranslationDepthPercent: 0,
    clampToParallax: true,
  },
};

export class FallbackDecisionEngine {
  private segmentationGate: SegmentationGate;
  private poseGate: PoseGate;
  private trackGate: TrackGate;
  private depthGate: DepthGate;

  constructor() {
    this.segmentationGate = new SegmentationGate();
    this.poseGate = new PoseGate();
    this.trackGate = new TrackGate();
    this.depthGate = new DepthGate();
  }

  /**
   * Decision tree from PRD Section E:
   * 1. Pose fails hard → Render-Only
   * 2. Pose borderline → Micro-Parallax
   * 3. Pose ok but subject unstable → 2.5D Subject
   * 4. Everything ok → Full Orbit
   */
  decide(metrics: QualityMetrics): FallbackDecisionResult {
    const reasons: string[] = [];

    // Check each gate
    const segResult = this.checkSegmentationGate(metrics.mask);
    const poseResult = this.checkPoseGate(metrics.pose);
    const trackResult = this.checkTrackGate(metrics.track);
    const depthResult = this.checkDepthGate(metrics.depth);

    // Decision logic
    let mode: OrbitMode = 'full-orbit';

    // 1. Check if pose fails hard → Render-Only
    if (!poseResult.passed && metrics.pose.score < 0.3) {
      mode = 'render-only';
      reasons.push('Pose estimation failed catastrophically');
      reasons.push(...poseResult.reasons);
    }
    // 2. Check if pose borderline → Micro-Parallax
    else if (!poseResult.passed) {
      mode = 'micro-parallax';
      reasons.push('Pose stability is borderline, using reduced orbit bounds');
      reasons.push(...poseResult.reasons);
    }
    // 3. Check if subject unstable (tracks/depth) → 2.5D Subject
    else if (!trackResult.passed || !depthResult.passed) {
      mode = '2.5d-subject';
      if (!trackResult.passed) {
        reasons.push('Foreground tracking insufficient for full 4D reconstruction');
        reasons.push(...trackResult.reasons);
      }
      if (!depthResult.passed) {
        reasons.push('Depth estimation unreliable for full reconstruction');
        reasons.push(...depthResult.reasons);
      }
    }
    // 4. Check segmentation issues
    else if (!segResult.passed) {
      // Segmentation issues might trigger 2.5D or micro-parallax
      if (segResult.poseRisk) {
        mode = 'micro-parallax';
        reasons.push('Subject dominates frame, reduced orbit bounds');
      } else if (segResult.fallbackCandidate) {
        mode = segResult.fallbackCandidate;
      }
      reasons.push(...segResult.reasons);
    }
    // 5. Everything passed → Full Orbit
    else {
      mode = 'full-orbit';
      reasons.push('All quality gates passed');
    }

    return {
      mode,
      bounds: MODE_BOUNDS[mode],
      reasons,
      gateResults: {
        segmentation: { passed: segResult.passed, reasons: segResult.reasons },
        pose: { passed: poseResult.passed, reasons: poseResult.reasons },
        track: { passed: trackResult.passed, reasons: trackResult.reasons },
        depth: { passed: depthResult.passed, reasons: depthResult.reasons },
      },
    };
  }

  /**
   * Get bounds for a specific mode
   */
  getBoundsForMode(mode: OrbitMode): OrbitBounds {
    return { ...MODE_BOUNDS[mode] };
  }

  /**
   * Get mode label for user display
   */
  getModeLabel(mode: OrbitMode): string {
    const labels: Record<OrbitMode, string> = {
      'full-orbit': 'Full Orbit',
      'micro-parallax': 'Micro-Parallax',
      '2.5d-subject': '2.5D Subject',
      'render-only': 'Render-Only',
    };
    return labels[mode];
  }

  /**
   * Get mode description for user display
   */
  getModeDescription(mode: OrbitMode): string {
    const descriptions: Record<OrbitMode, string> = {
      'full-orbit': 'Full parallax effect with maximum orbit range',
      'micro-parallax': 'Reduced orbit range for stable parallax',
      '2.5d-subject': 'Subject rendered as billboard with depth',
      'render-only': 'No interactive 3D, stabilized export only',
    };
    return descriptions[mode];
  }

  private checkSegmentationGate(metrics: MaskQualityMetrics): {
    passed: boolean;
    poseRisk: boolean;
    fallbackCandidate: 'micro-parallax' | '2.5d-subject' | null;
    reasons: string[];
  } {
    // Reconstruct check from metrics
    const reasons: string[] = [];
    let passed = true;
    let poseRisk = false;
    let fallbackCandidate: 'micro-parallax' | '2.5d-subject' | null = null;

    if (metrics.score < 0.5) {
      passed = false;
      reasons.push(`Mask quality score ${metrics.score.toFixed(2)} below threshold`);
    }

    const highCoveragePercent = metrics.highCoverageFrameCount / metrics.totalFrames;
    if (highCoveragePercent > 0.3) {
      poseRisk = true;
      reasons.push(`High subject coverage in ${(highCoveragePercent * 100).toFixed(1)}% of frames`);
    }

    if (metrics.edgeJitter > 0.5) {
      fallbackCandidate = 'micro-parallax';
      reasons.push(`Edge jitter ${metrics.edgeJitter.toFixed(2)} above threshold`);
    }

    if (metrics.leakScore > 0.3) {
      passed = false;
      fallbackCandidate = '2.5d-subject';
      reasons.push(`Leak score ${metrics.leakScore.toFixed(2)} above threshold`);
    }

    return { passed, poseRisk, fallbackCandidate, reasons };
  }

  private checkPoseGate(metrics: PoseQualityMetrics): {
    passed: boolean;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;

    if (metrics.inlierRatio < 0.35) {
      passed = false;
      reasons.push(`Inlier ratio ${metrics.inlierRatio.toFixed(2)} below 0.35`);
    }

    if (metrics.medianReprojectionError > 2.0) {
      passed = false;
      reasons.push(`Reprojection error ${metrics.medianReprojectionError.toFixed(2)}px above 2.0px`);
    }

    if (metrics.goodCoverageFramePercent < 0.6) {
      passed = false;
      reasons.push(`Quadrant coverage ${(metrics.goodCoverageFramePercent * 100).toFixed(1)}% below 60%`);
    }

    if (metrics.jitterScore > 0.5) {
      passed = false;
      reasons.push(`Jitter score ${metrics.jitterScore.toFixed(2)} above 0.5`);
    }

    return { passed, reasons };
  }

  private checkTrackGate(metrics: TrackQualityMetrics): {
    passed: boolean;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;

    if (metrics.numTracksKept < 50) {
      passed = false;
      reasons.push(`Only ${metrics.numTracksKept} tracks kept (minimum 50)`);
    }

    if (metrics.medianLifespan < 15) {
      passed = false;
      reasons.push(`Median track lifespan ${metrics.medianLifespan} below 15 frames`);
    }

    if (metrics.medianConfidence < 0.6) {
      passed = false;
      reasons.push(`Median track confidence ${metrics.medianConfidence.toFixed(2)} below 0.6`);
    }

    return { passed, reasons };
  }

  private checkDepthGate(metrics: DepthQualityMetrics): {
    passed: boolean;
    reasons: string[];
  } {
    const reasons: string[] = [];
    let passed = true;

    if (metrics.temporalConsistency < 0.7) {
      passed = false;
      reasons.push(`Temporal consistency ${metrics.temporalConsistency.toFixed(2)} below 0.7`);
    }

    if (metrics.edgeStability < 0.6) {
      passed = false;
      reasons.push(`Edge stability ${metrics.edgeStability.toFixed(2)} below 0.6`);
    }

    return { passed, reasons };
  }
}
