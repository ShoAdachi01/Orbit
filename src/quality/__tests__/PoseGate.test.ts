/**
 * Tests for PoseGate
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { PoseGate, PoseEstimate } from '../PoseGate';
import { Matrix4x4 } from '../../schemas/types';

describe('PoseGate', () => {
  let gate: PoseGate;

  beforeEach(() => {
    gate = new PoseGate();
  });

  const createIdentityTransform = (): Matrix4x4 => [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];

  const createPoseEstimate = (
    inliers: number,
    total: number,
    errors: number[],
    tracks: Array<{ x: number; y: number }>,
    transform: Matrix4x4 = createIdentityTransform()
  ): PoseEstimate => ({
    pose: {
      transform,
      timestamp: 0,
      frameIndex: 0,
    },
    inliers,
    totalMatches: total,
    reprojectionErrors: errors,
    trackPositions: tracks,
  });

  describe('analyze', () => {
    it('should return empty metrics for empty estimates', () => {
      const result = gate.analyze([], 1920, 1080);

      expect(result.score).toBe(0);
      expect(result.inlierRatio).toBe(0);
      expect(result.jitterScore).toBe(1);
    });

    it('should compute correct inlier ratio', () => {
      const estimates = [
        createPoseEstimate(70, 100, [1.0], []),
        createPoseEstimate(80, 100, [1.0], []),
      ];

      const result = gate.analyze(estimates, 1920, 1080);

      expect(result.inlierRatio).toBe(0.75);
    });

    it('should compute median reprojection error', () => {
      const estimates = [
        createPoseEstimate(100, 100, [1.0, 2.0, 3.0], []),
        createPoseEstimate(100, 100, [1.5, 2.5], []),
      ];

      const result = gate.analyze(estimates, 1920, 1080);

      // Median of [1.0, 1.5, 2.0, 2.5, 3.0] = 2.0
      expect(result.medianReprojectionError).toBe(2.0);
    });

    it('should compute quadrant coverage', () => {
      const estimates = [
        createPoseEstimate(100, 100, [1.0], [
          { x: 100, y: 100 },   // Quadrant 0 (top-left)
          { x: 1000, y: 100 },  // Quadrant 1 (top-right)
          { x: 100, y: 600 },   // Quadrant 2 (bottom-left)
          { x: 1000, y: 600 },  // Quadrant 3 (bottom-right)
        ]),
      ];

      const result = gate.analyze(estimates, 1920, 1080);

      expect(result.quadrantCoverage).toEqual([true, true, true, true]);
      expect(result.goodCoverageFramePercent).toBe(1.0);
    });

    it('should detect poor quadrant coverage', () => {
      const estimates = [
        createPoseEstimate(100, 100, [1.0], [
          { x: 100, y: 100 },  // Only top-left quadrant
          { x: 200, y: 200 },
        ]),
      ];

      const result = gate.analyze(estimates, 1920, 1080);

      expect(result.goodCoverageFramePercent).toBe(0);
    });
  });

  describe('checkGate', () => {
    it('should pass for good quality poses', () => {
      const metrics = {
        score: 0.85,
        inlierRatio: 0.7,
        medianReprojectionError: 1.2,
        quadrantCoverage: [true, true, true, true] as [boolean, boolean, boolean, boolean],
        goodCoverageFramePercent: 0.9,
        jitterScore: 0.2,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(true);
      expect(result.fallbackMode).toBeNull();
    });

    it('should fail and suggest micro-parallax for borderline quality', () => {
      const metrics = {
        score: 0.5,
        inlierRatio: 0.3,  // Below 0.35 threshold
        medianReprojectionError: 2.5,  // Above 2.0 threshold
        quadrantCoverage: [true, true, true, false] as [boolean, boolean, boolean, boolean],
        goodCoverageFramePercent: 0.5,  // Below 0.6 threshold
        jitterScore: 0.3,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.fallbackMode).toBe('micro-parallax');
      expect(result.reasons.length).toBeGreaterThan(0);
    });

    it('should suggest render-only for catastrophic pose failure', () => {
      const metrics = {
        score: 0.1,  // Very low score
        inlierRatio: 0.1,
        medianReprojectionError: 10.0,
        quadrantCoverage: [true, false, false, false] as [boolean, boolean, boolean, boolean],
        goodCoverageFramePercent: 0.1,
        jitterScore: 0.9,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.fallbackMode).toBe('render-only');
    });
  });
});
