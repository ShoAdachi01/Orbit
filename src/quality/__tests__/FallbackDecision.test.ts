/**
 * Tests for FallbackDecisionEngine
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { FallbackDecisionEngine, QualityMetrics } from '../FallbackDecision';

describe('FallbackDecisionEngine', () => {
  let engine: FallbackDecisionEngine;

  beforeEach(() => {
    engine = new FallbackDecisionEngine();
  });

  const createGoodMetrics = (): QualityMetrics => ({
    mask: {
      score: 0.9,
      subjectAreaRatios: [0.3, 0.3, 0.3],
      edgeJitter: 0.1,
      leakScore: 0.05,
      highCoverageFrameCount: 0,
      totalFrames: 3,
    },
    pose: {
      score: 0.85,
      inlierRatio: 0.7,
      medianReprojectionError: 1.2,
      quadrantCoverage: [true, true, true, true],
      goodCoverageFramePercent: 0.9,
      jitterScore: 0.2,
    },
    track: {
      numTracksKept: 150,
      numTracksDiscarded: 20,
      medianLifespan: 45,
      medianConfidence: 0.75,
    },
    depth: {
      frameConfidences: [0.8, 0.8, 0.8],
      temporalConsistency: 0.85,
      edgeStability: 0.8,
    },
  });

  describe('decide', () => {
    it('should return full-orbit for all passing metrics', () => {
      const metrics = createGoodMetrics();
      const result = engine.decide(metrics);

      expect(result.mode).toBe('full-orbit');
      expect(result.reasons).toContain('All quality gates passed');
    });

    it('should return render-only for catastrophic pose failure', () => {
      const metrics = createGoodMetrics();
      metrics.pose.score = 0.1;
      metrics.pose.inlierRatio = 0.1;
      metrics.pose.jitterScore = 0.9;

      const result = engine.decide(metrics);

      expect(result.mode).toBe('render-only');
      expect(result.gateResults.pose.passed).toBe(false);
    });

    it('should return micro-parallax for borderline pose', () => {
      const metrics = createGoodMetrics();
      metrics.pose.score = 0.5;
      metrics.pose.inlierRatio = 0.3;
      metrics.pose.jitterScore = 0.6;

      const result = engine.decide(metrics);

      expect(result.mode).toBe('micro-parallax');
    });

    it('should return 2.5d-subject for poor tracking', () => {
      const metrics = createGoodMetrics();
      metrics.track.numTracksKept = 30;  // Below 50 threshold
      metrics.track.medianLifespan = 10;  // Below 15 threshold

      const result = engine.decide(metrics);

      expect(result.mode).toBe('2.5d-subject');
      expect(result.gateResults.track.passed).toBe(false);
    });

    it('should return 2.5d-subject for poor depth', () => {
      const metrics = createGoodMetrics();
      metrics.depth.temporalConsistency = 0.5;  // Below 0.7 threshold
      metrics.depth.edgeStability = 0.4;  // Below 0.6 threshold

      const result = engine.decide(metrics);

      expect(result.mode).toBe('2.5d-subject');
      expect(result.gateResults.depth.passed).toBe(false);
    });

    it('should return appropriate bounds for each mode', () => {
      // Full orbit
      const fullOrbitResult = engine.decide(createGoodMetrics());
      expect(fullOrbitResult.bounds.maxYaw).toBe(20);
      expect(fullOrbitResult.bounds.maxPitch).toBe(10);

      // Micro parallax
      const microMetrics = createGoodMetrics();
      microMetrics.pose.inlierRatio = 0.3;
      const microResult = engine.decide(microMetrics);
      expect(microResult.bounds.maxYaw).toBe(8);
      expect(microResult.bounds.maxPitch).toBe(4);

      // Render only
      const renderMetrics = createGoodMetrics();
      renderMetrics.pose.score = 0.1;
      renderMetrics.pose.inlierRatio = 0.1;
      const renderResult = engine.decide(renderMetrics);
      expect(renderResult.bounds.maxYaw).toBe(0);
      expect(renderResult.bounds.maxPitch).toBe(0);
    });
  });

  describe('getModeLabel', () => {
    it('should return correct labels', () => {
      expect(engine.getModeLabel('full-orbit')).toBe('Full Orbit');
      expect(engine.getModeLabel('micro-parallax')).toBe('Micro-Parallax');
      expect(engine.getModeLabel('2.5d-subject')).toBe('2.5D Subject');
      expect(engine.getModeLabel('render-only')).toBe('Render-Only');
    });
  });

  describe('getModeDescription', () => {
    it('should return descriptions for all modes', () => {
      expect(engine.getModeDescription('full-orbit')).toContain('parallax');
      expect(engine.getModeDescription('micro-parallax')).toContain('Reduced');
      expect(engine.getModeDescription('2.5d-subject')).toContain('billboard');
      expect(engine.getModeDescription('render-only')).toContain('No interactive');
    });
  });
});
