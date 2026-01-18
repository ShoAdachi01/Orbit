/**
 * Tests for SegmentationGate
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SegmentationGate, MaskFrame } from '../SegmentationGate';

describe('SegmentationGate', () => {
  let gate: SegmentationGate;

  beforeEach(() => {
    gate = new SegmentationGate();
  });

  describe('analyze', () => {
    it('should return empty metrics for empty mask array', () => {
      const result = gate.analyze([]);

      expect(result.score).toBe(0);
      expect(result.subjectAreaRatios).toEqual([]);
      expect(result.totalFrames).toBe(0);
    });

    it('should compute correct subject area ratio', () => {
      // Create 10x10 mask with 25% subject coverage (25 pixels)
      const mask: MaskFrame = {
        data: new Uint8Array(100),
        width: 10,
        height: 10,
      };

      // Fill top-left quadrant with subject (255)
      for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
          mask.data[y * 10 + x] = 255;
        }
      }

      const result = gate.analyze([mask]);

      expect(result.subjectAreaRatios[0]).toBe(0.25);
      expect(result.totalFrames).toBe(1);
    });

    it('should detect high coverage frames', () => {
      // Create mask with 70% coverage (above 65% threshold)
      const mask: MaskFrame = {
        data: new Uint8Array(100).fill(255), // All subject
        width: 10,
        height: 10,
      };

      // Set 30% to background
      for (let i = 0; i < 30; i++) {
        mask.data[i] = 0;
      }

      const result = gate.analyze([mask]);

      expect(result.subjectAreaRatios[0]).toBe(0.7);
      expect(result.highCoverageFrameCount).toBe(1);
    });

    it('should compute edge jitter between frames', () => {
      // Create two frames with slightly different mask boundaries
      const mask1: MaskFrame = {
        data: new Uint8Array(100),
        width: 10,
        height: 10,
      };
      const mask2: MaskFrame = {
        data: new Uint8Array(100),
        width: 10,
        height: 10,
      };

      // First mask: center square
      for (let y = 3; y < 7; y++) {
        for (let x = 3; x < 7; x++) {
          mask1.data[y * 10 + x] = 255;
        }
      }

      // Second mask: slightly shifted
      for (let y = 4; y < 8; y++) {
        for (let x = 4; x < 8; x++) {
          mask2.data[y * 10 + x] = 255;
        }
      }

      const result = gate.analyze([mask1, mask2]);

      expect(result.edgeJitter).toBeGreaterThan(0);
    });
  });

  describe('checkGate', () => {
    it('should pass for good quality masks', () => {
      const metrics = {
        score: 0.85,
        subjectAreaRatios: [0.3, 0.3, 0.3],
        edgeJitter: 0.1,
        leakScore: 0.05,
        highCoverageFrameCount: 0,
        totalFrames: 3,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(true);
      expect(result.fallbackCandidate).toBeNull();
    });

    it('should flag pose risk for high coverage', () => {
      const metrics = {
        score: 0.8,
        subjectAreaRatios: [0.7, 0.7, 0.7],
        edgeJitter: 0.1,
        leakScore: 0.05,
        highCoverageFrameCount: 3,
        totalFrames: 3,
      };

      const result = gate.checkGate(metrics);

      expect(result.poseRisk).toBe(true);
      expect(result.reasons.length).toBeGreaterThan(0);
      expect(result.reasons.some(r => r.includes('65%'))).toBe(true);
    });

    it('should suggest micro-parallax for high edge jitter', () => {
      const metrics = {
        score: 0.7,
        subjectAreaRatios: [0.3],
        edgeJitter: 0.6,
        leakScore: 0.05,
        highCoverageFrameCount: 0,
        totalFrames: 1,
      };

      const result = gate.checkGate(metrics);

      expect(result.fallbackCandidate).toBe('micro-parallax');
    });

    it('should suggest 2.5d-subject for high leak score', () => {
      const metrics = {
        score: 0.4,
        subjectAreaRatios: [0.3],
        edgeJitter: 0.1,
        leakScore: 0.5,
        highCoverageFrameCount: 0,
        totalFrames: 1,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.fallbackCandidate).toBe('2.5d-subject');
    });
  });
});
