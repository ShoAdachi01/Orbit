/**
 * Tests for DepthGate
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DepthGate, DepthFrame } from '../DepthGate';

describe('DepthGate', () => {
  let gate: DepthGate;

  beforeEach(() => {
    gate = new DepthGate();
  });

  const createDepthFrame = (
    width: number,
    height: number,
    baseDepth: number,
    noise: number = 0
  ): DepthFrame => {
    const data = new Float32Array(width * height);
    for (let i = 0; i < data.length; i++) {
      data[i] = baseDepth + (Math.random() - 0.5) * noise;
    }
    return { data, width, height };
  };

  const createMaskFrame = (
    width: number,
    height: number,
    subjectCoverage: number
  ): { data: Uint8Array; width: number; height: number } => {
    const data = new Uint8Array(width * height);
    const subjectPixels = Math.floor(width * height * subjectCoverage);

    // Fill center with subject
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.sqrt(subjectPixels / Math.PI);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dx = x - centerX;
        const dy = y - centerY;
        if (dx * dx + dy * dy < radius * radius) {
          data[y * width + x] = 255;
        }
      }
    }

    return { data, width, height };
  };

  describe('analyze', () => {
    it('should return empty metrics for empty frames', () => {
      const result = gate.analyze([]);

      expect(result.frameConfidences).toEqual([]);
      expect(result.temporalConsistency).toBe(0);
      expect(result.edgeStability).toBe(0);
    });

    it('should compute frame confidences', () => {
      const frames = [
        createDepthFrame(100, 100, 5, 0.1),
        createDepthFrame(100, 100, 5, 0.1),
        createDepthFrame(100, 100, 5, 0.1),
      ];

      const result = gate.analyze(frames);

      expect(result.frameConfidences.length).toBe(3);
      result.frameConfidences.forEach((conf) => {
        expect(conf).toBeGreaterThan(0);
        expect(conf).toBeLessThanOrEqual(1);
      });
    });

    it('should compute high temporal consistency for stable depth', () => {
      const frames = [
        createDepthFrame(100, 100, 5, 0.01),
        createDepthFrame(100, 100, 5, 0.01),
        createDepthFrame(100, 100, 5, 0.01),
      ];

      const result = gate.analyze(frames);

      expect(result.temporalConsistency).toBeGreaterThan(0.8);
    });

    it('should compute low temporal consistency for varying depth', () => {
      const frames = [
        createDepthFrame(100, 100, 3, 0.1),
        createDepthFrame(100, 100, 5, 0.1),
        createDepthFrame(100, 100, 7, 0.1),
      ];

      const result = gate.analyze(frames);

      // Large depth changes should reduce consistency
      expect(result.temporalConsistency).toBeLessThan(0.8);
    });

    it('should compute edge stability with masks', () => {
      const frames = [
        createDepthFrame(100, 100, 5, 0.1),
        createDepthFrame(100, 100, 5, 0.1),
      ];

      // Set subject to be closer
      frames.forEach((frame) => {
        for (let y = 40; y < 60; y++) {
          for (let x = 40; x < 60; x++) {
            frame.data[y * 100 + x] = 2; // Subject at depth 2
          }
        }
      });

      const masks = [
        createMaskFrame(100, 100, 0.04),
        createMaskFrame(100, 100, 0.04),
      ];

      const result = gate.analyze(frames, masks);

      expect(result.edgeStability).toBeGreaterThan(0);
      expect(result.edgeStability).toBeLessThanOrEqual(1);
    });
  });

  describe('checkGate', () => {
    it('should pass for good depth metrics', () => {
      const metrics = {
        frameConfidences: [0.8, 0.8, 0.8],
        temporalConsistency: 0.85,
        edgeStability: 0.8,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(true);
      expect(result.depthWeight).toBeGreaterThan(0);
    });

    it('should fail for low temporal consistency', () => {
      const metrics = {
        frameConfidences: [0.8, 0.8, 0.8],
        temporalConsistency: 0.5, // Below 0.7 threshold
        edgeStability: 0.8,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.reasons.length).toBeGreaterThan(0);
      expect(result.reasons.some(r => r.includes('Temporal consistency'))).toBe(true);
    });

    it('should fail for low edge stability', () => {
      const metrics = {
        frameConfidences: [0.8, 0.8, 0.8],
        temporalConsistency: 0.85,
        edgeStability: 0.4, // Below 0.6 threshold
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.reasons.length).toBeGreaterThan(0);
      expect(result.reasons.some(r => r.includes('Edge stability'))).toBe(true);
    });
  });

  describe('computeDepthWeight', () => {
    it('should return weight based on confidence and config', () => {
      const metrics = {
        frameConfidences: [0.9, 0.9, 0.9],
        temporalConsistency: 0.9,
        edgeStability: 0.9,
      };

      const weight = gate.computeDepthWeight(metrics);

      expect(weight).toBeGreaterThan(0);
      expect(weight).toBeLessThanOrEqual(1);
    });

    it('should return lower weight for poor metrics', () => {
      const goodMetrics = {
        frameConfidences: [0.9, 0.9, 0.9],
        temporalConsistency: 0.9,
        edgeStability: 0.9,
      };

      const poorMetrics = {
        frameConfidences: [0.3, 0.3, 0.3],
        temporalConsistency: 0.4,
        edgeStability: 0.3,
      };

      const goodWeight = gate.computeDepthWeight(goodMetrics);
      const poorWeight = gate.computeDepthWeight(poorMetrics);

      expect(goodWeight).toBeGreaterThan(poorWeight);
    });
  });
});
