/**
 * Tests for TrackGate
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { TrackGate, Track } from '../TrackGate';

describe('TrackGate', () => {
  let gate: TrackGate;

  beforeEach(() => {
    gate = new TrackGate();
  });

  const createTrack = (
    id: number,
    lifespan: number,
    avgConfidence: number
  ): Track => {
    const points = [];
    for (let i = 0; i < lifespan; i++) {
      points.push({
        frameIndex: i,
        x: 100 + i * 2,
        y: 100 + i * 2,
        confidence: avgConfidence,
      });
    }
    return { id, points };
  };

  describe('analyze', () => {
    it('should return empty metrics for empty tracks', () => {
      const { metrics } = gate.analyze([]);

      expect(metrics.numTracksKept).toBe(0);
      expect(metrics.numTracksDiscarded).toBe(0);
    });

    it('should keep tracks with good quality', () => {
      const tracks = [
        createTrack(1, 30, 0.8),
        createTrack(2, 25, 0.75),
        createTrack(3, 40, 0.9),
      ];

      const { metrics, filteredTracks } = gate.analyze(tracks);

      expect(metrics.numTracksKept).toBe(3);
      expect(metrics.numTracksDiscarded).toBe(0);
      expect(filteredTracks.length).toBe(3);
    });

    it('should discard short tracks', () => {
      const tracks = [
        createTrack(1, 30, 0.8),
        createTrack(2, 10, 0.8), // Too short (< 15 frames)
        createTrack(3, 5, 0.8),  // Too short
      ];

      const { metrics, filteredTracks } = gate.analyze(tracks);

      expect(metrics.numTracksKept).toBe(1);
      expect(metrics.numTracksDiscarded).toBe(2);
      expect(filteredTracks.length).toBe(1);
      expect(filteredTracks[0].id).toBe(1);
    });

    it('should discard low confidence tracks', () => {
      const tracks = [
        createTrack(1, 30, 0.8),
        createTrack(2, 30, 0.4), // Low confidence (< 0.6)
        createTrack(3, 30, 0.5), // Low confidence
      ];

      const { metrics, filteredTracks } = gate.analyze(tracks);

      expect(metrics.numTracksKept).toBe(1);
      expect(metrics.numTracksDiscarded).toBe(2);
    });

    it('should compute correct median lifespan', () => {
      const tracks = [
        createTrack(1, 20, 0.8),
        createTrack(2, 30, 0.8),
        createTrack(3, 40, 0.8),
      ];

      const { metrics } = gate.analyze(tracks);

      expect(metrics.medianLifespan).toBe(30);
    });

    it('should compute correct median confidence', () => {
      const tracks = [
        createTrack(1, 20, 0.7),
        createTrack(2, 20, 0.8),
        createTrack(3, 20, 0.9),
      ];

      const { metrics } = gate.analyze(tracks);

      expect(metrics.medianConfidence).toBe(0.8);
    });
  });

  describe('checkGate', () => {
    it('should pass for good track metrics', () => {
      const metrics = {
        numTracksKept: 100,
        numTracksDiscarded: 20,
        medianLifespan: 35,
        medianConfidence: 0.75,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(true);
      expect(result.fallbackCandidate).toBeNull();
    });

    it('should fail for too few tracks', () => {
      const metrics = {
        numTracksKept: 30, // Below 50 threshold
        numTracksDiscarded: 100,
        medianLifespan: 35,
        medianConfidence: 0.75,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.fallbackCandidate).toBe('2.5d-subject');
    });

    it('should fail for short median lifespan', () => {
      const metrics = {
        numTracksKept: 100,
        numTracksDiscarded: 20,
        medianLifespan: 10, // Below 15 threshold
        medianConfidence: 0.75,
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.fallbackCandidate).toBe('2.5d-subject');
    });

    it('should fail for low median confidence', () => {
      const metrics = {
        numTracksKept: 100,
        numTracksDiscarded: 20,
        medianLifespan: 35,
        medianConfidence: 0.5, // Below 0.6 threshold
      };

      const result = gate.checkGate(metrics);

      expect(result.passed).toBe(false);
      expect(result.fallbackCandidate).toBe('2.5d-subject');
    });
  });

  describe('groupTracksBySpatialProximity', () => {
    it('should group tracks by position', () => {
      const tracks = [
        {
          id: 1,
          points: [{ frameIndex: 0, x: 10, y: 10, confidence: 0.8 }],
        },
        {
          id: 2,
          points: [{ frameIndex: 0, x: 15, y: 15, confidence: 0.8 }],
        },
        {
          id: 3,
          points: [{ frameIndex: 0, x: 100, y: 100, confidence: 0.8 }],
        },
      ];

      const groups = gate.groupTracksBySpatialProximity(tracks, 50);

      // Tracks 1 and 2 should be in the same grid cell (0,0)
      // Track 3 should be in a different cell (2,2)
      expect(groups.size).toBe(2);
    });
  });
});
