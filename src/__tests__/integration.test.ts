/**
 * Integration Tests for Orbit Pipeline
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  FallbackDecisionEngine,
  SegmentationGate,
  PoseGate,
  TrackGate,
  DepthGate,
} from '../quality';
import { getModelFactory } from '../ml/ModelInterface';
import { RenderPipeline } from '../render/RenderPipeline';
import { CameraPathEditor } from '../viewer/CameraPathEditor';
import { OrbitCamera } from '../viewer/Camera';
import {
  DEFAULT_ORBIT_BOUNDS,
  QualityReport,
  CameraPath,
} from '../schemas/types';

describe('Orbit Pipeline Integration', () => {
  describe('Quality Gates Pipeline', () => {
    it('should evaluate all quality gates and make fallback decision', () => {
      // Simulate good quality metrics
      const metrics = {
        mask: {
          score: 0.85,
          subjectAreaRatios: [0.3, 0.32, 0.28, 0.31],
          edgeJitter: 0.12,
          leakScore: 0.05,
          highCoverageFrameCount: 0,
          totalFrames: 4,
        },
        pose: {
          score: 0.82,
          inlierRatio: 0.68,
          medianReprojectionError: 1.4,
          quadrantCoverage: [true, true, true, true] as [boolean, boolean, boolean, boolean],
          goodCoverageFramePercent: 0.88,
          jitterScore: 0.22,
        },
        track: {
          numTracksKept: 120,
          numTracksDiscarded: 25,
          medianLifespan: 38,
          medianConfidence: 0.74,
        },
        depth: {
          frameConfidences: [0.78, 0.82, 0.79, 0.81],
          temporalConsistency: 0.86,
          edgeStability: 0.79,
        },
      };

      const engine = new FallbackDecisionEngine();
      const result = engine.decide(metrics);

      // With good metrics, should get full-orbit mode
      expect(result.mode).toBe('full-orbit');
      expect(result.gateResults.segmentation.passed).toBe(true);
      expect(result.gateResults.pose.passed).toBe(true);
      expect(result.gateResults.track.passed).toBe(true);
      expect(result.gateResults.depth.passed).toBe(true);

      // Check bounds
      expect(result.bounds.maxYaw).toBe(20);
      expect(result.bounds.maxPitch).toBe(10);
    });

    it('should fallback to micro-parallax for borderline pose', () => {
      const metrics = {
        mask: {
          score: 0.85,
          subjectAreaRatios: [0.3],
          edgeJitter: 0.12,
          leakScore: 0.05,
          highCoverageFrameCount: 0,
          totalFrames: 1,
        },
        pose: {
          score: 0.45,
          inlierRatio: 0.32, // Below threshold
          medianReprojectionError: 2.5, // Above threshold
          quadrantCoverage: [true, true, false, false] as [boolean, boolean, boolean, boolean],
          goodCoverageFramePercent: 0.5, // Below threshold
          jitterScore: 0.4,
        },
        track: {
          numTracksKept: 100,
          numTracksDiscarded: 20,
          medianLifespan: 35,
          medianConfidence: 0.72,
        },
        depth: {
          frameConfidences: [0.8],
          temporalConsistency: 0.85,
          edgeStability: 0.78,
        },
      };

      const engine = new FallbackDecisionEngine();
      const result = engine.decide(metrics);

      expect(result.mode).toBe('micro-parallax');
      expect(result.bounds.maxYaw).toBe(8);
    });

    it('should fallback to 2.5d-subject for poor tracking', () => {
      const metrics = {
        mask: {
          score: 0.85,
          subjectAreaRatios: [0.3],
          edgeJitter: 0.12,
          leakScore: 0.05,
          highCoverageFrameCount: 0,
          totalFrames: 1,
        },
        pose: {
          score: 0.82,
          inlierRatio: 0.68,
          medianReprojectionError: 1.4,
          quadrantCoverage: [true, true, true, true] as [boolean, boolean, boolean, boolean],
          goodCoverageFramePercent: 0.88,
          jitterScore: 0.22,
        },
        track: {
          numTracksKept: 25, // Too few tracks
          numTracksDiscarded: 150,
          medianLifespan: 10, // Too short
          medianConfidence: 0.5, // Too low
        },
        depth: {
          frameConfidences: [0.8],
          temporalConsistency: 0.85,
          edgeStability: 0.78,
        },
      };

      const engine = new FallbackDecisionEngine();
      const result = engine.decide(metrics);

      expect(result.mode).toBe('2.5d-subject');
    });
  });

  describe('Camera Path Editor', () => {
    let camera: OrbitCamera;
    let editor: CameraPathEditor;

    beforeAll(() => {
      camera = new OrbitCamera(DEFAULT_ORBIT_BOUNDS);
      editor = new CameraPathEditor(camera);
    });

    it('should add keyframes', () => {
      editor.addKeyframe(0);
      editor.addKeyframe(1);
      editor.addKeyframe(2);

      const path = editor.getPath();
      expect(path.keyframes.length).toBe(3);
    });

    it('should delete keyframes', () => {
      editor.clear();
      editor.addKeyframe(0);
      editor.addKeyframe(1);
      editor.addKeyframe(2);

      editor.deleteKeyframe(1);

      const path = editor.getPath();
      expect(path.keyframes.length).toBe(2);
      expect(path.keyframes[0].timestamp).toBe(0);
      expect(path.keyframes[1].timestamp).toBe(2);
    });

    it('should generate orbit path', () => {
      editor.clear();
      editor.generateOrbitPath(5, 1); // 5 seconds, 1 revolution

      const path = editor.getPath();
      expect(path.keyframes.length).toBeGreaterThan(0);
      expect(editor.getDuration()).toBe(5);
    });

    it('should export and import path', () => {
      editor.clear();
      editor.generateOrbitPath(3, 0.5);

      const exported = editor.exportPath();
      const originalKeyframes = editor.getPath().keyframes.length;

      editor.clear();
      expect(editor.getPath().keyframes.length).toBe(0);

      const success = editor.importPath(exported);
      expect(success).toBe(true);
      expect(editor.getPath().keyframes.length).toBe(originalKeyframes);
    });
  });

  describe('ML Model Stubs', () => {
    it('should create all model stubs', () => {
      const factory = getModelFactory();

      expect(factory.createSAM2()).toBeDefined();
      expect(factory.createTAPIR()).toBeDefined();
      expect(factory.createDepthCrafter()).toBeDefined();
      expect(factory.createCOLMAP()).toBeDefined();
      expect(factory.createFaceEmbedding()).toBeDefined();
      expect(factory.createVideoDiffusion()).toBeDefined();
    });

    it('should run SAM2 stub segmentation', async () => {
      const factory = getModelFactory();
      const sam2 = factory.createSAM2();

      await sam2.initialize({});

      const result = await sam2.segment({
        image: new Uint8Array(100 * 100 * 3),
        width: 100,
        height: 100,
      });

      expect(result.success).toBe(true);
      expect(result.data?.mask.length).toBe(100 * 100);
      expect(result.data?.score).toBeGreaterThan(0);

      sam2.destroy();
    });

    it('should run TAPIR stub tracking', async () => {
      const factory = getModelFactory();
      const tapir = factory.createTAPIR();

      await tapir.initialize({});

      const frames = [
        { data: new Uint8Array(100 * 100 * 3), width: 100, height: 100 },
        { data: new Uint8Array(100 * 100 * 3), width: 100, height: 100 },
      ];

      const result = await tapir.track({
        frames,
        queryPoints: [{ x: 50, y: 50 }],
        queryFrameIndex: 0,
      });

      expect(result.success).toBe(true);
      expect(result.data?.tracks.length).toBe(1);
      expect(result.data?.tracks[0].points.length).toBe(2);

      tapir.destroy();
    });

    it('should run DepthCrafter stub', async () => {
      const factory = getModelFactory();
      const depthCrafter = factory.createDepthCrafter();

      await depthCrafter.initialize({});

      const frames = [
        { data: new Uint8Array(100 * 100 * 3), width: 100, height: 100 },
      ];

      const result = await depthCrafter.estimateDepth({ frames });

      expect(result.success).toBe(true);
      expect(result.data?.depthMaps.length).toBe(1);
      expect(result.data?.depthMaps[0].length).toBe(100 * 100);

      depthCrafter.destroy();
    });
  });

  describe('Render Pipeline', () => {
    it('should initialize render pipeline', async () => {
      // Skip test in Node.js environment (OffscreenCanvas not available)
      if (typeof OffscreenCanvas === 'undefined') {
        console.log('Skipping render pipeline test - OffscreenCanvas not available in Node.js');
        return;
      }

      const pipeline = new RenderPipeline({
        width: 640,
        height: 480,
        fps: 30,
      });

      await pipeline.initialize();
      pipeline.destroy();
    });
  });

  describe('Orbit Bounds', () => {
    it('should enforce orbit bounds on camera', () => {
      const camera = new OrbitCamera(DEFAULT_ORBIT_BOUNDS);

      // Try to exceed bounds
      camera.yaw = 50; // Should be clamped to 20
      expect(camera.yaw).toBe(20);

      camera.pitch = -30; // Should be clamped to -10
      expect(camera.pitch).toBe(-10);

      camera.roll = 10; // Should be clamped to 3
      expect(camera.roll).toBe(3);
    });

    it('should allow changing bounds', () => {
      const camera = new OrbitCamera(DEFAULT_ORBIT_BOUNDS);

      camera.setBounds({ maxYaw: 40 });
      camera.yaw = 35;
      expect(camera.yaw).toBe(35);
    });
  });
});

describe('End-to-End Simulation', () => {
  it('should simulate full pipeline flow', async () => {
    // 1. Initialize models
    const factory = getModelFactory();
    const sam2 = factory.createSAM2();
    const tapir = factory.createTAPIR();
    const depthCrafter = factory.createDepthCrafter();
    const colmap = factory.createCOLMAP();

    await sam2.initialize({});
    await tapir.initialize({});
    await depthCrafter.initialize({});
    await colmap.initialize({});

    // 2. Simulate video frames
    const frames = [];
    for (let i = 0; i < 10; i++) {
      frames.push({
        data: new Uint8Array(100 * 100 * 3),
        width: 100,
        height: 100,
        id: i,
      });
    }

    // 3. Run segmentation
    const segResults = await sam2.segmentVideo(
      frames.map(f => ({ image: f.data, width: f.width, height: f.height }))
    );
    expect(segResults.success).toBe(true);

    // 4. Run pose estimation (masked)
    const poseResult = await colmap.reconstruct({
      images: frames,
      masks: segResults.data?.map(m => ({
        data: m.mask,
        width: 100,
        height: 100,
      })),
    });
    expect(poseResult.success).toBe(true);
    expect(poseResult.data?.poses.length).toBe(10);

    // 5. Run depth estimation
    const depthResult = await depthCrafter.estimateDepth({
      frames: frames.map(f => ({ data: f.data, width: f.width, height: f.height })),
    });
    expect(depthResult.success).toBe(true);

    // 6. Run tracking
    const trackResult = await tapir.track({
      frames: frames.map(f => ({ data: f.data, width: f.width, height: f.height })),
      queryPoints: [
        { x: 30, y: 30 },
        { x: 50, y: 50 },
        { x: 70, y: 70 },
      ],
      queryFrameIndex: 0,
    });
    expect(trackResult.success).toBe(true);
    expect(trackResult.data?.tracks.length).toBe(3);

    // 7. Evaluate quality and decide mode
    const segGate = new SegmentationGate();
    const poseGate = new PoseGate();
    const trackGate = new TrackGate();
    const depthGate = new DepthGate();

    // Use stub metrics for this simulation
    const metrics = {
      mask: {
        score: 0.85,
        subjectAreaRatios: segResults.data!.map(m =>
          m.mask.filter(v => v > 0).length / m.mask.length
        ),
        edgeJitter: 0.1,
        leakScore: 0.05,
        highCoverageFrameCount: 0,
        totalFrames: 10,
      },
      pose: {
        score: 0.8,
        inlierRatio: 0.7,
        medianReprojectionError: 1.5,
        quadrantCoverage: [true, true, true, true] as [boolean, boolean, boolean, boolean],
        goodCoverageFramePercent: 0.9,
        jitterScore: 0.2,
      },
      track: {
        numTracksKept: trackResult.data!.tracks.length,
        numTracksDiscarded: 0,
        medianLifespan: 10,
        medianConfidence: 0.8,
      },
      depth: {
        frameConfidences: depthResult.data!.depthMaps.map(() => 0.8),
        temporalConsistency: 0.85,
        edgeStability: 0.75,
      },
    };

    const engine = new FallbackDecisionEngine();
    const decision = engine.decide(metrics);

    expect(['full-orbit', 'micro-parallax', '2.5d-subject', 'render-only']).toContain(
      decision.mode
    );

    // 8. Create camera path
    const camera = new OrbitCamera(decision.bounds);
    const pathEditor = new CameraPathEditor(camera, { bounds: decision.bounds });
    pathEditor.generateOrbitPath(3, 0.5);

    const path = pathEditor.getPath();
    expect(path.keyframes.length).toBeGreaterThan(0);

    // 9. Cleanup
    sam2.destroy();
    tapir.destroy();
    depthCrafter.destroy();
    colmap.destroy();

    console.log('Pipeline simulation complete');
    console.log(`Mode: ${decision.mode}`);
    console.log(`Bounds: yaw=${decision.bounds.maxYaw}°, pitch=${decision.bounds.maxPitch}°`);
  });
});
