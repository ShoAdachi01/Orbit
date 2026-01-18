/**
 * Job Orchestrator
 * Manages the processing pipeline with progress events and error handling
 */

import {
  JobStatus,
  JobResult,
  PipelineStage,
  ProgressEvent,
  QualityReport,
  OrbitMode,
} from '../schemas/types';

export type ProgressCallback = (event: ProgressEvent) => void;

// Use 'any' for stage processor to allow different input/output types per stage
// Each stage transforms the pipeline data progressively
export type StageProcessor = (
  input: any,
  onProgress: (progress: number, message?: string) => void
) => Promise<any>;

interface PipelineStageConfig {
  name: PipelineStage;
  processor: StageProcessor;
  optional?: boolean;
}

// Re-export JobResult from types for convenience
export type { JobResult } from '../schemas/types';

export interface JobConfig {
  id: string;
  inputPath: string;
  outputPath: string;
  onProgress?: ProgressCallback;
}

export class JobOrchestrator {
  private jobs: Map<string, JobContext> = new Map();
  private stages: PipelineStageConfig[] = [];

  constructor() {
    this.initializeStages();
  }

  private initializeStages(): void {
    this.stages = [
      { name: 'upload', processor: this.processUpload.bind(this) },
      { name: 'segmentation', processor: this.processSegmentation.bind(this) },
      { name: 'pose-estimation', processor: this.processPoseEstimation.bind(this) },
      { name: 'depth-estimation', processor: this.processDepthEstimation.bind(this) },
      { name: 'tracking', processor: this.processTracking.bind(this) },
      { name: 'background-reconstruction', processor: this.processBackgroundReconstruction.bind(this) },
      { name: 'subject-reconstruction', processor: this.processSubjectReconstruction.bind(this) },
      { name: 'base-render', processor: this.processBaseRender.bind(this) },
      { name: 'refinement', processor: this.processRefinement.bind(this), optional: true },
      { name: 'export', processor: this.processExport.bind(this) },
    ];
  }

  async submitJob(config: JobConfig): Promise<string> {
    const context = new JobContext(config);
    this.jobs.set(config.id, context);

    // Start processing asynchronously
    this.processJob(context).catch((error) => {
      context.setError(error);
    });

    return config.id;
  }

  getJobStatus(jobId: string): JobResult | null {
    const context = this.jobs.get(jobId);
    if (!context) return null;
    return context.getResult();
  }

  async cancelJob(jobId: string): Promise<boolean> {
    const context = this.jobs.get(jobId);
    if (!context) return false;
    context.cancel();
    return true;
  }

  private async processJob(context: JobContext): Promise<void> {
    let stageData: unknown = { inputPath: context.config.inputPath };

    for (const stage of this.stages) {
      if (context.isCancelled) {
        context.setStatus('failed');
        return;
      }

      try {
        context.setCurrentStage(stage.name);

        stageData = await stage.processor(
          stageData,
          (progress, message) => {
            context.emitProgress(stage.name, progress, message);
          }
        );

        // Check for fallback after quality gates
        if (this.shouldSkipRemainingStages(stageData, stage.name)) {
          break;
        }
      } catch (error) {
        if (stage.optional) {
          console.warn(`Optional stage ${stage.name} failed:`, error);
          continue;
        }
        throw error;
      }
    }

    context.setStatus('completed');
    context.setOutputData(stageData);
  }

  private shouldSkipRemainingStages(data: unknown, stageName: PipelineStage): boolean {
    // After pose estimation, check if we should go to render-only mode
    const pipelineData = data as PipelineData;
    if (stageName === 'pose-estimation' && pipelineData.mode === 'render-only') {
      return true;
    }
    return false;
  }

  // Stage processors (to be implemented with actual ML models)
  private async processUpload(
    input: { inputPath: string },
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Reading video file...');

    // TODO: Actually read video file and extract frames
    const videoInfo = await this.extractVideoInfo(input.inputPath);

    onProgress(100, 'Upload complete');

    return {
      inputPath: input.inputPath,
      videoInfo,
      frames: [],
      mode: 'full-orbit',
    };
  }

  private async processSegmentation(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Running SAM2 segmentation...');

    // TODO: Run SAM2 for subject segmentation
    // This would call a Python backend or WASM model

    onProgress(50, 'Computing mask quality metrics...');

    // Compute mask quality metrics
    const maskMetrics = {
      score: 0.85,
      subjectAreaRatios: new Array(input.videoInfo.frameCount).fill(0.3),
      edgeJitter: 0.1,
      leakScore: 0.05,
      highCoverageFrameCount: 0,
      totalFrames: input.videoInfo.frameCount,
    };

    onProgress(100, 'Segmentation complete');

    return {
      ...input,
      masks: [],
      maskMetrics,
    };
  }

  private async processPoseEstimation(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Running background-only pose estimation...');

    // TODO: Run COLMAP or similar for pose estimation
    // Use masked frames to exclude subject

    onProgress(60, 'Computing pose quality metrics...');

    // Compute pose quality metrics
    const poseMetrics = {
      score: 0.8,
      inlierRatio: 0.65,
      medianReprojectionError: 1.5,
      quadrantCoverage: [true, true, true, true] as [boolean, boolean, boolean, boolean],
      goodCoverageFramePercent: 0.85,
      jitterScore: 0.2,
    };

    // Check pose gate
    const posePassed = this.checkPoseGate(poseMetrics);

    onProgress(100, 'Pose estimation complete');

    return {
      ...input,
      poses: [],
      poseMetrics,
      mode: posePassed ? input.mode : (poseMetrics.score < 0.3 ? 'render-only' : 'micro-parallax'),
    };
  }

  private async processDepthEstimation(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Running DepthCrafter depth estimation...');

    // TODO: Run DepthCrafter for depth priors

    onProgress(70, 'Computing depth confidence...');

    const depthMetrics = {
      frameConfidences: new Array(input.videoInfo.frameCount).fill(0.75),
      temporalConsistency: 0.85,
      edgeStability: 0.8,
    };

    onProgress(100, 'Depth estimation complete');

    return {
      ...input,
      depthMaps: [],
      depthMetrics,
    };
  }

  private async processTracking(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Running TAPIR foreground tracking...');

    // TODO: Run TAPIR for dense point tracking

    onProgress(60, 'Filtering tracks...');

    const trackMetrics = {
      numTracksKept: 120,
      numTracksDiscarded: 25,
      medianLifespan: 40,
      medianConfidence: 0.72,
    };

    // Check track gate
    const trackPassed = this.checkTrackGate(trackMetrics);

    onProgress(100, 'Tracking complete');

    return {
      ...input,
      tracks: [],
      trackMetrics,
      mode: trackPassed ? input.mode : '2.5d-subject',
    };
  }

  private async processBackgroundReconstruction(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Reconstructing background splat...');

    // TODO: Generate Gaussian splat from masked background frames

    onProgress(50, 'Running GaMO background completion...');

    // Background completion with confidence-weighted fusion

    onProgress(100, 'Background reconstruction complete');

    return {
      ...input,
      backgroundSplat: null, // Would be actual splat data
    };
  }

  private async processSubjectReconstruction(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Reconstructing 4D subject...');

    // TODO: Run SoM or similar for subject reconstruction

    if (input.mode === '2.5d-subject') {
      onProgress(50, 'Creating 2.5D billboard subject...');
      // Simplified 2.5D reconstruction
    } else {
      onProgress(50, 'Running full 4D reconstruction...');
      // Full 4D Gaussian reconstruction
    }

    onProgress(100, 'Subject reconstruction complete');

    return {
      ...input,
      subjectSplat: null, // Would be actual 4D splat data
    };
  }

  private async processBaseRender(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Generating base render...');

    // TODO: Render OrbitScene along camera path

    onProgress(100, 'Base render complete');

    return {
      ...input,
      baseRenderPath: `${input.inputPath.replace(/\.[^.]+$/, '')}_base_render.mp4`,
    };
  }

  private async processRefinement(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Selecting anchor frames...');

    // Select anchor frames for identity lock
    const anchors = await this.selectAnchorFrames(input);

    onProgress(30, 'Running generative refinement...');

    // TODO: Run video diffusion model for refinement

    onProgress(70, 'Checking identity drift...');

    const driftResult = await this.checkIdentityDrift(input, anchors);

    if (driftResult.driftExceeded) {
      onProgress(100, 'Identity drift detected, using base render');
      return {
        ...input,
        finalRenderPath: input.baseRenderPath,
        identityLockFailed: true,
      };
    }

    onProgress(100, 'Refinement complete');

    return {
      ...input,
      finalRenderPath: `${input.inputPath.replace(/\.[^.]+$/, '')}_refined.mp4`,
    };
  }

  private async processExport(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Packaging OrbitScene...');

    // Create OrbitScene pack
    const outputPath = input.inputPath.replace(/\.[^.]+$/, '_orbit/');

    onProgress(50, 'Writing quality report...');

    // Build quality report
    const quality: QualityReport = {
      mask: input.maskMetrics!,
      pose: input.poseMetrics!,
      track: input.trackMetrics!,
      depth: input.depthMetrics!,
      mode: input.mode,
      enforcedBounds: this.getBoundsForMode(input.mode),
      modeReasons: this.getModeReasons(input),
    };

    onProgress(100, 'Export complete');

    return {
      ...input,
      outputPath,
      quality,
    };
  }

  // Quality gate helpers
  private checkPoseGate(metrics: PipelineData['poseMetrics']): boolean {
    if (!metrics) return false;
    return (
      metrics.inlierRatio >= 0.35 &&
      metrics.medianReprojectionError <= 2.0 &&
      metrics.goodCoverageFramePercent >= 0.6 &&
      metrics.jitterScore <= 0.5
    );
  }

  private checkTrackGate(metrics: PipelineData['trackMetrics']): boolean {
    if (!metrics) return false;
    return (
      metrics.numTracksKept >= 50 &&
      metrics.medianLifespan >= 15 &&
      metrics.medianConfidence >= 0.6
    );
  }

  private async extractVideoInfo(path: string): Promise<VideoInfo> {
    // TODO: Use ffprobe or similar
    return {
      width: 1920,
      height: 1080,
      fps: 30,
      duration: 10,
      frameCount: 300,
    };
  }

  private async selectAnchorFrames(input: PipelineData): Promise<AnchorFrame[]> {
    // TODO: Select K anchor frames based on sharpness, face visibility, etc.
    return [];
  }

  private async checkIdentityDrift(
    input: PipelineData,
    anchors: AnchorFrame[]
  ): Promise<{ driftExceeded: boolean; maxDrift: number }> {
    // TODO: Compare face embeddings between refined output and anchors
    return { driftExceeded: false, maxDrift: 0.1 };
  }

  private getBoundsForMode(mode: OrbitMode) {
    const bounds = {
      'full-orbit': { maxYaw: 20, maxPitch: 10, maxRoll: 3, maxTranslation: 0.1, maxTranslationDepthPercent: 2, clampToParallax: true },
      'micro-parallax': { maxYaw: 8, maxPitch: 4, maxRoll: 1, maxTranslation: 0.03, maxTranslationDepthPercent: 1, clampToParallax: true },
      '2.5d-subject': { maxYaw: 15, maxPitch: 8, maxRoll: 2, maxTranslation: 0.05, maxTranslationDepthPercent: 1.5, clampToParallax: true },
      'render-only': { maxYaw: 0, maxPitch: 0, maxRoll: 0, maxTranslation: 0, maxTranslationDepthPercent: 0, clampToParallax: true },
    };
    return bounds[mode];
  }

  private getModeReasons(input: PipelineData): string[] {
    const reasons: string[] = [];

    if (input.mode === 'render-only') {
      reasons.push('Pose estimation failed quality gate');
    } else if (input.mode === 'micro-parallax') {
      reasons.push('Pose stability borderline, reduced orbit bounds');
    } else if (input.mode === '2.5d-subject') {
      reasons.push('Insufficient foreground tracks for full 4D reconstruction');
    } else {
      reasons.push('All quality gates passed');
    }

    if (input.identityLockFailed) {
      reasons.push('Identity drift exceeded threshold, using base render');
    }

    return reasons;
  }
}

interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  duration: number;
  frameCount: number;
}

interface AnchorFrame {
  frameIndex: number;
  sharpness: number;
  faceEmbedding?: Float32Array;
}

interface PipelineData {
  inputPath: string;
  outputPath?: string;
  videoInfo: VideoInfo;
  frames: unknown[];
  masks?: unknown[];
  poses?: unknown[];
  depthMaps?: unknown[];
  tracks?: unknown[];
  backgroundSplat?: unknown;
  subjectSplat?: unknown;
  baseRenderPath?: string;
  finalRenderPath?: string;
  quality?: QualityReport;
  maskMetrics?: QualityReport['mask'];
  poseMetrics?: QualityReport['pose'];
  trackMetrics?: QualityReport['track'];
  depthMetrics?: QualityReport['depth'];
  mode: OrbitMode;
  identityLockFailed?: boolean;
}

class JobContext {
  config: JobConfig;
  private status: JobStatus = 'pending';
  private currentStage: PipelineStage = 'upload';
  private outputData: unknown = null;
  private error: Error | null = null;
  private _isCancelled = false;

  constructor(config: JobConfig) {
    this.config = config;
  }

  get isCancelled(): boolean {
    return this._isCancelled;
  }

  setStatus(status: JobStatus): void {
    this.status = status;
  }

  setCurrentStage(stage: PipelineStage): void {
    this.currentStage = stage;
  }

  setOutputData(data: unknown): void {
    this.outputData = data;
  }

  setError(error: Error): void {
    this.error = error;
    this.status = 'failed';
  }

  cancel(): void {
    this._isCancelled = true;
  }

  emitProgress(stage: PipelineStage, progress: number, message?: string): void {
    if (this.config.onProgress) {
      this.config.onProgress({
        jobId: this.config.id,
        stage,
        progress,
        message,
        timestamp: Date.now(),
      });
    }
  }

  getResult(): JobResult {
    const data = this.outputData as PipelineData | null;
    return {
      jobId: this.config.id,
      status: this.status,
      scenePackPath: data?.outputPath,
      baseRenderPath: data?.baseRenderPath,
      finalRenderPath: data?.finalRenderPath,
      quality: data?.quality,
      error: this.error?.message,
    };
  }
}
