/**
 * Job Orchestrator
 * Manages the processing pipeline with progress events and error handling
 */

import {
  JobStatus,
  JobResult,
  PipelineStage,
  ProgressEvent,
  OutputOrientation,
  PreviewCameraPose,
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
  orientation?: OutputOrientation;
  cameraPose?: PreviewCameraPose;
  snapshotTimeSec?: number;
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
      { name: 'snapshot', processor: this.processSnapshot.bind(this) },
      { name: 'reconstruction', processor: this.processReconstruction.bind(this) },
      { name: 'preview-generation', processor: this.processPreviewGeneration.bind(this) },
      { name: 'veo-generation', processor: this.processVeoGeneration.bind(this) },
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

  // Stage processors (to be implemented with actual ML models)
  private async processUpload(
    input: {
      inputPath: string;
      outputPath: string;
      orientation?: OutputOrientation;
      cameraPose?: PreviewCameraPose;
      snapshotTimeSec?: number;
    },
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Reading video file...');

    // TODO: Actually read video file and extract frames
    const videoInfo = await this.extractVideoInfo(input.inputPath);
    const orientation = input.orientation || 'horizontal';
    const cameraPose = input.cameraPose || this.getDefaultCameraPose();
    const snapshotTimeSec = this.normalizeSnapshotTime(
      input.snapshotTimeSec,
      videoInfo.duration
    );

    onProgress(100, 'Upload complete');

    return {
      inputPath: input.inputPath,
      outputPath: input.outputPath,
      videoInfo,
      orientation,
      cameraPose,
      snapshotTimeSec,
    };
  }

  private async processSnapshot(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Selecting snapshot frame...');

    const snapshotPath = this.buildArtifactPath(input, 'snapshot.jpg');
    const frameIndex = Math.min(
      Math.max(Math.round(input.snapshotTimeSec * input.videoInfo.fps), 0),
      Math.max(input.videoInfo.frameCount - 1, 0)
    );

    const snapshot: SnapshotData = {
      frameIndex,
      timestamp: input.snapshotTimeSec,
      width: input.videoInfo.width,
      height: input.videoInfo.height,
      path: snapshotPath,
    };

    onProgress(100, 'Snapshot captured');

    return {
      ...input,
      snapshot,
      snapshotPath,
    };
  }

  private async processReconstruction(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Running SAM 3D Body reconstruction...');

    // TODO: Call SAM 3D Body and (optionally) SAM 3D Objects.
    const reconstructionPath = this.buildArtifactPath(input, 'sam3d_body.glb');

    const reconstruction: ReconstructionData = {
      path: reconstructionPath,
      format: 'sam3d-body',
    };

    onProgress(100, '3D reconstruction complete');

    return {
      ...input,
      reconstruction,
      reconstructionPath,
    };
  }

  private async processPreviewGeneration(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Generating preview still with Nano Banana Pro...');

    // TODO: Generate preview still based on snapshot + camera pose.
    const previewImagePath = this.buildArtifactPath(input, 'preview.jpg');
    const resolution = this.getPreviewResolution(input.videoInfo, input.orientation);

    const preview: PreviewData = {
      path: previewImagePath,
      width: resolution.width,
      height: resolution.height,
    };

    onProgress(100, 'Preview still generated');

    return {
      ...input,
      preview,
      previewImagePath,
    };
  }

  private async processVeoGeneration(
    input: PipelineData,
    onProgress: (progress: number, message?: string) => void
  ): Promise<PipelineData> {
    onProgress(0, 'Generating Veo 3.1 video...');

    // TODO: Call Veo 3.1 with preview still + context prompt.
    const veoVideoPath = this.buildArtifactPath(input, 'veo.mp4');

    const veo: VeoData = {
      path: veoVideoPath,
      duration: input.videoInfo.duration,
    };

    onProgress(100, 'Veo generation complete');

    return {
      ...input,
      veo,
      veoVideoPath,
    };
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

  private getDefaultCameraPose(): PreviewCameraPose {
    return {
      position: [0, 0, 0],
      rotation: [0, 0, 0, 1],
      fov: 45,
    };
  }

  private normalizeSnapshotTime(timeSec: number | undefined, duration: number): number {
    if (typeof timeSec !== 'number' || Number.isNaN(timeSec)) {
      return Math.min(1, Math.max(duration / 2, 0));
    }
    if (duration <= 0) {
      return 0;
    }
    return Math.min(Math.max(timeSec, 0), Math.max(duration - 0.001, 0));
  }

  private getPreviewResolution(
    videoInfo: VideoInfo,
    orientation: OutputOrientation
  ): { width: number; height: number } {
    if (orientation === 'vertical') {
      return { width: videoInfo.height, height: videoInfo.width };
    }
    return { width: videoInfo.width, height: videoInfo.height };
  }

  private buildArtifactPath(input: PipelineData, suffix: string): string {
    const base = input.outputPath || input.inputPath.replace(/\.[^.]+$/, '');
    const normalized = base.replace(/\/$/, '');
    return `${normalized}_${suffix}`;
  }
}

interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  duration: number;
  frameCount: number;
}

interface SnapshotData {
  frameIndex: number;
  timestamp: number;
  width: number;
  height: number;
  path: string;
}

interface ReconstructionData {
  path: string;
  format: 'sam3d-body';
}

interface PreviewData {
  path: string;
  width: number;
  height: number;
}

interface VeoData {
  path: string;
  duration: number;
}

interface PipelineData {
  inputPath: string;
  outputPath: string;
  videoInfo: VideoInfo;
  orientation: OutputOrientation;
  cameraPose: PreviewCameraPose;
  snapshotTimeSec: number;
  snapshot?: SnapshotData;
  reconstruction?: ReconstructionData;
  preview?: PreviewData;
  veo?: VeoData;
  snapshotPath?: string;
  reconstructionPath?: string;
  previewImagePath?: string;
  veoVideoPath?: string;
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
      snapshotPath: data?.snapshotPath,
      reconstructionPath: data?.reconstructionPath,
      previewImagePath: data?.previewImagePath,
      veoVideoPath: data?.veoVideoPath,
      orientation: data?.orientation,
      error: this.error?.message,
    };
  }
}
