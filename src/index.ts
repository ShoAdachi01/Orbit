/**
 * Orbit
 * Generate a 3D preview from a snapshot and produce a new-angle video with Veo 3.1
 */

import type { OutputOrientation, PreviewCameraPose } from './schemas/types';

// Core Types
export * from './schemas/types';

// Pipeline
export { JobOrchestrator } from './pipeline/JobOrchestrator';
export type { JobConfig, ProgressCallback } from './pipeline/JobOrchestrator';

// Math Utilities
export * as math from './utils/math';

// ML Model Interface
export { getModelFactory } from './ml/ModelInterface';
export type {
  ModelFactory,
  SAM2Model,
  SAM2Config,
  SAM2Input,
  SAM2Output,
  TAPIRModel,
  TAPIRConfig,
  TAPIRInput,
  TAPIROutput,
  TAPIRTrack,
  DepthCrafterModel,
  DepthCrafterConfig,
  DepthCrafterInput,
  DepthCrafterOutput,
  COLMAPModel,
  COLMAPConfig,
  COLMAPInput,
  COLMAPOutput,
  COLMAPPose,
  FaceEmbeddingModel,
  FaceEmbeddingConfig,
  FaceEmbeddingInput,
  FaceEmbeddingOutput,
  VideoDiffusionModel,
  VideoDiffusionConfig,
  VideoDiffusionInput,
  VideoDiffusionOutput,
  SAM3DBodyModel,
  SAM3DBodyConfig,
  SAM3DBodyInput,
  SAM3DBodyOutput,
  NanoBananaModel,
  NanoBananaConfig,
  NanoBananaInput,
  NanoBananaOutput,
  VeoModel,
  VeoConfig,
  VeoInput,
  VeoOutput,
} from './ml/ModelInterface';

// API Client
export { createOrbitApiClient, OrbitApiClientImpl } from './api/OrbitAPI';
export type {
  OrbitApiClient,
  SubmitJobRequest,
  SubmitJobResponse,
  JobStatusResponse,
  JobResultResponse,
  ProcessingOptions,
  ApiError,
} from './api/OrbitAPI';

// Modal Backend Client
export { ModalClient, createModalClient, extractFramesFromVideo } from './api/ModalClient';
export type { ModalJobRequest, ModalJobStatus, ModalJobResult } from './api/ModalClient';

// Configuration
export { config, setConfig, getEndpointUrl } from './config';
export type { OrbitConfig } from './config';

/**
 * Quick start: Process a video through the full pipeline
 */
export async function processVideo(
  inputPath: string,
  outputPath: string,
  onProgressOrOptions?: ((stage: string, progress: number) => void) | ProcessVideoOptions
): Promise<import('./pipeline/JobOrchestrator').JobResult> {
  const { JobOrchestrator } = await import('./pipeline/JobOrchestrator');

  const orchestrator = new JobOrchestrator();
  const jobId = `job_${Date.now()}`;
  const options =
    typeof onProgressOrOptions === 'function' ? { onProgress: onProgressOrOptions } : onProgressOrOptions;

  await orchestrator.submitJob({
    id: jobId,
    inputPath,
    outputPath,
    orientation: options?.orientation,
    cameraPose: options?.cameraPose,
    snapshotTimeSec: options?.snapshotTimeSec,
    onProgress: (event) => {
      options?.onProgress?.(event.stage, event.progress);
    },
  });

  // Poll for completion
  return new Promise((resolve) => {
    const poll = setInterval(() => {
      const result = orchestrator.getJobStatus(jobId);
      if (result && (result.status === 'completed' || result.status === 'failed')) {
        clearInterval(poll);
        resolve(result);
      }
    }, 100);
  });
}

export interface ProcessVideoOptions {
  orientation?: OutputOrientation;
  cameraPose?: PreviewCameraPose;
  snapshotTimeSec?: number;
  onProgress?: (stage: string, progress: number) => void;
}

// Version info
export const VERSION = '1.0.0';
export const NAME = 'Orbit';
export const DESCRIPTION = 'Create a single-frame 3D preview and generate a new-angle video with Veo 3.1';
