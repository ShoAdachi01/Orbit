/**
 * Orbit - Multi-Cam v1
 * Turn a single creator video into a bounded parallax orbit preview + high-fidelity exported MP4
 */

// Core Types
export * from './schemas/types';

// Viewer Components
export { OrbitViewer, WebGPUContext, OrbitCamera, SplatRenderer, SceneLoader } from './viewer/OrbitViewer';
export type { OrbitViewerConfig, ViewerState } from './viewer/OrbitViewer';
export type { OrbitScene, SplatAsset } from './viewer/SceneLoader';
export { CameraPathEditor } from './viewer/CameraPathEditor';
export type { PathEditorConfig, PathEditorMode, PathEditorState } from './viewer/CameraPathEditor';

// Pipeline
export { JobOrchestrator } from './pipeline/JobOrchestrator';
export type { JobConfig, ProgressCallback } from './pipeline/JobOrchestrator';

// Quality Gates
export { SegmentationGate } from './quality/SegmentationGate';
export { PoseGate } from './quality/PoseGate';
export { TrackGate } from './quality/TrackGate';
export { DepthGate } from './quality/DepthGate';
export { FallbackDecisionEngine } from './quality/FallbackDecision';

// Reconstruction
export { BackgroundReconstruction } from './reconstruction/BackgroundReconstruction';
export { SubjectReconstruction } from './reconstruction/SubjectReconstruction';
export type { GaussianSplat } from './reconstruction/BackgroundReconstruction';
export type { Subject4DSplat } from './reconstruction/SubjectReconstruction';

// Refinement
export { IdentityLock } from './refinement/IdentityLock';

// Rendering
export { RenderPipeline } from './render/RenderPipeline';
export type { RenderConfig, RenderFrame } from './render/RenderPipeline';

// Export
export { MP4Export, downloadBlob } from './export/MP4Export';
export type { ExportConfig, ExportResult } from './export/MP4Export';

// Debug Artifacts
export { DebugArtifacts } from './debug/DebugArtifacts';
export type { DebugConfig, DebugArtifactPaths } from './debug/DebugArtifacts';

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
  RenderFrameRequest,
  RenderFrameResponse,
  ExportVideoRequest,
  ExportVideoResponse,
  ApiError,
} from './api/OrbitAPI';

// Modal Backend Client
export { ModalClient, createModalClient, extractFramesFromVideo } from './api/ModalClient';
export type { ModalJobRequest, ModalJobStatus, ModalJobResult } from './api/ModalClient';

// Configuration
export { config, setConfig, getEndpointUrl } from './config';
export type { OrbitConfig } from './config';

/**
 * Quick start: Create and initialize an OrbitViewer
 */
export async function createOrbitViewer(canvas: HTMLCanvasElement): Promise<import('./viewer/OrbitViewer').OrbitViewer> {
  const { OrbitViewer } = await import('./viewer/OrbitViewer');

  const viewer = new OrbitViewer({
    canvas,
    onProgress: (msg) => console.log('[Orbit]', msg),
    onError: (err) => console.error('[Orbit Error]', err),
    onModeChange: (mode) => console.log('[Orbit Mode]', mode),
  });

  await viewer.initialize();
  return viewer;
}

/**
 * Quick start: Process a video through the full pipeline
 */
export async function processVideo(
  inputPath: string,
  outputPath: string,
  onProgress?: (stage: string, progress: number) => void
): Promise<import('./pipeline/JobOrchestrator').JobResult> {
  const { JobOrchestrator } = await import('./pipeline/JobOrchestrator');

  const orchestrator = new JobOrchestrator();
  const jobId = `job_${Date.now()}`;

  await orchestrator.submitJob({
    id: jobId,
    inputPath,
    outputPath,
    onProgress: (event) => {
      onProgress?.(event.stage, event.progress);
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

// Version info
export const VERSION = '1.0.0';
export const NAME = 'Orbit';
export const DESCRIPTION = 'Turn a single creator video into a bounded parallax orbit preview + high-fidelity exported MP4';
