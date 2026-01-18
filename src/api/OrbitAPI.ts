/**
 * Orbit API Contract
 * Defines the public API for the Orbit processing service
 */

import {
  JobResult,
  JobStatus,
  PipelineStage,
  ProgressEvent,
  OutputOrientation,
  PreviewCameraPose,
} from '../schemas/types';

// ============================================================================
// Request Types
// ============================================================================

/** Request to submit a new processing job */
export interface SubmitJobRequest {
  /** Unique job identifier (client-provided or auto-generated) */
  jobId?: string;
  /** Input video URL or file path */
  inputUrl: string;
  /** Output directory URL or path */
  outputUrl: string;
  /** Processing options */
  options?: ProcessingOptions;
  /** Callback URLs for progress notifications */
  callbacks?: CallbackConfig;
}

/** Processing options */
export interface ProcessingOptions {
  /** Snapshot time in seconds from video start */
  snapshotTimeSec?: number;
  /** Segment start time in seconds */
  segmentStartSec?: number;
  /** Segment end time in seconds */
  segmentEndSec?: number;
  /** Output orientation */
  orientation?: OutputOrientation;
  /** Camera pose for preview + Veo generation */
  cameraPose?: PreviewCameraPose;
  /** Target output resolution (optional) */
  resolution?: {
    width: number;
    height: number;
  };
  /** Target output FPS (optional) */
  fps?: number;
  /** Output duration in seconds (default: match input segment) */
  durationSec?: number;
}

/** Callback configuration */
export interface CallbackConfig {
  /** URL to POST progress events */
  progressUrl?: string;
  /** URL to POST completion notification */
  completionUrl?: string;
  /** URL to POST error notification */
  errorUrl?: string;
}

// ============================================================================
// Response Types
// ============================================================================

/** Response from job submission */
export interface SubmitJobResponse {
  /** Assigned job ID */
  jobId: string;
  /** Initial job status */
  status: JobStatus;
  /** Estimated processing stages */
  estimatedStages: PipelineStage[];
  /** URLs for polling and artifacts */
  urls: {
    status: string;
    cancel: string;
    artifacts?: string;
  };
}

/** Response from job status query */
export interface JobStatusResponse {
  jobId: string;
  status: JobStatus;
  currentStage?: PipelineStage;
  progress?: number;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  /** Available when completed */
  result?: JobResultResponse;
}

/** Job result details */
export interface JobResultResponse {
  /** Snapshot image URL */
  snapshotUrl: string;
  /** 3D reconstruction asset URL */
  reconstructionUrl: string;
  /** Preview still URL */
  previewImageUrl: string;
  /** Veo-generated video URL */
  veoVideoUrl: string;
  /** Output orientation */
  orientation: OutputOrientation;
  /** Output metadata */
  metadata?: {
    duration: number;
    width: number;
    height: number;
    fps?: number;
  };
}

// ============================================================================
// Event Types
// ============================================================================

/** Progress event payload (sent to callback URL) */
export interface ProgressEventPayload {
  jobId: string;
  stage: PipelineStage;
  progress: number;
  message?: string;
  timestamp: string;
  /** Estimated remaining time in seconds */
  estimatedRemaining?: number;
}

/** Completion event payload */
export interface CompletionEventPayload {
  jobId: string;
  status: 'completed' | 'failed';
  timestamp: string;
  result?: JobResultResponse;
  error?: string;
}

// ============================================================================
// Error Types
// ============================================================================

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export const ErrorCodes = {
  INVALID_INPUT: 'INVALID_INPUT',
  JOB_NOT_FOUND: 'JOB_NOT_FOUND',
  PROCESSING_FAILED: 'PROCESSING_FAILED',
  QUALITY_GATE_FAILED: 'QUALITY_GATE_FAILED',
  REFINEMENT_FAILED: 'REFINEMENT_FAILED',
  EXPORT_FAILED: 'EXPORT_FAILED',
  RATE_LIMITED: 'RATE_LIMITED',
  UNAUTHORIZED: 'UNAUTHORIZED',
} as const;

// ============================================================================
// API Client Interface
// ============================================================================

export interface OrbitApiClient {
  /** Submit a new processing job */
  submitJob(request: SubmitJobRequest): Promise<SubmitJobResponse>;

  /** Get job status */
  getJobStatus(jobId: string): Promise<JobStatusResponse>;

  /** Cancel a job */
  cancelJob(jobId: string): Promise<{ cancelled: boolean }>;

  /** Get job result (when completed) */
  getJobResult(jobId: string): Promise<JobResultResponse>;
}

// ============================================================================
// Client Implementation
// ============================================================================

export class OrbitApiClientImpl implements OrbitApiClient {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl: string, apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json() as ApiError;
      throw new Error(`API Error: ${error.code} - ${error.message}`);
    }

    return response.json() as Promise<T>;
  }

  async submitJob(request: SubmitJobRequest): Promise<SubmitJobResponse> {
    return this.request<SubmitJobResponse>('POST', '/api/v1/jobs', request);
  }

  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>('GET', `/api/v1/jobs/${jobId}`);
  }

  async cancelJob(jobId: string): Promise<{ cancelled: boolean }> {
    return this.request<{ cancelled: boolean }>('POST', `/api/v1/jobs/${jobId}/cancel`);
  }

  async getJobResult(jobId: string): Promise<JobResultResponse> {
    return this.request<JobResultResponse>('GET', `/api/v1/jobs/${jobId}/result`);
  }
}

/**
 * Create an Orbit API client
 */
export function createOrbitApiClient(baseUrl: string, apiKey?: string): OrbitApiClient {
  return new OrbitApiClientImpl(baseUrl, apiKey);
}

// ============================================================================
// Modal Backend Client
// ============================================================================

import { config, getEndpointUrl } from '../config';

/** Modal backend job submission request */
export interface ModalSubmitRequest {
  video_url?: string;
  frames?: string[];  // Base64 encoded frames
  width: number;
  height: number;
  fps?: number;
  snapshot_time_sec?: number;
  orientation?: OutputOrientation;
  camera_pose?: PreviewCameraPose;
  options?: Record<string, unknown>;
}

/** Modal backend job status response */
export interface ModalJobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  stage?: string;
  progress: number;
  message?: string;
  created_at: string;
  updated_at: string;
}

/** Modal backend job result */
export interface ModalJobResult {
  job_id: string;
  status: string;
  snapshot_url: string;
  reconstruction_url: string;
  preview_image_url: string;
  veo_video_url: string;
  orientation: OutputOrientation;
  metadata?: {
    duration: number;
    width: number;
    height: number;
    fps?: number;
  };
}

/** Progress callback for job polling */
export type ProgressCallback = (status: ModalJobStatus) => void;

/**
 * Modal Backend Client
 * Handles communication with Modal serverless GPU backend
 */
export class ModalBackendClient {
  private pollInterval: number;
  private maxPollAttempts: number;

  constructor() {
    this.pollInterval = config.pollInterval;
    this.maxPollAttempts = config.maxPollAttempts;
  }

  /**
   * Check backend health
   */
  async checkHealth(): Promise<{ status: string; service: string; version: string }> {
    const response = await fetch(getEndpointUrl('health'));
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }

  /**
   * Submit a job to the Modal backend
   */
  async submitJob(request: ModalSubmitRequest): Promise<{ job_id: string; status: string; message: string }> {
    const response = await fetch(getEndpointUrl('submitJob'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Submit failed: ${error.error || response.status}`);
    }

    return response.json();
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<ModalJobStatus> {
    const url = new URL(getEndpointUrl('getJobStatus'));
    url.searchParams.set('job_id', jobId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Get job result
   */
  async getJobResult(jobId: string): Promise<ModalJobResult> {
    const url = new URL(getEndpointUrl('getJobResult'));
    url.searchParams.set('job_id', jobId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Result fetch failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Cancel a job
   */
  async cancelJob(jobId: string): Promise<{ cancelled: boolean; job_id: string }> {
    const url = new URL(getEndpointUrl('cancelJob'));
    url.searchParams.set('job_id', jobId);

    const response = await fetch(url.toString(), { method: 'POST' });
    if (!response.ok) {
      throw new Error(`Cancel failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * List recent jobs
   */
  async listJobs(): Promise<{ jobs: ModalJobStatus[] }> {
    const response = await fetch(getEndpointUrl('listJobs'));
    if (!response.ok) {
      throw new Error(`List jobs failed: ${response.status}`);
    }
    return response.json();
  }

  /**
   * Submit job and poll until completion
   */
  async submitAndWait(
    request: ModalSubmitRequest,
    onProgress?: ProgressCallback,
  ): Promise<ModalJobResult> {
    // Submit job
    const { job_id } = await this.submitJob(request);

    // Poll for completion
    let attempts = 0;
    while (attempts < this.maxPollAttempts) {
      await this.sleep(this.pollInterval);

      const status = await this.getJobStatus(job_id);

      if (onProgress) {
        onProgress(status);
      }

      if (status.status === 'completed') {
        return this.getJobResult(job_id);
      }

      if (status.status === 'failed') {
        throw new Error(`Job failed: ${status.message}`);
      }

      if (status.status === 'cancelled') {
        throw new Error('Job was cancelled');
      }

      attempts++;
    }

    throw new Error('Job timed out waiting for completion');
  }

  /**
   * Convert video frames to base64 for submission
   */
  async framesToBase64(frames: ImageData[]): Promise<string[]> {
    const encoded: string[] = [];

    for (const frame of frames) {
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(frame, 0, 0);

      // Convert to PNG base64
      const dataUrl = canvas.toDataURL('image/png');
      const base64 = dataUrl.split(',')[1];
      encoded.push(base64);
    }

    return encoded;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Create a Modal backend client
 */
export function createModalClient(): ModalBackendClient {
  return new ModalBackendClient();
}
