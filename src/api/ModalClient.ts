/**
 * Modal Backend Client
 * Connects frontend to Modal serverless GPU backend
 */

import { config } from '../config';
import type { QualityReport, OrbitMode, OrbitBounds } from '../schemas/types';

export interface ModalJobRequest {
  /** Video frames as base64 encoded strings */
  frames: string[];
  /** Frame width */
  width: number;
  /** Frame height */
  height: number;
  /** Video FPS */
  fps: number;
  /** Optional prompt points for segmentation */
  promptPoints?: Array<{ x: number; y: number; label?: number }>;
}

export interface ModalJobStatus {
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  stage?: string;
  progress: number;
  message?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ModalJobResult {
  jobId: string;
  status: string;
  mode: OrbitMode;
  bounds: OrbitBounds;
  quality: {
    mask: QualityReport['mask'];
    pose: QualityReport['pose'];
    track: QualityReport['track'];
    depth: QualityReport['depth'];
    gateResults: Record<string, { passed: boolean; reasons?: string[] }>;
  };
  assets: {
    masks: string[];
    poses?: string;
    depth?: string[];
    tracks?: string;
  };
  error?: string;
}

/**
 * Modal Backend Client
 */
export class ModalClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || config.backendUrl;
  }

  /**
   * Get Modal endpoint URL
   * ASGI app serves all endpoints from a single base URL with path routing
   */
  private getEndpointUrl(endpoint: string): string {
    // baseUrl is like: https://shoadachi01--orbit-backend-fastapi-app.modal.run
    // endpoint is like: submit_job
    // result: https://shoadachi01--orbit-backend-fastapi-app.modal.run/submit_job
    const cleanBase = this.baseUrl.replace(/\/$/, ''); // Remove trailing slash if any
    return `${cleanBase}/${endpoint}`;
  }

  /**
   * Submit a video processing job to Modal backend
   */
  async submitJob(request: ModalJobRequest): Promise<{ jobId: string }> {
    const response = await fetch(this.getEndpointUrl('submit_job'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        frames: request.frames,
        width: request.width,
        height: request.height,
        fps: request.fps,
        prompt_points: request.promptPoints,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to submit job: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<ModalJobStatus> {
    const response = await fetch(`${this.getEndpointUrl('get_job_status')}?job_id=${jobId}`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to get job status: ${response.statusText}`);
    }

    const data = await response.json();

    return {
      jobId: data.job_id,
      status: data.status,
      stage: data.stage,
      progress: data.progress,
      message: data.message,
      createdAt: data.created_at,
      updatedAt: data.updated_at,
    };
  }

  /**
   * Get job result
   */
  async getJobResult(jobId: string): Promise<ModalJobResult> {
    const response = await fetch(`${this.getEndpointUrl('get_job_result')}?job_id=${jobId}`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to get job result: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Poll job until completion
   */
  async waitForCompletion(
    jobId: string,
    onProgress?: (status: ModalJobStatus) => void,
    pollInterval: number = 2000,
    maxWait: number = 3600000, // 1 hour default
  ): Promise<ModalJobResult> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWait) {
      const status = await this.getJobStatus(jobId);

      onProgress?.(status);

      if (status.status === 'completed') {
        return this.getJobResult(jobId);
      }

      if (status.status === 'failed') {
        throw new Error(`Job failed: ${status.message}`);
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error('Job timed out');
  }

  /**
   * Check if backend is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.getEndpointUrl('health')}`, {
        method: 'GET',
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Upload video file directly to backend
   */
  async uploadVideo(
    file: File,
    onProgress?: (progress: { loaded: number; total: number; percent: number }) => void,
  ): Promise<{ videoId: string; sizeBytes: number }> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      xhr.open('POST', this.getEndpointUrl('upload_video'));

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress({
            loaded: e.loaded,
            total: e.total,
            percent: Math.round((e.loaded / e.total) * 100),
          });
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            if (data.error) {
              reject(new Error(data.error));
            } else {
              resolve({
                videoId: data.video_id,
                sizeBytes: data.size_bytes,
              });
            }
          } catch {
            reject(new Error('Invalid response from server'));
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      };

      xhr.onerror = () => reject(new Error('Upload failed: Network error'));
      xhr.timeout = 600000; // 10 minute timeout
      xhr.ontimeout = () => reject(new Error('Upload timed out'));

      xhr.send(formData);
    });
  }

  /**
   * Submit job using uploaded video ID
   */
  async submitJobWithVideo(request: {
    videoId: string;
    maxFrames?: number;
    targetFps?: number;
    promptPoints?: Array<{ x: number; y: number; label?: number }>;
  }): Promise<{ jobId: string }> {
    const response = await fetch(this.getEndpointUrl('submit_job'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_id: request.videoId,
        max_frames: request.maxFrames || 150,
        target_fps: request.targetFps || 10,
        prompt_points: request.promptPoints,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to submit job: ${response.statusText}`);
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }

    return { jobId: data.job_id };
  }
}

/**
 * Extract frames from video file
 */
export async function extractFramesFromVideo(
  videoFile: File,
  maxFrames: number = 300,
): Promise<{ frames: string[]; width: number; height: number; fps: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    video.muted = true;
    video.src = URL.createObjectURL(videoFile);

    video.onloadedmetadata = async () => {
      const width = video.videoWidth;
      const height = video.videoHeight;
      const duration = video.duration;
      const fps = 30; // Assume 30fps, could detect from file

      canvas.width = width;
      canvas.height = height;

      const frameCount = Math.min(maxFrames, Math.floor(duration * fps));
      const frames: string[] = [];

      for (let i = 0; i < frameCount; i++) {
        video.currentTime = i / fps;
        await new Promise(r => video.onseeked = r);

        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        // Extract base64 part
        const base64 = dataUrl.split(',')[1];
        frames.push(base64);
      }

      URL.revokeObjectURL(video.src);
      resolve({ frames, width, height, fps });
    };

    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      reject(new Error('Failed to load video'));
    };
  });
}

/**
 * Create a Modal client
 */
export function createModalClient(baseUrl?: string): ModalClient {
  return new ModalClient(baseUrl);
}
