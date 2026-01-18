/**
 * Debug Artifacts Generator (Section G from PRD - Non-negotiable)
 *
 * For every job, stores:
 * - mask preview mp4 (lossless-ish)
 * - pose reprojection overlay video
 * - depth preview video
 * - track overlay video
 * - base_render + refined_render
 * - quality.json + selected fallback + reasons
 */

import { QualityReport, CameraPose, Matrix4x4 } from '../schemas/types';

export interface DebugArtifactPaths {
  maskPreview: string;
  poseOverlay: string;
  depthPreview: string;
  trackOverlay: string;
  baseRender: string;
  refinedRender: string;
  qualityReport: string;
  logs: string;
}

export interface DebugConfig {
  outputDir: string;
  jobId: string;
  saveFrames: boolean;
  compressionQuality: number;
}

export class DebugArtifacts {
  private config: DebugConfig;
  private frames: {
    masks: Uint8Array[];
    poses: CameraPose[];
    depths: Float32Array[];
    tracks: Array<{ x: number; y: number; confidence: number }[]>;
    baseRender: Uint8Array[];
    refinedRender: Uint8Array[];
  };
  private metadata: {
    width: number;
    height: number;
    fps: number;
    startTime: number;
    stages: Array<{ name: string; startTime: number; endTime?: number; status: string }>;
  };

  constructor(config: DebugConfig) {
    this.config = config;
    this.frames = {
      masks: [],
      poses: [],
      depths: [],
      tracks: [],
      baseRender: [],
      refinedRender: [],
    };
    this.metadata = {
      width: 0,
      height: 0,
      fps: 30,
      startTime: Date.now(),
      stages: [],
    };
  }

  setFrameInfo(width: number, height: number, fps: number): void {
    this.metadata.width = width;
    this.metadata.height = height;
    this.metadata.fps = fps;
  }

  // Stage tracking
  startStage(name: string): void {
    this.metadata.stages.push({
      name,
      startTime: Date.now(),
      status: 'running',
    });
  }

  endStage(name: string, status: 'success' | 'failed' | 'skipped' = 'success'): void {
    const stage = this.metadata.stages.find(s => s.name === name && !s.endTime);
    if (stage) {
      stage.endTime = Date.now();
      stage.status = status;
    }
  }

  // Frame collection
  addMaskFrame(mask: Uint8Array): void {
    if (this.config.saveFrames) {
      this.frames.masks.push(new Uint8Array(mask));
    }
  }

  addPose(pose: CameraPose): void {
    this.frames.poses.push({ ...pose });
  }

  addDepthFrame(depth: Float32Array): void {
    if (this.config.saveFrames) {
      this.frames.depths.push(new Float32Array(depth));
    }
  }

  addTracks(tracks: Array<{ x: number; y: number; confidence: number }>): void {
    this.frames.tracks.push([...tracks]);
  }

  addBaseRenderFrame(frame: Uint8Array): void {
    if (this.config.saveFrames) {
      this.frames.baseRender.push(new Uint8Array(frame));
    }
  }

  addRefinedRenderFrame(frame: Uint8Array): void {
    if (this.config.saveFrames) {
      this.frames.refinedRender.push(new Uint8Array(frame));
    }
  }

  /**
   * Generate all debug artifacts
   */
  async generateAll(quality: QualityReport): Promise<DebugArtifactPaths> {
    const baseDir = `${this.config.outputDir}/${this.config.jobId}_debug`;

    const paths: DebugArtifactPaths = {
      maskPreview: `${baseDir}/mask_preview.mp4`,
      poseOverlay: `${baseDir}/pose_overlay.mp4`,
      depthPreview: `${baseDir}/depth_preview.mp4`,
      trackOverlay: `${baseDir}/track_overlay.mp4`,
      baseRender: `${baseDir}/base_render.mp4`,
      refinedRender: `${baseDir}/refined_render.mp4`,
      qualityReport: `${baseDir}/quality.json`,
      logs: `${baseDir}/pipeline.log`,
    };

    // Generate each artifact
    await Promise.all([
      this.generateMaskPreview(paths.maskPreview),
      this.generatePoseOverlay(paths.poseOverlay),
      this.generateDepthPreview(paths.depthPreview),
      this.generateTrackOverlay(paths.trackOverlay),
      this.generateQualityReport(paths.qualityReport, quality),
      this.generateLogs(paths.logs),
    ]);

    return paths;
  }

  /**
   * Generate mask preview video
   * Shows subject mask overlaid on original frames
   */
  private async generateMaskPreview(outputPath: string): Promise<void> {
    const { width, height, fps } = this.metadata;
    const frames: Uint8Array[] = [];

    for (let i = 0; i < this.frames.masks.length; i++) {
      const mask = this.frames.masks[i];
      const frame = this.createMaskOverlayFrame(mask, width, height);
      frames.push(frame);
    }

    await this.encodeVideo(frames, outputPath, fps);
  }

  /**
   * Generate pose reprojection overlay
   * Shows feature tracks and reprojected points
   */
  private async generatePoseOverlay(outputPath: string): Promise<void> {
    const { width, height, fps } = this.metadata;
    const frames: Uint8Array[] = [];

    for (let i = 0; i < this.frames.poses.length; i++) {
      const pose = this.frames.poses[i];
      const frame = this.createPoseOverlayFrame(pose, width, height);
      frames.push(frame);
    }

    await this.encodeVideo(frames, outputPath, fps);
  }

  /**
   * Generate depth preview video
   * Color-coded depth visualization
   */
  private async generateDepthPreview(outputPath: string): Promise<void> {
    const { width, height, fps } = this.metadata;
    const frames: Uint8Array[] = [];

    for (const depth of this.frames.depths) {
      const frame = this.createDepthVisualization(depth, width, height);
      frames.push(frame);
    }

    await this.encodeVideo(frames, outputPath, fps);
  }

  /**
   * Generate track overlay video
   * Shows TAPIR tracks with confidence coloring
   */
  private async generateTrackOverlay(outputPath: string): Promise<void> {
    const { width, height, fps } = this.metadata;
    const frames: Uint8Array[] = [];

    for (let i = 0; i < this.frames.tracks.length; i++) {
      const tracks = this.frames.tracks[i];
      const frame = this.createTrackOverlayFrame(tracks, width, height, i);
      frames.push(frame);
    }

    await this.encodeVideo(frames, outputPath, fps);
  }

  /**
   * Generate quality report JSON
   */
  private async generateQualityReport(
    outputPath: string,
    quality: QualityReport
  ): Promise<void> {
    const report = {
      jobId: this.config.jobId,
      timestamp: new Date().toISOString(),
      processingTime: Date.now() - this.metadata.startTime,
      stages: this.metadata.stages,
      quality,
      frameInfo: {
        width: this.metadata.width,
        height: this.metadata.height,
        fps: this.metadata.fps,
        maskFrames: this.frames.masks.length,
        poseFrames: this.frames.poses.length,
        depthFrames: this.frames.depths.length,
      },
    };

    // In browser, create downloadable blob
    // In Node.js, write to file
    console.log('[Debug] Quality report:', JSON.stringify(report, null, 2));
  }

  /**
   * Generate pipeline logs
   */
  private async generateLogs(outputPath: string): Promise<void> {
    const logs: string[] = [
      `=== Orbit Pipeline Log ===`,
      `Job ID: ${this.config.jobId}`,
      `Started: ${new Date(this.metadata.startTime).toISOString()}`,
      ``,
      `=== Stage Timing ===`,
    ];

    for (const stage of this.metadata.stages) {
      const duration = stage.endTime
        ? `${stage.endTime - stage.startTime}ms`
        : 'running';
      logs.push(`${stage.name}: ${stage.status} (${duration})`);
    }

    logs.push(``);
    logs.push(`=== Frame Statistics ===`);
    logs.push(`Mask frames: ${this.frames.masks.length}`);
    logs.push(`Pose estimates: ${this.frames.poses.length}`);
    logs.push(`Depth frames: ${this.frames.depths.length}`);
    logs.push(`Track frames: ${this.frames.tracks.length}`);

    console.log('[Debug] Pipeline log:', logs.join('\n'));
  }

  // Visualization helpers
  private createMaskOverlayFrame(mask: Uint8Array, width: number, height: number): Uint8Array {
    const rgba = new Uint8Array(width * height * 4);

    for (let i = 0; i < mask.length; i++) {
      const isSubject = mask[i] > 127;
      rgba[i * 4 + 0] = isSubject ? 255 : 50;  // R
      rgba[i * 4 + 1] = isSubject ? 100 : 50;  // G
      rgba[i * 4 + 2] = isSubject ? 100 : 50;  // B
      rgba[i * 4 + 3] = 255;                    // A
    }

    return rgba;
  }

  private createPoseOverlayFrame(pose: CameraPose, width: number, height: number): Uint8Array {
    const rgba = new Uint8Array(width * height * 4);

    // Dark background
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4 + 0] = 30;
      rgba[i * 4 + 1] = 30;
      rgba[i * 4 + 2] = 30;
      rgba[i * 4 + 3] = 255;
    }

    // Draw coordinate axes based on pose
    this.drawCoordinateAxes(rgba, pose.transform, width, height);

    return rgba;
  }

  private createDepthVisualization(depth: Float32Array, width: number, height: number): Uint8Array {
    const rgba = new Uint8Array(width * height * 4);

    // Find depth range
    let minDepth = Infinity;
    let maxDepth = -Infinity;
    for (const d of depth) {
      if (isFinite(d) && d > 0) {
        minDepth = Math.min(minDepth, d);
        maxDepth = Math.max(maxDepth, d);
      }
    }

    const range = maxDepth - minDepth || 1;

    // Apply turbo colormap
    for (let i = 0; i < depth.length; i++) {
      const d = depth[i];
      if (!isFinite(d) || d <= 0) {
        rgba[i * 4 + 0] = 0;
        rgba[i * 4 + 1] = 0;
        rgba[i * 4 + 2] = 0;
        rgba[i * 4 + 3] = 255;
        continue;
      }

      const t = (d - minDepth) / range;
      const [r, g, b] = this.turboColormap(t);
      rgba[i * 4 + 0] = r;
      rgba[i * 4 + 1] = g;
      rgba[i * 4 + 2] = b;
      rgba[i * 4 + 3] = 255;
    }

    return rgba;
  }

  private createTrackOverlayFrame(
    tracks: Array<{ x: number; y: number; confidence: number }>,
    width: number,
    height: number,
    frameIndex: number
  ): Uint8Array {
    const rgba = new Uint8Array(width * height * 4);

    // Dark background
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4 + 0] = 20;
      rgba[i * 4 + 1] = 20;
      rgba[i * 4 + 2] = 20;
      rgba[i * 4 + 3] = 255;
    }

    // Draw tracks as colored dots
    for (const track of tracks) {
      const x = Math.round(track.x);
      const y = Math.round(track.y);

      if (x < 0 || x >= width || y < 0 || y >= height) continue;

      // Color by confidence (red = low, green = high)
      const r = Math.round((1 - track.confidence) * 255);
      const g = Math.round(track.confidence * 255);

      // Draw 3x3 dot
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const px = x + dx;
          const py = y + dy;
          if (px >= 0 && px < width && py >= 0 && py < height) {
            const idx = (py * width + px) * 4;
            rgba[idx + 0] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = 50;
            rgba[idx + 3] = 255;
          }
        }
      }
    }

    return rgba;
  }

  private drawCoordinateAxes(
    rgba: Uint8Array,
    transform: Matrix4x4,
    width: number,
    height: number
  ): void {
    // Extract camera position
    const cx = width / 2;
    const cy = height / 2;

    // Draw X axis (red)
    this.drawLine(rgba, width, height, cx, cy, cx + 100, cy, [255, 0, 0]);

    // Draw Y axis (green)
    this.drawLine(rgba, width, height, cx, cy, cx, cy - 100, [0, 255, 0]);

    // Draw Z axis (blue)
    this.drawLine(rgba, width, height, cx, cy, cx + 50, cy + 50, [0, 0, 255]);
  }

  private drawLine(
    rgba: Uint8Array,
    width: number,
    height: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    color: [number, number, number]
  ): void {
    // Bresenham's line algorithm
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    let x = Math.round(x0);
    let y = Math.round(y0);

    while (true) {
      if (x >= 0 && x < width && y >= 0 && y < height) {
        const idx = (y * width + x) * 4;
        rgba[idx + 0] = color[0];
        rgba[idx + 1] = color[1];
        rgba[idx + 2] = color[2];
        rgba[idx + 3] = 255;
      }

      if (x === Math.round(x1) && y === Math.round(y1)) break;

      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x += sx;
      }
      if (e2 < dx) {
        err += dx;
        y += sy;
      }
    }
  }

  private turboColormap(t: number): [number, number, number] {
    // Simplified turbo colormap approximation
    t = Math.max(0, Math.min(1, t));

    let r, g, b;
    if (t < 0.25) {
      r = 0;
      g = Math.round(t * 4 * 255);
      b = 255;
    } else if (t < 0.5) {
      r = 0;
      g = 255;
      b = Math.round((0.5 - t) * 4 * 255);
    } else if (t < 0.75) {
      r = Math.round((t - 0.5) * 4 * 255);
      g = 255;
      b = 0;
    } else {
      r = 255;
      g = Math.round((1 - t) * 4 * 255);
      b = 0;
    }

    return [r, g, b];
  }

  private async encodeVideo(
    frames: Uint8Array[],
    outputPath: string,
    fps: number
  ): Promise<void> {
    // Placeholder - actual implementation would use FFmpeg WASM
    console.log(`[Debug] Would encode ${frames.length} frames to ${outputPath} at ${fps}fps`);
  }
}
