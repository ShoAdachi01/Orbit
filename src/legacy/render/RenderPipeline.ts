/**
 * Base Render Pipeline
 * Renders OrbitScene along camera path to video frames
 */

import { CameraPath, CameraKeyframe, Vector3, Quaternion, OrbitBounds } from '../schemas/types';
import { GaussianSplat } from '../reconstruction/BackgroundReconstruction';
import { Subject4DSplat } from '../reconstruction/SubjectReconstruction';

export interface RenderConfig {
  width: number;
  height: number;
  fps: number;
  antialiasing: boolean;
  backgroundColor: [number, number, number, number];
}

const DEFAULT_CONFIG: RenderConfig = {
  width: 1920,
  height: 1080,
  fps: 30,
  antialiasing: true,
  backgroundColor: [0.1, 0.1, 0.1, 1.0],
};

export interface RenderFrame {
  data: Uint8Array;
  width: number;
  height: number;
  timestamp: number;
  frameIndex: number;
}

export class RenderPipeline {
  private config: RenderConfig;
  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;

  constructor(config: Partial<RenderConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize offscreen rendering context
   */
  async initialize(): Promise<void> {
    this.canvas = new OffscreenCanvas(this.config.width, this.config.height);
    this.ctx = this.canvas.getContext('2d');

    if (!this.ctx) {
      throw new Error('Failed to create 2D rendering context');
    }
  }

  /**
   * Render scene along camera path
   */
  async render(
    backgroundSplat: GaussianSplat,
    subjectSplat: Subject4DSplat | null,
    cameraPath: CameraPath,
    bounds: OrbitBounds,
    onProgress?: (progress: number, frame: number, total: number) => void
  ): Promise<RenderFrame[]> {
    if (!this.canvas || !this.ctx) {
      throw new Error('Pipeline not initialized');
    }

    // Resample camera path to target FPS
    const frames = this.resampleCameraPath(cameraPath);
    const renderedFrames: RenderFrame[] = [];

    for (let i = 0; i < frames.length; i++) {
      const keyframe = frames[i];
      const timestamp = i / this.config.fps;

      // Get subject splat for this frame
      const subjectFrameSplat = subjectSplat?.frameSplats.get(i) || null;

      // Render frame
      const frameData = await this.renderFrame(
        backgroundSplat,
        subjectFrameSplat,
        keyframe,
        bounds
      );

      renderedFrames.push({
        data: frameData,
        width: this.config.width,
        height: this.config.height,
        timestamp,
        frameIndex: i,
      });

      onProgress?.(((i + 1) / frames.length) * 100, i + 1, frames.length);
    }

    return renderedFrames;
  }

  /**
   * Resample camera path to target FPS using slerp/linear interpolation
   */
  private resampleCameraPath(path: CameraPath): CameraKeyframe[] {
    if (path.keyframes.length === 0) return [];
    if (path.keyframes.length === 1) return [path.keyframes[0]];

    const duration = path.keyframes[path.keyframes.length - 1].timestamp - path.keyframes[0].timestamp;
    const numFrames = Math.ceil(duration * this.config.fps);
    const result: CameraKeyframe[] = [];

    for (let i = 0; i < numFrames; i++) {
      const t = i / this.config.fps + path.keyframes[0].timestamp;
      const keyframe = this.interpolateKeyframe(path.keyframes, t);
      result.push(keyframe);
    }

    return result;
  }

  /**
   * Interpolate keyframe at time t
   */
  private interpolateKeyframe(keyframes: CameraKeyframe[], t: number): CameraKeyframe {
    // Find surrounding keyframes
    let k0 = keyframes[0];
    let k1 = keyframes[keyframes.length - 1];

    for (let i = 0; i < keyframes.length - 1; i++) {
      if (keyframes[i].timestamp <= t && keyframes[i + 1].timestamp > t) {
        k0 = keyframes[i];
        k1 = keyframes[i + 1];
        break;
      }
    }

    // Compute interpolation factor
    const dt = k1.timestamp - k0.timestamp;
    const factor = dt > 0 ? (t - k0.timestamp) / dt : 0;

    // Interpolate position (linear)
    const position: Vector3 = [
      k0.position[0] + (k1.position[0] - k0.position[0]) * factor,
      k0.position[1] + (k1.position[1] - k0.position[1]) * factor,
      k0.position[2] + (k1.position[2] - k0.position[2]) * factor,
    ];

    // Interpolate rotation (slerp)
    const rotation = this.slerp(k0.rotation, k1.rotation, factor);

    // Interpolate FOV (linear)
    const fov = k0.fov !== undefined && k1.fov !== undefined
      ? k0.fov + (k1.fov - k0.fov) * factor
      : k0.fov || k1.fov || 45;

    return { timestamp: t, position, rotation, fov };
  }

  /**
   * Spherical linear interpolation for quaternions
   */
  private slerp(q0: Quaternion, q1: Quaternion, t: number): Quaternion {
    // Normalize inputs
    q0 = this.normalizeQuaternion(q0);
    q1 = this.normalizeQuaternion(q1);

    // Compute dot product
    let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];

    // If negative dot, negate one quaternion to take shorter path
    if (dot < 0) {
      q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
      dot = -dot;
    }

    // If very close, use linear interpolation
    if (dot > 0.9995) {
      return this.normalizeQuaternion([
        q0[0] + t * (q1[0] - q0[0]),
        q0[1] + t * (q1[1] - q0[1]),
        q0[2] + t * (q1[2] - q0[2]),
        q0[3] + t * (q1[3] - q0[3]),
      ]);
    }

    // Compute slerp
    const theta0 = Math.acos(dot);
    const theta = theta0 * t;
    const sinTheta = Math.sin(theta);
    const sinTheta0 = Math.sin(theta0);

    const s0 = Math.cos(theta) - dot * sinTheta / sinTheta0;
    const s1 = sinTheta / sinTheta0;

    return [
      s0 * q0[0] + s1 * q1[0],
      s0 * q0[1] + s1 * q1[1],
      s0 * q0[2] + s1 * q1[2],
      s0 * q0[3] + s1 * q1[3],
    ];
  }

  private normalizeQuaternion(q: Quaternion): Quaternion {
    const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (len === 0) return [0, 0, 0, 1];
    return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
  }

  /**
   * Render single frame (placeholder - actual implementation would use WebGPU)
   */
  private async renderFrame(
    backgroundSplat: GaussianSplat,
    subjectSplat: GaussianSplat | null,
    keyframe: CameraKeyframe,
    bounds: OrbitBounds
  ): Promise<Uint8Array> {
    if (!this.ctx || !this.canvas) {
      throw new Error('Context not initialized');
    }

    const { width, height } = this.config;

    // Clear with background color
    const [r, g, b, a] = this.config.backgroundColor;
    this.ctx.fillStyle = `rgba(${r * 255}, ${g * 255}, ${b * 255}, ${a})`;
    this.ctx.fillRect(0, 0, width, height);

    // Compute view-projection matrix from keyframe
    const viewMatrix = this.computeViewMatrix(keyframe.position, keyframe.rotation);
    const projMatrix = this.computeProjectionMatrix(keyframe.fov || 45, width / height);

    // Render background splats
    this.renderSplats(backgroundSplat, viewMatrix, projMatrix);

    // Render subject splats (composited on top)
    if (subjectSplat) {
      this.renderSplats(subjectSplat, viewMatrix, projMatrix);
    }

    // Get image data
    const imageData = this.ctx.getImageData(0, 0, width, height);
    return new Uint8Array(imageData.data);
  }

  /**
   * Render splats using software rasterization (placeholder)
   */
  private renderSplats(
    splat: GaussianSplat,
    viewMatrix: Float32Array,
    projMatrix: Float32Array
  ): void {
    if (!this.ctx) return;

    const { width, height } = this.config;

    // Simple point rendering for now (actual impl would use proper splatting)
    for (let i = 0; i < splat.count; i++) {
      const px = splat.positions[i * 3];
      const py = splat.positions[i * 3 + 1];
      const pz = splat.positions[i * 3 + 2];

      // Transform to clip space
      const clipPos = this.transformPoint(px, py, pz, viewMatrix, projMatrix);

      if (clipPos[3] <= 0) continue; // Behind camera

      // Perspective divide
      const ndcX = clipPos[0] / clipPos[3];
      const ndcY = clipPos[1] / clipPos[3];

      // Convert to screen space
      const screenX = (ndcX + 1) * 0.5 * width;
      const screenY = (1 - ndcY) * 0.5 * height;

      if (screenX < 0 || screenX >= width || screenY < 0 || screenY >= height) {
        continue;
      }

      // Get color and opacity
      const r = Math.floor(splat.colors[i * 4 + 0] * 255);
      const g = Math.floor(splat.colors[i * 4 + 1] * 255);
      const b = Math.floor(splat.colors[i * 4 + 2] * 255);
      const opacity = splat.opacities[i];

      // Compute screen-space size from scale
      const scale = Math.max(splat.scales[i * 3], splat.scales[i * 3 + 1]);
      const screenSize = Math.max(1, scale * 100 / Math.max(0.1, clipPos[3]));

      // Draw splat as circle
      this.ctx.globalAlpha = opacity;
      this.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      this.ctx.beginPath();
      this.ctx.arc(screenX, screenY, screenSize, 0, Math.PI * 2);
      this.ctx.fill();
    }

    this.ctx.globalAlpha = 1;
  }

  /**
   * Compute view matrix from position and rotation
   */
  private computeViewMatrix(position: Vector3, rotation: Quaternion): Float32Array {
    const [qx, qy, qz, qw] = rotation;

    // Quaternion to rotation matrix
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;

    const rotMatrix = new Float32Array([
      1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
      2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
      2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
      0, 0, 0, 1,
    ]);

    // Apply translation
    const [px, py, pz] = position;
    rotMatrix[12] = -(rotMatrix[0] * px + rotMatrix[4] * py + rotMatrix[8] * pz);
    rotMatrix[13] = -(rotMatrix[1] * px + rotMatrix[5] * py + rotMatrix[9] * pz);
    rotMatrix[14] = -(rotMatrix[2] * px + rotMatrix[6] * py + rotMatrix[10] * pz);

    return rotMatrix;
  }

  /**
   * Compute perspective projection matrix
   */
  private computeProjectionMatrix(fov: number, aspect: number): Float32Array {
    const near = 0.1;
    const far = 1000;
    const f = 1 / Math.tan((fov * Math.PI / 180) / 2);

    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0,
    ]);
  }

  /**
   * Transform point through view-projection
   */
  private transformPoint(
    x: number, y: number, z: number,
    viewMatrix: Float32Array,
    projMatrix: Float32Array
  ): [number, number, number, number] {
    // View transform
    const vx = viewMatrix[0] * x + viewMatrix[4] * y + viewMatrix[8] * z + viewMatrix[12];
    const vy = viewMatrix[1] * x + viewMatrix[5] * y + viewMatrix[9] * z + viewMatrix[13];
    const vz = viewMatrix[2] * x + viewMatrix[6] * y + viewMatrix[10] * z + viewMatrix[14];
    const vw = viewMatrix[3] * x + viewMatrix[7] * y + viewMatrix[11] * z + viewMatrix[15];

    // Projection transform
    const px = projMatrix[0] * vx + projMatrix[4] * vy + projMatrix[8] * vz + projMatrix[12] * vw;
    const py = projMatrix[1] * vx + projMatrix[5] * vy + projMatrix[9] * vz + projMatrix[13] * vw;
    const pz = projMatrix[2] * vx + projMatrix[6] * vy + projMatrix[10] * vz + projMatrix[14] * vw;
    const pw = projMatrix[3] * vx + projMatrix[7] * vy + projMatrix[11] * vz + projMatrix[15] * vw;

    return [px, py, pz, pw];
  }

  destroy(): void {
    this.canvas = null;
    this.ctx = null;
  }
}
