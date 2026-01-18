/**
 * Camera Path Editor
 * UI component for authoring orbit camera paths
 */

import {
  CameraPath,
  CameraKeyframe,
  Vector3,
  Quaternion,
  OrbitBounds,
  DEFAULT_ORBIT_BOUNDS,
} from '../schemas/types';
import { OrbitCamera } from './Camera';

export interface PathEditorConfig {
  /** Target FPS for path playback */
  targetFps: number;
  /** Maximum path duration in seconds */
  maxDuration: number;
  /** Orbit bounds */
  bounds: OrbitBounds;
}

const DEFAULT_CONFIG: PathEditorConfig = {
  targetFps: 30,
  maxDuration: 10,
  bounds: DEFAULT_ORBIT_BOUNDS,
};

export type PathEditorMode = 'idle' | 'recording' | 'playing' | 'editing';

export interface PathEditorState {
  mode: PathEditorMode;
  path: CameraPath;
  currentTime: number;
  selectedKeyframe: number | null;
  isPlaying: boolean;
}

export class CameraPathEditor {
  private config: PathEditorConfig;
  private state: PathEditorState;
  private camera: OrbitCamera;
  private animationFrame: number | null = null;
  private playStartTime: number = 0;
  private playStartOffset: number = 0;

  private onStateChange?: (state: PathEditorState) => void;
  private onCameraUpdate?: (camera: OrbitCamera) => void;

  constructor(
    camera: OrbitCamera,
    config: Partial<PathEditorConfig> = {},
    callbacks?: {
      onStateChange?: (state: PathEditorState) => void;
      onCameraUpdate?: (camera: OrbitCamera) => void;
    }
  ) {
    this.camera = camera;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.onStateChange = callbacks?.onStateChange;
    this.onCameraUpdate = callbacks?.onCameraUpdate;

    this.state = {
      mode: 'idle',
      path: {
        keyframes: [],
        interpolation: {
          rotation: 'slerp',
          translation: 'linear',
          fov: 'linear',
        },
        targetFps: this.config.targetFps,
      },
      currentTime: 0,
      selectedKeyframe: null,
      isPlaying: false,
    };
  }

  getState(): PathEditorState {
    return { ...this.state };
  }

  getPath(): CameraPath {
    return { ...this.state.path };
  }

  /**
   * Add a keyframe at the current camera position
   */
  addKeyframe(timestamp?: number): number {
    const t = timestamp ?? this.state.currentTime;

    // Ensure timestamp is within bounds
    const clampedTime = Math.max(0, Math.min(t, this.config.maxDuration));

    // Get current camera state
    const keyframe: CameraKeyframe = {
      timestamp: clampedTime,
      position: this.camera.position,
      rotation: this.positionToQuaternion(this.camera.yaw, this.camera.pitch, this.camera.roll),
      fov: 45, // Default FOV
    };

    // Insert in sorted order by timestamp
    const keyframes = [...this.state.path.keyframes];
    let insertIndex = keyframes.findIndex(k => k.timestamp > clampedTime);
    if (insertIndex === -1) {
      insertIndex = keyframes.length;
    }
    keyframes.splice(insertIndex, 0, keyframe);

    this.state.path.keyframes = keyframes;
    this.state.selectedKeyframe = insertIndex;
    this.emitStateChange();

    return insertIndex;
  }

  /**
   * Update a keyframe
   */
  updateKeyframe(index: number, updates: Partial<CameraKeyframe>): void {
    if (index < 0 || index >= this.state.path.keyframes.length) return;

    const keyframe = this.state.path.keyframes[index];
    this.state.path.keyframes[index] = {
      ...keyframe,
      ...updates,
    };

    // Re-sort if timestamp changed
    if (updates.timestamp !== undefined) {
      this.state.path.keyframes.sort((a, b) => a.timestamp - b.timestamp);
      this.state.selectedKeyframe = this.state.path.keyframes.findIndex(
        k => k === this.state.path.keyframes[index]
      );
    }

    this.emitStateChange();
  }

  /**
   * Delete a keyframe
   */
  deleteKeyframe(index: number): void {
    if (index < 0 || index >= this.state.path.keyframes.length) return;

    this.state.path.keyframes.splice(index, 1);

    if (this.state.selectedKeyframe === index) {
      this.state.selectedKeyframe = null;
    } else if (this.state.selectedKeyframe !== null && this.state.selectedKeyframe > index) {
      this.state.selectedKeyframe--;
    }

    this.emitStateChange();
  }

  /**
   * Select a keyframe
   */
  selectKeyframe(index: number | null): void {
    if (index !== null && (index < 0 || index >= this.state.path.keyframes.length)) {
      return;
    }

    this.state.selectedKeyframe = index;

    // Move camera to selected keyframe
    if (index !== null) {
      const keyframe = this.state.path.keyframes[index];
      this.setCameraFromKeyframe(keyframe);
      this.state.currentTime = keyframe.timestamp;
    }

    this.emitStateChange();
  }

  /**
   * Start recording camera movements
   */
  startRecording(): void {
    if (this.state.mode === 'recording') return;

    this.state.mode = 'recording';
    this.state.path.keyframes = [];
    this.state.currentTime = 0;
    this.addKeyframe(0);

    this.emitStateChange();
  }

  /**
   * Stop recording
   */
  stopRecording(): void {
    if (this.state.mode !== 'recording') return;

    // Add final keyframe if moved
    if (this.state.path.keyframes.length > 0) {
      const last = this.state.path.keyframes[this.state.path.keyframes.length - 1];
      const currentPos = this.camera.position;
      if (
        Math.abs(currentPos[0] - last.position[0]) > 0.01 ||
        Math.abs(currentPos[1] - last.position[1]) > 0.01 ||
        Math.abs(currentPos[2] - last.position[2]) > 0.01
      ) {
        this.addKeyframe(this.state.currentTime);
      }
    }

    this.state.mode = 'idle';
    this.emitStateChange();
  }

  /**
   * Record current position during recording mode
   */
  recordFrame(deltaTime: number): void {
    if (this.state.mode !== 'recording') return;

    this.state.currentTime += deltaTime;

    if (this.state.currentTime >= this.config.maxDuration) {
      this.stopRecording();
      return;
    }

    // Sample camera position at intervals
    const lastKeyframe = this.state.path.keyframes[this.state.path.keyframes.length - 1];
    const timeSinceLast = this.state.currentTime - lastKeyframe.timestamp;

    // Add keyframe every 0.5 seconds or on significant movement
    if (timeSinceLast >= 0.5 || this.hasSignificantMovement(lastKeyframe)) {
      this.addKeyframe(this.state.currentTime);
    }
  }

  /**
   * Start playback
   */
  play(): void {
    if (this.state.path.keyframes.length < 2) return;
    if (this.state.isPlaying) return;

    this.state.isPlaying = true;
    this.state.mode = 'playing';
    this.playStartTime = performance.now();
    this.playStartOffset = this.state.currentTime;

    this.animationFrame = requestAnimationFrame(this.playbackLoop.bind(this));
    this.emitStateChange();
  }

  /**
   * Pause playback
   */
  pause(): void {
    if (!this.state.isPlaying) return;

    this.state.isPlaying = false;
    this.state.mode = 'idle';

    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    this.emitStateChange();
  }

  /**
   * Stop playback and reset to start
   */
  stop(): void {
    this.pause();
    this.seekTo(0);
  }

  /**
   * Seek to a specific time
   */
  seekTo(time: number): void {
    const duration = this.getDuration();
    this.state.currentTime = Math.max(0, Math.min(time, duration));

    // Update camera to match time
    const keyframe = this.interpolateKeyframe(this.state.currentTime);
    if (keyframe) {
      this.setCameraFromKeyframe(keyframe);
    }

    this.playStartOffset = this.state.currentTime;
    this.playStartTime = performance.now();

    this.emitStateChange();
  }

  /**
   * Get total path duration
   */
  getDuration(): number {
    if (this.state.path.keyframes.length === 0) return 0;
    return this.state.path.keyframes[this.state.path.keyframes.length - 1].timestamp;
  }

  /**
   * Clear all keyframes
   */
  clear(): void {
    this.stop();
    this.state.path.keyframes = [];
    this.state.selectedKeyframe = null;
    this.state.currentTime = 0;
    this.emitStateChange();
  }

  /**
   * Generate a simple orbit path
   */
  generateOrbitPath(duration: number, revolutions: number = 1): void {
    this.clear();

    const numKeyframes = Math.ceil(duration * 4); // 4 keyframes per second
    const bounds = this.config.bounds;

    for (let i = 0; i <= numKeyframes; i++) {
      const t = i / numKeyframes;
      const timestamp = t * duration;

      // Sinusoidal orbit motion
      const yaw = Math.sin(t * Math.PI * 2 * revolutions) * bounds.maxYaw;
      const pitch = Math.sin(t * Math.PI * 4 * revolutions) * bounds.maxPitch * 0.5;

      // Compute position from orbit angles
      const distance = 5;
      const yawRad = yaw * (Math.PI / 180);
      const pitchRad = pitch * (Math.PI / 180);

      const position: Vector3 = [
        distance * Math.sin(yawRad) * Math.cos(pitchRad),
        distance * Math.sin(pitchRad),
        distance * Math.cos(yawRad) * Math.cos(pitchRad),
      ];

      const rotation = this.positionToQuaternion(yaw, pitch, 0);

      this.state.path.keyframes.push({
        timestamp,
        position,
        rotation,
        fov: 45,
      });
    }

    this.emitStateChange();
  }

  /**
   * Export path as JSON
   */
  exportPath(): string {
    return JSON.stringify(this.state.path, null, 2);
  }

  /**
   * Import path from JSON
   */
  importPath(json: string): boolean {
    try {
      const path = JSON.parse(json) as CameraPath;

      // Validate
      if (!Array.isArray(path.keyframes)) return false;

      this.clear();
      this.state.path = path;
      this.emitStateChange();

      return true;
    } catch {
      return false;
    }
  }

  // Private methods
  private playbackLoop(): void {
    if (!this.state.isPlaying) return;

    const elapsed = (performance.now() - this.playStartTime) / 1000;
    const newTime = this.playStartOffset + elapsed;
    const duration = this.getDuration();

    if (newTime >= duration) {
      // Loop or stop
      this.state.currentTime = duration;
      this.pause();
      return;
    }

    this.state.currentTime = newTime;

    // Update camera
    const keyframe = this.interpolateKeyframe(newTime);
    if (keyframe) {
      this.setCameraFromKeyframe(keyframe);
      this.onCameraUpdate?.(this.camera);
    }

    this.emitStateChange();
    this.animationFrame = requestAnimationFrame(this.playbackLoop.bind(this));
  }

  private interpolateKeyframe(time: number): CameraKeyframe | null {
    const keyframes = this.state.path.keyframes;
    if (keyframes.length === 0) return null;
    if (keyframes.length === 1) return keyframes[0];

    // Find surrounding keyframes
    let k0 = keyframes[0];
    let k1 = keyframes[keyframes.length - 1];

    for (let i = 0; i < keyframes.length - 1; i++) {
      if (keyframes[i].timestamp <= time && keyframes[i + 1].timestamp > time) {
        k0 = keyframes[i];
        k1 = keyframes[i + 1];
        break;
      }
    }

    // Compute interpolation factor
    const dt = k1.timestamp - k0.timestamp;
    const factor = dt > 0 ? (time - k0.timestamp) / dt : 0;

    // Interpolate position (linear)
    const position: Vector3 = [
      k0.position[0] + (k1.position[0] - k0.position[0]) * factor,
      k0.position[1] + (k1.position[1] - k0.position[1]) * factor,
      k0.position[2] + (k1.position[2] - k0.position[2]) * factor,
    ];

    // Interpolate rotation (slerp)
    const rotation = this.slerp(k0.rotation, k1.rotation, factor);

    // Interpolate FOV (linear)
    const fov = (k0.fov || 45) + ((k1.fov || 45) - (k0.fov || 45)) * factor;

    return { timestamp: time, position, rotation, fov };
  }

  private slerp(q0: Quaternion, q1: Quaternion, t: number): Quaternion {
    // Normalize
    const normalize = (q: Quaternion): Quaternion => {
      const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      return len > 0 ? [q[0] / len, q[1] / len, q[2] / len, q[3] / len] : [0, 0, 0, 1];
    };

    q0 = normalize(q0);
    q1 = normalize(q1);

    let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];

    if (dot < 0) {
      q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
      dot = -dot;
    }

    if (dot > 0.9995) {
      return normalize([
        q0[0] + t * (q1[0] - q0[0]),
        q0[1] + t * (q1[1] - q0[1]),
        q0[2] + t * (q1[2] - q0[2]),
        q0[3] + t * (q1[3] - q0[3]),
      ]);
    }

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

  private setCameraFromKeyframe(keyframe: CameraKeyframe): void {
    const [yaw, pitch, roll] = this.quaternionToEuler(keyframe.rotation);
    this.camera.yaw = yaw;
    this.camera.pitch = pitch;
    this.camera.roll = roll;

    if (keyframe.fov) {
      this.camera.setFov(keyframe.fov);
    }
  }

  private positionToQuaternion(yaw: number, pitch: number, roll: number): Quaternion {
    // Convert Euler angles (degrees) to quaternion
    const yr = yaw * (Math.PI / 180) / 2;
    const pr = pitch * (Math.PI / 180) / 2;
    const rr = roll * (Math.PI / 180) / 2;

    const cy = Math.cos(yr);
    const sy = Math.sin(yr);
    const cp = Math.cos(pr);
    const sp = Math.sin(pr);
    const cr = Math.cos(rr);
    const sr = Math.sin(rr);

    return [
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
      cr * cp * cy + sr * sp * sy,
    ];
  }

  private quaternionToEuler(q: Quaternion): [number, number, number] {
    const [x, y, z, w] = q;

    // Roll (x-axis rotation)
    const sinr_cosp = 2 * (w * x + y * z);
    const cosr_cosp = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    const sinp = 2 * (w * y - z * x);
    const pitch = Math.abs(sinp) >= 1
      ? Math.sign(sinp) * Math.PI / 2
      : Math.asin(sinp);

    // Yaw (z-axis rotation)
    const siny_cosp = 2 * (w * z + x * y);
    const cosy_cosp = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(siny_cosp, cosy_cosp);

    return [
      yaw * (180 / Math.PI),
      pitch * (180 / Math.PI),
      roll * (180 / Math.PI),
    ];
  }

  private hasSignificantMovement(keyframe: CameraKeyframe): boolean {
    const threshold = 0.1; // 10cm
    const pos = this.camera.position;
    const dx = pos[0] - keyframe.position[0];
    const dy = pos[1] - keyframe.position[1];
    const dz = pos[2] - keyframe.position[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz) > threshold;
  }

  private emitStateChange(): void {
    this.onStateChange?.({ ...this.state });
  }

  destroy(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
    }
  }
}
