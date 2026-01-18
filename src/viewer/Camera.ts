/**
 * Orbit Camera Controller
 * Implements bounded orbit camera with PRD constraints
 */

import {
  Vector3,
  Quaternion,
  Matrix4x4,
  OrbitBounds,
  DEFAULT_ORBIT_BOUNDS,
  CameraIntrinsics,
} from '../schemas/types';

export class OrbitCamera {
  private _position: Vector3 = [0, 0, 5];
  private _target: Vector3 = [0, 0, 0];
  private _up: Vector3 = [0, 1, 0];

  private _yaw = 0;    // degrees
  private _pitch = 0;  // degrees
  private _roll = 0;   // degrees
  private _distance = 5;
  private _translation: Vector3 = [0, 0, 0];

  private _fov = 45;   // degrees
  private _aspect = 1;
  private _near = 0.1;
  private _far = 1000;

  private _bounds: OrbitBounds;
  private _medianSceneDepth = 5;

  constructor(bounds: OrbitBounds = DEFAULT_ORBIT_BOUNDS) {
    this._bounds = { ...bounds };
  }

  // Setters with bounds enforcement
  set yaw(value: number) {
    this._yaw = this.clamp(value, -this._bounds.maxYaw, this._bounds.maxYaw);
    this.updatePosition();
  }

  set pitch(value: number) {
    this._pitch = this.clamp(value, -this._bounds.maxPitch, this._bounds.maxPitch);
    this.updatePosition();
  }

  set roll(value: number) {
    this._roll = this.clamp(value, -this._bounds.maxRoll, this._bounds.maxRoll);
  }

  set translation(value: Vector3) {
    const maxTrans = this.getMaxTranslation();
    this._translation = [
      this.clamp(value[0], -maxTrans, maxTrans),
      this.clamp(value[1], -maxTrans, maxTrans),
      this.clamp(value[2], -maxTrans, maxTrans),
    ];
    this.updatePosition();
  }

  get yaw(): number { return this._yaw; }
  get pitch(): number { return this._pitch; }
  get roll(): number { return this._roll; }
  get translation(): Vector3 { return [...this._translation] as Vector3; }
  get position(): Vector3 { return [...this._position] as Vector3; }
  get target(): Vector3 { return [...this._target] as Vector3; }

  setTarget(target: Vector3): void {
    this._target = [...target] as Vector3;
    this.updatePosition();
  }

  setMedianSceneDepth(depth: number): void {
    this._medianSceneDepth = depth;
  }

  setDistance(distance: number): void {
    this._distance = Math.max(0.1, distance);
    this.updatePosition();
  }

  setAspect(aspect: number): void {
    this._aspect = aspect;
  }

  setFov(fov: number): void {
    this._fov = this.clamp(fov, 10, 120);
  }

  setBounds(bounds: Partial<OrbitBounds>): void {
    this._bounds = { ...this._bounds, ...bounds };
    // Re-apply current values to enforce new bounds
    this.yaw = this._yaw;
    this.pitch = this._pitch;
    this.roll = this._roll;
    this.translation = this._translation;
  }

  /** Apply camera intrinsics from scene */
  applyIntrinsics(intrinsics: CameraIntrinsics): void {
    // Convert focal length to FOV
    this._fov = 2 * Math.atan(intrinsics.height / (2 * intrinsics.fy)) * (180 / Math.PI);
    this._aspect = intrinsics.width / intrinsics.height;
  }

  /** Get view matrix (world to camera) */
  getViewMatrix(): Matrix4x4 {
    return this.lookAt(this._position, this._target, this._up);
  }

  /** Get projection matrix */
  getProjectionMatrix(): Matrix4x4 {
    return this.perspective(
      this._fov * (Math.PI / 180),
      this._aspect,
      this._near,
      this._far
    );
  }

  /** Get view-projection matrix */
  getViewProjectionMatrix(): Matrix4x4 {
    const view = this.getViewMatrix();
    const proj = this.getProjectionMatrix();
    return this.multiplyMatrices(proj, view);
  }

  /** Orbit by delta angles (in degrees) */
  orbit(deltaYaw: number, deltaPitch: number): void {
    this.yaw = this._yaw + deltaYaw;
    this.pitch = this._pitch + deltaPitch;
  }

  /** Pan camera */
  pan(deltaX: number, deltaY: number): void {
    const right = this.normalize(this.cross(
      this.subtract(this._target, this._position),
      this._up
    ));
    const up = this._up;

    this.translation = [
      this._translation[0] + deltaX * right[0] + deltaY * up[0],
      this._translation[1] + deltaX * right[1] + deltaY * up[1],
      this._translation[2] + deltaX * right[2] + deltaY * up[2],
    ];
  }

  /** Reset to initial state */
  reset(): void {
    this._yaw = 0;
    this._pitch = 0;
    this._roll = 0;
    this._translation = [0, 0, 0];
    this.updatePosition();
  }

  // Private helpers
  private getMaxTranslation(): number {
    const depthBasedMax = this._medianSceneDepth * (this._bounds.maxTranslationDepthPercent / 100);
    return Math.min(this._bounds.maxTranslation, depthBasedMax);
  }

  private updatePosition(): void {
    const yawRad = this._yaw * (Math.PI / 180);
    const pitchRad = this._pitch * (Math.PI / 180);

    // Spherical coordinates to Cartesian
    const x = this._distance * Math.sin(yawRad) * Math.cos(pitchRad);
    const y = this._distance * Math.sin(pitchRad);
    const z = this._distance * Math.cos(yawRad) * Math.cos(pitchRad);

    this._position = [
      this._target[0] + x + this._translation[0],
      this._target[1] + y + this._translation[1],
      this._target[2] + z + this._translation[2],
    ];
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  private lookAt(eye: Vector3, target: Vector3, up: Vector3): Matrix4x4 {
    const zAxis = this.normalize(this.subtract(eye, target));
    const xAxis = this.normalize(this.cross(up, zAxis));
    const yAxis = this.cross(zAxis, xAxis);

    return [
      xAxis[0], yAxis[0], zAxis[0], 0,
      xAxis[1], yAxis[1], zAxis[1], 0,
      xAxis[2], yAxis[2], zAxis[2], 0,
      -this.dot(xAxis, eye), -this.dot(yAxis, eye), -this.dot(zAxis, eye), 1,
    ];
  }

  private perspective(fovy: number, aspect: number, near: number, far: number): Matrix4x4 {
    const f = 1 / Math.tan(fovy / 2);
    const rangeInv = 1 / (near - far);

    return [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (near + far) * rangeInv, -1,
      0, 0, near * far * rangeInv * 2, 0,
    ];
  }

  private multiplyMatrices(a: Matrix4x4, b: Matrix4x4): Matrix4x4 {
    const result: number[] = new Array(16);

    // Column-major matrix multiply (out = a * b) to match WGSL.
    for (let col = 0; col < 4; col++) {
      const b0 = b[col * 4 + 0];
      const b1 = b[col * 4 + 1];
      const b2 = b[col * 4 + 2];
      const b3 = b[col * 4 + 3];

      result[col * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      result[col * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      result[col * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      result[col * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }

    return result as Matrix4x4;
  }

  private subtract(a: Vector3, b: Vector3): Vector3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  private cross(a: Vector3, b: Vector3): Vector3 {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  private dot(a: Vector3, b: Vector3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  private normalize(v: Vector3): Vector3 {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len === 0) return [0, 0, 0];
    return [v[0] / len, v[1] / len, v[2] / len];
  }
}
