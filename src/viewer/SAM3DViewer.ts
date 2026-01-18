import {
  mat4Identity,
  mat4Multiply,
  mat4LookAt,
  mat4Perspective,
  mat4FromTranslation,
  vec3Normalize,
  vec3Sub,
  quatFromAxisAngle,
  quatRotateVector,
} from '../utils/math';
import type { Matrix4x4, Vector3 } from '../schemas/types';

export interface SAM3DMesh {
  format: 'sam3d-body';
  vertices: Float32Array;
  faces: Uint32Array;
  bounds?: {
    center: Vector3;
    radius: number;
  };
}

export interface MeshBounds {
  center: Vector3;
  radius: number;
}

export class SAM3DViewer {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private positionBuffer: WebGLBuffer | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private indexCount = 0;
  private indexType: number | null = null;
  private bounds: MeshBounds | null = null;
  private animationId: number | null = null;

  private yaw = 0;
  private pitch = 0;
  private roll = 0;
  private distance = 5;

  private modelMatrix: Matrix4x4 = mat4Identity();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async initialize(): Promise<void> {
    const gl = this.canvas.getContext('webgl');
    if (!gl) {
      throw new Error('WebGL not supported');
    }
    this.gl = gl;

    const vertexShader = this.compileShader(
      gl.VERTEX_SHADER,
      `
        attribute vec3 aPosition;
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        void main() {
          gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);
        }
      `
    );

    const fragmentShader = this.compileShader(
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        uniform vec3 uColor;
        void main() {
          gl_FragColor = vec4(uColor, 1.0);
        }
      `
    );

    this.program = gl.createProgram();
    if (!this.program) {
      throw new Error('Failed to create shader program');
    }

    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(this.program) || 'Unknown shader error';
      throw new Error(log);
    }

    gl.useProgram(this.program);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);

    this.positionBuffer = gl.createBuffer();
    this.indexBuffer = gl.createBuffer();
  }

  start(): void {
    if (this.animationId) return;
    const loop = () => {
      this.render();
      this.animationId = requestAnimationFrame(loop);
    };
    loop();
  }

  stop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  async loadFromUrl(url: string): Promise<MeshBounds | null> {
    if (!url) return null;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load mesh: ${response.statusText}`);
    }
    const data = await response.json();
    const vertices = new Float32Array(data.vertices || []);
    const faces = new Uint32Array(data.faces || []);
    const mesh: SAM3DMesh = {
      format: data.format || 'sam3d-body',
      vertices,
      faces,
      bounds: data.bounds,
    };
    this.setMesh(mesh);
    return this.bounds;
  }

  setMesh(mesh: SAM3DMesh): void {
    if (!this.gl || !this.program || !this.positionBuffer || !this.indexBuffer) return;

    const gl = this.gl;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);

    const { data: indexData, type: indexType } = this.prepareIndexBuffer(mesh.faces);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexData, gl.STATIC_DRAW);
    this.indexCount = indexData.length;
    this.indexType = indexType;

    const bounds = mesh.bounds || this.computeBounds(mesh.vertices);
    this.bounds = bounds;

    this.modelMatrix = mat4FromTranslation([
      -bounds.center[0],
      -bounds.center[1],
      -bounds.center[2],
    ]);

    this.distance = Math.max(bounds.radius * 2.5, 2.5);
    this.render();
  }

  clearMesh(): void {
    this.indexCount = 0;
    this.indexType = null;
    this.bounds = null;
    this.render();
  }

  setCamera(yaw: number, pitch: number, roll: number, distance: number): void {
    this.yaw = yaw;
    this.pitch = pitch;
    this.roll = roll;
    this.distance = distance;
    this.render();
  }

  resize(width: number, height: number): void {
    if (!this.gl) return;
    this.gl.viewport(0, 0, width, height);
  }

  getSuggestedDistance(): number {
    return this.distance;
  }

  private prepareIndexBuffer(
    faces: Uint32Array
  ): { data: Uint16Array | Uint32Array; type: number } {
    if (!this.gl) {
      return { data: faces, type: WebGLRenderingContext.UNSIGNED_INT };
    }

    const gl = this.gl;
    const supportsUint32 = gl.getExtension('OES_element_index_uint');

    let maxIndex = 0;
    for (let i = 0; i < faces.length; i++) {
      maxIndex = Math.max(maxIndex, faces[i]);
    }

    if (maxIndex > 65535 && !supportsUint32) {
      const truncated = new Uint16Array(faces.length);
      for (let i = 0; i < faces.length; i++) {
        truncated[i] = faces[i] % 65535;
      }
      return { data: truncated, type: gl.UNSIGNED_SHORT };
    }

    if (supportsUint32) {
      return { data: faces, type: gl.UNSIGNED_INT };
    }

    return { data: new Uint16Array(faces), type: gl.UNSIGNED_SHORT };
  }

  private render(): void {
    if (!this.gl || !this.program) return;

    const gl = this.gl;

    gl.clearColor(0.97, 0.96, 0.94, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (!this.indexCount) return;

    gl.useProgram(this.program);

    const positionLoc = gl.getAttribLocation(this.program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);

    const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
    const projection = mat4Perspective(45 * (Math.PI / 180), aspect, 0.1, 1000);

    const target: Vector3 = [0, 0, 0];
    const yawRad = (this.yaw * Math.PI) / 180;
    const pitchRad = (this.pitch * Math.PI) / 180;
    const eye: Vector3 = [
      this.distance * Math.sin(yawRad) * Math.cos(pitchRad),
      this.distance * Math.sin(pitchRad),
      this.distance * Math.cos(yawRad) * Math.cos(pitchRad),
    ];

    let up: Vector3 = [0, 1, 0];
    const forward = vec3Normalize(vec3Sub(target, eye));

    if (this.roll !== 0) {
      const rollQuat = quatFromAxisAngle(forward, this.roll);
      up = quatRotateVector(rollQuat, up);
    }

    const view = mat4LookAt(eye, target, up);
    const model = this.modelMatrix;

    const uModel = gl.getUniformLocation(this.program, 'uModel');
    const uView = gl.getUniformLocation(this.program, 'uView');
    const uProjection = gl.getUniformLocation(this.program, 'uProjection');
    const uColor = gl.getUniformLocation(this.program, 'uColor');

    gl.uniformMatrix4fv(uModel, false, this.toColumnMajor(model));
    gl.uniformMatrix4fv(uView, false, this.toColumnMajor(view));
    gl.uniformMatrix4fv(uProjection, false, this.toColumnMajor(projection));
    gl.uniform3f(uColor, 0.12, 0.45, 0.42);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.drawElements(
      gl.TRIANGLES,
      this.indexCount,
      this.indexType ?? gl.UNSIGNED_SHORT,
      0
    );
  }

  private compileShader(type: number, source: string): WebGLShader {
    if (!this.gl) {
      throw new Error('WebGL not initialized');
    }

    const shader = this.gl.createShader(type);
    if (!shader) {
      throw new Error('Failed to create shader');
    }

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      const log = this.gl.getShaderInfoLog(shader) || 'Unknown shader error';
      throw new Error(log);
    }

    return shader;
  }

  private computeBounds(vertices: Float32Array): MeshBounds {
    if (vertices.length < 3) {
      return { center: [0, 0, 0], radius: 1 };
    }

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const y = vertices[i + 1];
      const z = vertices[i + 2];

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }

    const center: Vector3 = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];

    let radius = 0;
    for (let i = 0; i < vertices.length; i += 3) {
      const dx = vertices[i] - center[0];
      const dy = vertices[i + 1] - center[1];
      const dz = vertices[i + 2] - center[2];
      radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy + dz * dz));
    }

    return { center, radius: radius || 1 };
  }

  private toColumnMajor(matrix: Matrix4x4): Float32Array {
    return new Float32Array([
      matrix[0], matrix[4], matrix[8], matrix[12],
      matrix[1], matrix[5], matrix[9], matrix[13],
      matrix[2], matrix[6], matrix[10], matrix[14],
      matrix[3], matrix[7], matrix[11], matrix[15],
    ]);
  }
}
