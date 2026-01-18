/**
 * Gaussian Splat Renderer
 * WebGPU-based renderer for 3D Gaussian Splatting
 */

import { WebGPUContext } from './WebGPUContext';
import { OrbitCamera } from './Camera';
import { Matrix4x4 } from '../schemas/types';

// Splat vertex data layout
interface SplatData {
  positions: Float32Array;      // xyz
  scales: Float32Array;         // xyz
  rotations: Float32Array;      // quaternion xyzw
  colors: Float32Array;         // rgba (spherical harmonics or direct)
  opacities: Float32Array;      // single float per splat
  count: number;
}

// Shader code
const SPLAT_SHADER = /* wgsl */`
struct Uniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad0: f32,
  viewport: vec2<f32>,
  _pad1: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct SplatInstance {
  @location(0) position: vec3<f32>,
  @location(1) scale: vec3<f32>,
  @location(2) rotation: vec4<f32>,
  @location(3) color: vec4<f32>,
  @location(4) opacity: f32,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) opacity: f32,
}

// Quad vertices for billboarding
const QUAD_VERTS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(1.0, 1.0),
);

fn quaternionToMatrix(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x;
  let y = q.y;
  let z = q.z;
  let w = q.w;

  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)),
    vec3<f32>(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3<f32>(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)),
  );
}

@vertex
fn vs_main(
  splat: SplatInstance,
  @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
  var output: VertexOutput;

  let quadVert = QUAD_VERTS[vertexIndex];

  // Transform splat to view space
  let viewPos = uniforms.viewProj * vec4<f32>(splat.position, 1.0);

  // Compute splat size in screen space
  let distance = length(splat.position - uniforms.cameraPos);
  let screenScale = 2.0 / max(distance, 0.1);

  // Apply rotation and scale to quad
  let rotMat = quaternionToMatrix(splat.rotation);
  let scaledOffset = rotMat * vec3<f32>(quadVert * splat.scale.xy, 0.0);

  // Billboard in screen space
  let screenOffset = vec2<f32>(
    scaledOffset.x * uniforms.viewport.y / uniforms.viewport.x,
    scaledOffset.y
  ) * screenScale;

  output.position = viewPos;
  output.position.x += screenOffset.x * output.position.w;
  output.position.y += screenOffset.y * output.position.w;

  output.color = splat.color;
  output.uv = quadVert * 0.5 + 0.5;
  output.opacity = splat.opacity;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // Gaussian falloff
  let dist = length(input.uv - vec2<f32>(0.5, 0.5)) * 2.0;
  let alpha = exp(-dist * dist * 2.0) * input.opacity;

  if (alpha < 0.01) {
    discard;
  }

  return vec4<f32>(input.color.rgb, alpha);
}
`;

export class SplatRenderer {
  private ctx: WebGPUContext;
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private uniformBindGroup: GPUBindGroup | null = null;

  private splatBuffer: GPUBuffer | null = null;
  private splatCount = 0;

  private depthTexture: GPUTexture | null = null;
  private depthTextureView: GPUTextureView | null = null;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  async initialize(): Promise<void> {
    const device = this.ctx.device;

    // Create uniform buffer
    this.uniformBuffer = device.createBuffer({
      size: 96, // mat4x4 + vec3 + pad + vec2 + pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
      ],
    });

    // Create bind group
    this.uniformBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
      ],
    });

    // Create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // Create shader module
    const shaderModule = device.createShaderModule({
      code: SPLAT_SHADER,
    });

    // Create render pipeline
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 60, // 3+3+4+4+1 floats = 15 * 4 = 60 bytes
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },   // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' },  // scale
              { shaderLocation: 2, offset: 24, format: 'float32x4' },  // rotation
              { shaderLocation: 3, offset: 40, format: 'float32x4' },  // color
              { shaderLocation: 4, offset: 56, format: 'float32' },    // opacity
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: this.ctx.format,
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.createDepthTexture();
  }

  private createDepthTexture(): void {
    this.depthTexture?.destroy();

    this.depthTexture = this.ctx.device.createTexture({
      size: {
        width: this.ctx.canvas.width,
        height: this.ctx.canvas.height,
      },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.depthTextureView = this.depthTexture.createView();
  }

  resize(width: number, height: number): void {
    this.ctx.resize(width, height);
    this.createDepthTexture();
  }

  uploadSplats(data: SplatData): void {
    // Interleave splat data
    const interleaved = new Float32Array(data.count * 15);
    for (let i = 0; i < data.count; i++) {
      const offset = i * 15;
      // Position
      interleaved[offset + 0] = data.positions[i * 3 + 0];
      interleaved[offset + 1] = data.positions[i * 3 + 1];
      interleaved[offset + 2] = data.positions[i * 3 + 2];
      // Scale
      interleaved[offset + 3] = data.scales[i * 3 + 0];
      interleaved[offset + 4] = data.scales[i * 3 + 1];
      interleaved[offset + 5] = data.scales[i * 3 + 2];
      // Rotation
      interleaved[offset + 6] = data.rotations[i * 4 + 0];
      interleaved[offset + 7] = data.rotations[i * 4 + 1];
      interleaved[offset + 8] = data.rotations[i * 4 + 2];
      interleaved[offset + 9] = data.rotations[i * 4 + 3];
      // Color
      interleaved[offset + 10] = data.colors[i * 4 + 0];
      interleaved[offset + 11] = data.colors[i * 4 + 1];
      interleaved[offset + 12] = data.colors[i * 4 + 2];
      interleaved[offset + 13] = data.colors[i * 4 + 3];
      // Opacity
      interleaved[offset + 14] = data.opacities[i];
    }

    this.splatBuffer?.destroy();
    this.splatBuffer = this.ctx.device.createBuffer({
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.ctx.device.queue.writeBuffer(this.splatBuffer, 0, interleaved);
    this.splatCount = data.count;
  }

  render(camera: OrbitCamera): void {
    if (!this.pipeline || !this.uniformBuffer || !this.uniformBindGroup || !this.depthTextureView) {
      return;
    }

    const device = this.ctx.device;

    // Update uniforms
    const viewProj = camera.getViewProjectionMatrix();
    const cameraPos = camera.position;
    const viewport = [this.ctx.canvas.width, this.ctx.canvas.height];

    const uniformData = new Float32Array(24);
    uniformData.set(viewProj, 0);
    uniformData.set(cameraPos, 16);
    uniformData.set(viewport, 20);
    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Get current texture
    const colorTexture = this.ctx.getCurrentTexture();
    const colorView = colorTexture.createView();

    // Create command encoder
    const commandEncoder = device.createCommandEncoder();

    // Begin render pass
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorView,
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.uniformBindGroup);

    if (this.splatBuffer && this.splatCount > 0) {
      renderPass.setVertexBuffer(0, this.splatBuffer);
      renderPass.draw(6, this.splatCount); // 6 vertices per quad, instanced
    }

    renderPass.end();

    // Submit
    device.queue.submit([commandEncoder.finish()]);
  }

  destroy(): void {
    this.uniformBuffer?.destroy();
    this.splatBuffer?.destroy();
    this.depthTexture?.destroy();
  }
}
