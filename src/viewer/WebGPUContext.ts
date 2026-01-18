/**
 * WebGPU Context Manager
 * Handles WebGPU device initialization and resource management
 */

export interface WebGPUContextConfig {
  canvas: HTMLCanvasElement;
  powerPreference?: GPUPowerPreference;
  requiredFeatures?: GPUFeatureName[];
}

export class WebGPUContext {
  private _device: GPUDevice | null = null;
  private _context: GPUCanvasContext | null = null;
  private _format: GPUTextureFormat = 'bgra8unorm';
  private _canvas: HTMLCanvasElement;

  constructor(private config: WebGPUContextConfig) {
    this._canvas = config.canvas;
  }

  get device(): GPUDevice {
    if (!this._device) {
      throw new Error('WebGPU device not initialized. Call initialize() first.');
    }
    return this._device;
  }

  get context(): GPUCanvasContext {
    if (!this._context) {
      throw new Error('WebGPU context not initialized. Call initialize() first.');
    }
    return this._context;
  }

  get format(): GPUTextureFormat {
    return this._format;
  }

  get canvas(): HTMLCanvasElement {
    return this._canvas;
  }

  static async isSupported(): Promise<boolean> {
    if (!navigator.gpu) {
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: this.config.powerPreference ?? 'high-performance',
    });

    if (!adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    this._device = await adapter.requestDevice({
      requiredFeatures: this.config.requiredFeatures ?? [],
    });

    this._device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
      if (info.reason !== 'destroyed') {
        this.initialize();
      }
    });

    this._context = this._canvas.getContext('webgpu');
    if (!this._context) {
      throw new Error('Failed to get WebGPU canvas context');
    }

    this._format = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({
      device: this._device,
      format: this._format,
      alphaMode: 'premultiplied',
    });
  }

  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    return this.device.createBuffer(descriptor);
  }

  createTexture(descriptor: GPUTextureDescriptor): GPUTexture {
    return this.device.createTexture(descriptor);
  }

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    return this.device.createBindGroupLayout(descriptor);
  }

  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    return this.device.createPipelineLayout(descriptor);
  }

  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline {
    return this.device.createRenderPipeline(descriptor);
  }

  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline {
    return this.device.createComputePipeline(descriptor);
  }

  getCurrentTexture(): GPUTexture {
    return this.context.getCurrentTexture();
  }

  resize(width: number, height: number): void {
    this._canvas.width = width;
    this._canvas.height = height;
  }

  destroy(): void {
    this._device?.destroy();
    this._device = null;
    this._context = null;
  }
}
