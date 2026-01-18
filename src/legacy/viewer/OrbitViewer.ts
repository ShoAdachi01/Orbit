/**
 * OrbitViewer
 * Main viewer component that combines WebGPU context, camera, and splat rendering
 */

import { WebGPUContext } from './WebGPUContext';
import { OrbitCamera } from './Camera';
import { SplatRenderer } from './SplatRenderer';
import { SceneLoader, OrbitScene, SplatAsset } from './SceneLoader';
import { OrbitMode, OrbitBounds, DEFAULT_ORBIT_BOUNDS, Vector3 } from '../schemas/types';

export interface OrbitViewerConfig {
  canvas: HTMLCanvasElement;
  onProgress?: (message: string) => void;
  onError?: (error: Error) => void;
  onModeChange?: (mode: OrbitMode) => void;
}

export interface ViewerState {
  isInitialized: boolean;
  isLoading: boolean;
  scene: OrbitScene | null;
  currentMode: OrbitMode;
  bounds: OrbitBounds;
}

export class OrbitViewer {
  private config: OrbitViewerConfig;
  private ctx: WebGPUContext | null = null;
  private camera: OrbitCamera;
  private renderer: SplatRenderer | null = null;
  private sceneLoader: SceneLoader;

  private scene: OrbitScene | null = null;
  private animationFrameId: number | null = null;

  private isDragging = false;
  private lastMouseX = 0;
  private lastMouseY = 0;

  private state: ViewerState = {
    isInitialized: false,
    isLoading: false,
    scene: null,
    currentMode: 'full-orbit',
    bounds: { ...DEFAULT_ORBIT_BOUNDS },
  };

  constructor(config: OrbitViewerConfig) {
    this.config = config;
    this.camera = new OrbitCamera();
    this.sceneLoader = new SceneLoader();
    this.setupEventListeners();
  }

  get viewerState(): ViewerState {
    return { ...this.state };
  }

  async initialize(): Promise<void> {
    try {
      this.config.onProgress?.('Checking WebGPU support...');

      if (!(await WebGPUContext.isSupported())) {
        throw new Error('WebGPU is not supported in this browser');
      }

      this.ctx = new WebGPUContext({ canvas: this.config.canvas });
      await this.ctx.initialize();

      this.config.onProgress?.('Initializing renderer...');
      this.renderer = new SplatRenderer(this.ctx);
      await this.renderer.initialize();

      this.state.isInitialized = true;
      this.config.onProgress?.('Viewer ready');

      // Start render loop
      this.startRenderLoop();
    } catch (error) {
      this.config.onError?.(error as Error);
      throw error;
    }
  }

  async loadScene(packPath: string): Promise<void> {
    if (!this.state.isInitialized) {
      throw new Error('Viewer not initialized. Call initialize() first.');
    }

    try {
      this.state.isLoading = true;
      this.config.onProgress?.('Loading scene...');

      this.scene = await this.sceneLoader.load(packPath);
      this.state.scene = this.scene;

      // Apply scene quality settings
      const { quality } = this.scene;
      this.state.currentMode = quality.mode;
      this.state.bounds = quality.enforcedBounds;
      this.camera.setBounds(quality.enforcedBounds);

      // Apply camera intrinsics
      this.camera.applyIntrinsics(this.scene.camera.intrinsics);

      // Load splats into renderer
      if (this.scene.backgroundSplat && this.renderer) {
        this.config.onProgress?.('Uploading background splats...');
        this.renderer.uploadSplats(this.scene.backgroundSplat);
        const bounds = this.computeSceneBounds(this.scene.backgroundSplat);
        if (bounds) {
          // Auto-frame the scene so splats are in view.
          this.camera.setTarget(bounds.center);
          this.camera.setDistance(Math.max(bounds.radius * 2.5, 0.1));
          this.camera.setMedianSceneDepth(Math.max(bounds.radius, 0.1));
          this.config.onProgress?.(
            `Scene bounds: center=[${bounds.center.map((v) => v.toFixed(2)).join(', ')}], radius=${bounds.radius.toFixed(2)}`
          );
        }
      }

      // Notify mode change
      this.config.onModeChange?.(quality.mode);

      this.state.isLoading = false;
      this.config.onProgress?.('Scene loaded');
    } catch (error) {
      this.state.isLoading = false;
      this.config.onError?.(error as Error);
      throw error;
    }
  }

  loadDummyScene(): void {
    if (!this.state.isInitialized || !this.renderer) {
      console.warn('Viewer not initialized');
      return;
    }

    this.scene = SceneLoader.createDummyScene();
    this.state.scene = this.scene;

    if (this.scene.backgroundSplat) {
      this.renderer.uploadSplats(this.scene.backgroundSplat);
    }
  }

  setMode(mode: OrbitMode): void {
    this.state.currentMode = mode;

    // Adjust bounds based on mode
    switch (mode) {
      case 'micro-parallax':
        this.camera.setBounds({
          maxYaw: 8,
          maxPitch: 4,
          maxRoll: 1,
          maxTranslation: 0.03,
        });
        break;
      case '2.5d-subject':
        this.camera.setBounds({
          maxYaw: 15,
          maxPitch: 8,
          maxRoll: 2,
          maxTranslation: 0.05,
        });
        break;
      case 'render-only':
        // No interactive controls
        this.camera.setBounds({
          maxYaw: 0,
          maxPitch: 0,
          maxRoll: 0,
          maxTranslation: 0,
        });
        break;
      default:
        this.camera.setBounds(DEFAULT_ORBIT_BOUNDS);
    }

    this.config.onModeChange?.(mode);
  }

  resetCamera(): void {
    this.camera.reset();
  }

  resize(width: number, height: number): void {
    this.renderer?.resize(width, height);
    this.camera.setAspect(width / height);
  }

  private setupEventListeners(): void {
    const canvas = this.config.canvas;

    canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
    canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
    canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));
    canvas.addEventListener('wheel', this.onWheel.bind(this));

    // Touch events for mobile
    canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
    canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
    canvas.addEventListener('touchend', this.onTouchEnd.bind(this));

    // Resize observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        this.resize(width, height);
      }
    });
    resizeObserver.observe(canvas);
  }

  private onMouseDown(event: MouseEvent): void {
    if (this.state.currentMode === 'render-only') return;

    this.isDragging = true;
    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.isDragging || this.state.currentMode === 'render-only') return;

    const deltaX = event.clientX - this.lastMouseX;
    const deltaY = event.clientY - this.lastMouseY;

    if (event.shiftKey) {
      // Pan
      this.camera.pan(deltaX * 0.001, -deltaY * 0.001);
    } else {
      // Orbit
      this.camera.orbit(deltaX * 0.2, deltaY * 0.2);
    }

    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;
  }

  private onMouseUp(): void {
    this.isDragging = false;
  }

  private onWheel(event: WheelEvent): void {
    if (this.state.currentMode === 'render-only') return;

    event.preventDefault();
    const delta = event.deltaY * 0.001;
    this.camera.setDistance(this.camera.position[2] * (1 + delta));
  }

  private onTouchStart(event: TouchEvent): void {
    if (event.touches.length === 1 && this.state.currentMode !== 'render-only') {
      this.isDragging = true;
      this.lastMouseX = event.touches[0].clientX;
      this.lastMouseY = event.touches[0].clientY;
    }
  }

  private onTouchMove(event: TouchEvent): void {
    if (!this.isDragging || event.touches.length !== 1) return;
    if (this.state.currentMode === 'render-only') return;

    event.preventDefault();

    const deltaX = event.touches[0].clientX - this.lastMouseX;
    const deltaY = event.touches[0].clientY - this.lastMouseY;

    this.camera.orbit(deltaX * 0.2, deltaY * 0.2);

    this.lastMouseX = event.touches[0].clientX;
    this.lastMouseY = event.touches[0].clientY;
  }

  private onTouchEnd(): void {
    this.isDragging = false;
  }

  private startRenderLoop(): void {
    const render = (): void => {
      if (this.renderer) {
        this.renderer.render(this.camera);
      }
      this.animationFrameId = requestAnimationFrame(render);
    };
    render();
  }

  private computeSceneBounds(splat: SplatAsset): { center: Vector3; radius: number } | null {
    const { positions, count } = splat;
    if (!positions || count <= 0) {
      return null;
    }

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    let validCount = 0;

    for (let i = 0; i < count; i++) {
      const index = i * 3;
      const x = positions[index];
      const y = positions[index + 1];
      const z = positions[index + 2];

      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        continue;
      }

      validCount++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }

    if (validCount === 0) {
      return null;
    }

    const center: Vector3 = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];
    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { center, radius };
  }

  destroy(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.renderer?.destroy();
    this.ctx?.destroy();
  }
}

export { WebGPUContext } from './WebGPUContext';
export { OrbitCamera } from './Camera';
export { SplatRenderer } from './SplatRenderer';
export { SceneLoader } from './SceneLoader';
export type { OrbitScene, SplatAsset } from './SceneLoader';
