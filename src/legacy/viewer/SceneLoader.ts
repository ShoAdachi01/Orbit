/**
 * OrbitScene Pack Loader
 * Loads and parses OrbitScene pack assets
 */

import {
  OrbitSceneManifest,
  SceneCamera,
  QualityReport,
  CameraPose,
  CameraIntrinsics,
} from '../schemas/types';

export interface SplatAsset {
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  count: number;
}

export interface OrbitScene {
  manifest: OrbitSceneManifest;
  camera: SceneCamera;
  quality: QualityReport;
  backgroundSplat: SplatAsset | null;
  subjectSplat: SplatAsset | null;
}

export class SceneLoader {
  private baseUrl: string;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }

  /**
   * Load OrbitScene pack from URL or file path
   *
   * Supports two URL formats:
   * 1. Directory path: "/scenes/my-scene/" - appends filename directly
   * 2. URL template: "/api/get_scene_file?job_id=xxx&filename=" - appends filename to query param
   */
  async load(packPath: string): Promise<OrbitScene> {
    // Determine how to construct asset URLs
    const getAssetUrl = (filename: string): string => {
      if (packPath.includes('?') && packPath.endsWith('=')) {
        // URL template format: /api/get_scene_file?job_id=xxx&filename=
        return `${packPath}${filename}`;
      } else {
        // Directory format: /scenes/my-scene/
        const basePath = packPath.endsWith('/') ? packPath : packPath + '/';
        return `${basePath}${filename}`;
      }
    };

    // Load manifest
    const manifestResponse = await fetch(getAssetUrl('manifest.json'));
    if (!manifestResponse.ok) {
      throw new Error(`Failed to load manifest: ${manifestResponse.statusText}`);
    }
    const manifest: OrbitSceneManifest = await manifestResponse.json();

    // Load camera
    const cameraResponse = await fetch(getAssetUrl(manifest.assets.camera));
    if (!cameraResponse.ok) {
      throw new Error(`Failed to load camera: ${cameraResponse.statusText}`);
    }
    const camera: SceneCamera = await cameraResponse.json();

    // Load quality report
    const qualityResponse = await fetch(getAssetUrl(manifest.assets.quality));
    if (!qualityResponse.ok) {
      throw new Error(`Failed to load quality: ${qualityResponse.statusText}`);
    }
    const quality: QualityReport = await qualityResponse.json();

    // Load splats
    const backgroundSplat = await this.loadSplat(getAssetUrl(manifest.assets.backgroundSplat));
    const subjectSplat = await this.loadSplat(getAssetUrl(manifest.assets.subjectSplat));

    return {
      manifest,
      camera,
      quality,
      backgroundSplat,
      subjectSplat,
    };
  }

  /**
   * Load splat file (.splat or .ply format)
   */
  private async loadSplat(url: string): Promise<SplatAsset | null> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        console.warn(`Failed to load splat: ${url}`);
        return null;
      }

      const buffer = await response.arrayBuffer();

      // Check file signature to determine format
      const header = new Uint8Array(buffer, 0, 4);
      const signature = String.fromCharCode(...header);

      if (signature === 'ply ') {
        return this.parsePLY(buffer);
      } else {
        return this.parseSplat(buffer);
      }
    } catch (error) {
      console.error(`Error loading splat from ${url}:`, error);
      return null;
    }
  }

  /**
   * Parse binary .splat format
   */
  private parseSplat(buffer: ArrayBuffer): SplatAsset {
    const view = new DataView(buffer);
    let offset = 0;

    // Read header
    const magic = view.getUint32(offset, true);
    offset += 4;

    if (magic !== 0x53504C54) { // "SPLT"
      throw new Error('Invalid splat file magic number');
    }

    const version = view.getUint32(offset, true);
    offset += 4;

    const count = view.getUint32(offset, true);
    offset += 4;

    // Allocate arrays
    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    const rotations = new Float32Array(count * 4);
    const colors = new Float32Array(count * 4);
    const opacities = new Float32Array(count);

    // Read splat data
    for (let i = 0; i < count; i++) {
      // Position (3 floats)
      positions[i * 3 + 0] = view.getFloat32(offset, true); offset += 4;
      positions[i * 3 + 1] = view.getFloat32(offset, true); offset += 4;
      positions[i * 3 + 2] = view.getFloat32(offset, true); offset += 4;

      // Scale (3 floats)
      scales[i * 3 + 0] = view.getFloat32(offset, true); offset += 4;
      scales[i * 3 + 1] = view.getFloat32(offset, true); offset += 4;
      scales[i * 3 + 2] = view.getFloat32(offset, true); offset += 4;

      // Rotation quaternion (4 floats)
      rotations[i * 4 + 0] = view.getFloat32(offset, true); offset += 4;
      rotations[i * 4 + 1] = view.getFloat32(offset, true); offset += 4;
      rotations[i * 4 + 2] = view.getFloat32(offset, true); offset += 4;
      rotations[i * 4 + 3] = view.getFloat32(offset, true); offset += 4;

      // Color RGBA (4 floats, normalized 0-1)
      colors[i * 4 + 0] = view.getFloat32(offset, true); offset += 4;
      colors[i * 4 + 1] = view.getFloat32(offset, true); offset += 4;
      colors[i * 4 + 2] = view.getFloat32(offset, true); offset += 4;
      colors[i * 4 + 3] = view.getFloat32(offset, true); offset += 4;

      // Opacity (1 float)
      opacities[i] = view.getFloat32(offset, true); offset += 4;
    }

    return { positions, scales, rotations, colors, opacities, count };
  }

  /**
   * Parse PLY format (common for Gaussian splats)
   */
  private parsePLY(buffer: ArrayBuffer): SplatAsset {
    const decoder = new TextDecoder();
    const text = decoder.decode(buffer);

    // Find header end
    const headerEnd = text.indexOf('end_header');
    if (headerEnd === -1) {
      throw new Error('Invalid PLY file: no end_header');
    }

    const header = text.substring(0, headerEnd);
    const lines = header.split('\n');

    // Parse header
    let vertexCount = 0;
    const properties: string[] = [];

    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts[0] === 'element' && parts[1] === 'vertex') {
        vertexCount = parseInt(parts[2], 10);
      } else if (parts[0] === 'property') {
        properties.push(parts[parts.length - 1]);
      }
    }

    // Find property indices
    const propIndex = (name: string): number => properties.indexOf(name);

    const xIdx = propIndex('x');
    const yIdx = propIndex('y');
    const zIdx = propIndex('z');
    const scaleXIdx = propIndex('scale_0') !== -1 ? propIndex('scale_0') : propIndex('scale_x');
    const scaleYIdx = propIndex('scale_1') !== -1 ? propIndex('scale_1') : propIndex('scale_y');
    const scaleZIdx = propIndex('scale_2') !== -1 ? propIndex('scale_2') : propIndex('scale_z');
    const rotWIdx = propIndex('rot_0') !== -1 ? propIndex('rot_0') : propIndex('rot_w');
    const rotXIdx = propIndex('rot_1') !== -1 ? propIndex('rot_1') : propIndex('rot_x');
    const rotYIdx = propIndex('rot_2') !== -1 ? propIndex('rot_2') : propIndex('rot_y');
    const rotZIdx = propIndex('rot_3') !== -1 ? propIndex('rot_3') : propIndex('rot_z');
    const rIdx = propIndex('red') !== -1 ? propIndex('red') : propIndex('f_dc_0');
    const gIdx = propIndex('green') !== -1 ? propIndex('green') : propIndex('f_dc_1');
    const bIdx = propIndex('blue') !== -1 ? propIndex('blue') : propIndex('f_dc_2');
    const opacityIdx = propIndex('opacity');

    // Allocate arrays
    const positions = new Float32Array(vertexCount * 3);
    const scales = new Float32Array(vertexCount * 3);
    const rotations = new Float32Array(vertexCount * 4);
    const colors = new Float32Array(vertexCount * 4);
    const opacities = new Float32Array(vertexCount);

    // Parse binary data after header
    const binaryStart = headerEnd + 'end_header'.length + 1;
    const binaryData = new Uint8Array(buffer, binaryStart);
    const view = new DataView(binaryData.buffer, binaryData.byteOffset);

    const floatsPerVertex = properties.length;
    const bytesPerVertex = floatsPerVertex * 4;

    for (let i = 0; i < vertexCount; i++) {
      const base = i * bytesPerVertex;

      // Position
      positions[i * 3 + 0] = view.getFloat32(base + xIdx * 4, true);
      positions[i * 3 + 1] = view.getFloat32(base + yIdx * 4, true);
      positions[i * 3 + 2] = view.getFloat32(base + zIdx * 4, true);

      // Scale (convert from log scale if needed)
      if (scaleXIdx !== -1) {
        scales[i * 3 + 0] = Math.exp(view.getFloat32(base + scaleXIdx * 4, true));
        scales[i * 3 + 1] = Math.exp(view.getFloat32(base + scaleYIdx * 4, true));
        scales[i * 3 + 2] = Math.exp(view.getFloat32(base + scaleZIdx * 4, true));
      } else {
        scales[i * 3 + 0] = 0.01;
        scales[i * 3 + 1] = 0.01;
        scales[i * 3 + 2] = 0.01;
      }

      // Rotation (normalize quaternion)
      if (rotWIdx !== -1) {
        const qw = view.getFloat32(base + rotWIdx * 4, true);
        const qx = view.getFloat32(base + rotXIdx * 4, true);
        const qy = view.getFloat32(base + rotYIdx * 4, true);
        const qz = view.getFloat32(base + rotZIdx * 4, true);
        const len = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        rotations[i * 4 + 0] = qx / len;
        rotations[i * 4 + 1] = qy / len;
        rotations[i * 4 + 2] = qz / len;
        rotations[i * 4 + 3] = qw / len;
      } else {
        rotations[i * 4 + 0] = 0;
        rotations[i * 4 + 1] = 0;
        rotations[i * 4 + 2] = 0;
        rotations[i * 4 + 3] = 1;
      }

      // Color (SH to RGB conversion for f_dc coefficients)
      if (rIdx !== -1) {
        const r = view.getFloat32(base + rIdx * 4, true);
        const g = view.getFloat32(base + gIdx * 4, true);
        const b = view.getFloat32(base + bIdx * 4, true);

        // Check if using SH coefficients (f_dc_*)
        if (properties[rIdx].startsWith('f_dc')) {
          // SH DC coefficient to RGB: (val * SH_C0 + 0.5)
          const SH_C0 = 0.28209479177387814;
          colors[i * 4 + 0] = Math.max(0, Math.min(1, r * SH_C0 + 0.5));
          colors[i * 4 + 1] = Math.max(0, Math.min(1, g * SH_C0 + 0.5));
          colors[i * 4 + 2] = Math.max(0, Math.min(1, b * SH_C0 + 0.5));
        } else {
          // Direct RGB (0-255 to 0-1)
          colors[i * 4 + 0] = r / 255;
          colors[i * 4 + 1] = g / 255;
          colors[i * 4 + 2] = b / 255;
        }
        colors[i * 4 + 3] = 1.0;
      } else {
        colors[i * 4 + 0] = 0.5;
        colors[i * 4 + 1] = 0.5;
        colors[i * 4 + 2] = 0.5;
        colors[i * 4 + 3] = 1.0;
      }

      // Opacity (sigmoid from raw value)
      if (opacityIdx !== -1) {
        const rawOpacity = view.getFloat32(base + opacityIdx * 4, true);
        opacities[i] = 1 / (1 + Math.exp(-rawOpacity));
      } else {
        opacities[i] = 1.0;
      }
    }

    return { positions, scales, rotations, colors, opacities, count: vertexCount };
  }

  /**
   * Create a dummy scene for testing
   */
  static createDummyScene(): OrbitScene {
    const now = new Date().toISOString();

    const manifest: OrbitSceneManifest = {
      version: '1.0',
      id: 'dummy-scene',
      createdAt: now,
      source: {
        filename: 'dummy.mp4',
        duration: 10,
        fps: 30,
        width: 1920,
        height: 1080,
      },
      assets: {
        backgroundSplat: 'bg.splat',
        subjectSplat: 'subject_4d.splat',
        camera: 'camera.json',
        quality: 'quality.json',
      },
      quality: {
        mask: {
          score: 0.9,
          subjectAreaRatios: [],
          edgeJitter: 0.1,
          leakScore: 0.05,
          highCoverageFrameCount: 0,
          totalFrames: 300,
        },
        pose: {
          score: 0.85,
          inlierRatio: 0.7,
          medianReprojectionError: 1.2,
          quadrantCoverage: [true, true, true, true],
          goodCoverageFramePercent: 0.9,
          jitterScore: 0.15,
        },
        track: {
          numTracksKept: 150,
          numTracksDiscarded: 30,
          medianLifespan: 45,
          medianConfidence: 0.75,
        },
        depth: {
          frameConfidences: [],
          temporalConsistency: 0.88,
          edgeStability: 0.82,
        },
        mode: 'full-orbit',
        enforcedBounds: {
          maxYaw: 20,
          maxPitch: 10,
          maxRoll: 3,
          maxTranslation: 0.1,
          maxTranslationDepthPercent: 2,
          clampToParallax: true,
        },
        modeReasons: ['All quality gates passed'],
      },
    };

    const camera: SceneCamera = {
      intrinsics: {
        fx: 1000,
        fy: 1000,
        cx: 960,
        cy: 540,
        width: 1920,
        height: 1080,
      },
      referencePose: {
        transform: [
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ],
        timestamp: 0,
        frameIndex: 0,
      },
      poses: [],
    };

    // Create dummy splat (single Gaussian for testing)
    const dummySplat: SplatAsset = {
      positions: new Float32Array([0, 0, 0]),
      scales: new Float32Array([0.1, 0.1, 0.1]),
      rotations: new Float32Array([0, 0, 0, 1]),
      colors: new Float32Array([0.8, 0.3, 0.3, 1.0]),
      opacities: new Float32Array([1.0]),
      count: 1,
    };

    return {
      manifest,
      camera,
      quality: manifest.quality,
      backgroundSplat: dummySplat,
      subjectSplat: null,
    };
  }
}
