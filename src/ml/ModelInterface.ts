/**
 * ML Model Integration Interface
 * Defines contracts for integrating with external ML models
 */

// ============================================================================
// Common Types
// ============================================================================

export interface ModelConfig {
  /** Model endpoint URL (for cloud) */
  endpoint?: string;
  /** Local model path (for WASM/ONNX) */
  modelPath?: string;
  /** Device preference */
  device?: 'cpu' | 'gpu' | 'auto';
  /** Batch size for inference */
  batchSize?: number;
}

export interface InferenceResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  inferenceTime?: number;
}

// ============================================================================
// SAM2 (Segment Anything Model 2)
// ============================================================================

export interface SAM2Config extends ModelConfig {
  /** Minimum mask area threshold */
  minMaskArea?: number;
  /** Points per side for auto-segmentation */
  pointsPerSide?: number;
  /** Prediction IOU threshold */
  predIouThresh?: number;
  /** Stability score threshold */
  stabilityScoreThresh?: number;
}

export interface SAM2Input {
  /** Image as Uint8Array (RGB) */
  image: Uint8Array;
  width: number;
  height: number;
  /** Optional prompt points */
  points?: Array<{ x: number; y: number; label: 0 | 1 }>;
  /** Optional bounding box prompt */
  box?: { x: number; y: number; width: number; height: number };
}

export interface SAM2Output {
  /** Segmentation mask (0 or 255) */
  mask: Uint8Array;
  /** Confidence score */
  score: number;
  /** IOU prediction */
  iouPrediction: number;
}

export interface SAM2Model {
  initialize(config: SAM2Config): Promise<void>;
  segment(input: SAM2Input): Promise<InferenceResult<SAM2Output>>;
  segmentVideo(frames: SAM2Input[]): Promise<InferenceResult<SAM2Output[]>>;
  destroy(): void;
}

// ============================================================================
// TAPIR (Tracking Any Point with per-frame Initialization and temporal Refinement)
// ============================================================================

export interface TAPIRConfig extends ModelConfig {
  /** Number of points to track */
  numPoints?: number;
  /** Query frame index */
  queryFrame?: number;
  /** Confidence threshold for track filtering */
  confidenceThreshold?: number;
}

export interface TAPIRInput {
  /** Video frames as array of RGB images */
  frames: Array<{
    data: Uint8Array;
    width: number;
    height: number;
  }>;
  /** Query points to track (from query frame) */
  queryPoints: Array<{ x: number; y: number }>;
  /** Index of the query frame */
  queryFrameIndex: number;
}

export interface TAPIRTrack {
  /** Track ID */
  id: number;
  /** Points across all frames */
  points: Array<{
    frameIndex: number;
    x: number;
    y: number;
    visible: boolean;
    confidence: number;
  }>;
}

export interface TAPIROutput {
  tracks: TAPIRTrack[];
  /** Visibility map per track per frame */
  visibilityMap?: boolean[][];
}

export interface TAPIRModel {
  initialize(config: TAPIRConfig): Promise<void>;
  track(input: TAPIRInput): Promise<InferenceResult<TAPIROutput>>;
  destroy(): void;
}

// ============================================================================
// DepthCrafter (Video Depth Estimation)
// ============================================================================

export interface DepthCrafterConfig extends ModelConfig {
  /** Number of inference steps */
  numInferenceSteps?: number;
  /** Guidance scale */
  guidanceScale?: number;
  /** Window size for temporal processing */
  windowSize?: number;
  /** Overlap between windows */
  overlap?: number;
}

export interface DepthCrafterInput {
  /** Video frames as RGB images */
  frames: Array<{
    data: Uint8Array;
    width: number;
    height: number;
  }>;
}

export interface DepthCrafterOutput {
  /** Depth maps for each frame */
  depthMaps: Float32Array[];
  /** Confidence maps (optional) */
  confidenceMaps?: Float32Array[];
  /** Median depth per frame */
  medianDepths: number[];
}

export interface DepthCrafterModel {
  initialize(config: DepthCrafterConfig): Promise<void>;
  estimateDepth(input: DepthCrafterInput): Promise<InferenceResult<DepthCrafterOutput>>;
  destroy(): void;
}

// ============================================================================
// COLMAP (Structure from Motion)
// ============================================================================

export interface COLMAPConfig extends ModelConfig {
  /** Matcher type */
  matcherType?: 'exhaustive' | 'sequential' | 'vocab_tree';
  /** Use GPU for feature extraction */
  useGpu?: boolean;
  /** Minimum number of matches */
  minNumMatches?: number;
  /** SfM algorithm */
  sfmAlgorithm?: 'incremental' | 'global';
}

export interface COLMAPInput {
  /** Image frames */
  images: Array<{
    data: Uint8Array;
    width: number;
    height: number;
    id: number;
  }>;
  /** Optional masks to exclude regions */
  masks?: Array<{
    data: Uint8Array;
    width: number;
    height: number;
  }>;
  /** Known camera intrinsics (optional) */
  intrinsics?: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    model?: string;
  };
}

export interface COLMAPCamera {
  /** Camera model */
  model: string;
  /** Image dimensions */
  width: number;
  height: number;
  /** Intrinsic parameters */
  params: number[];
}

export interface COLMAPPose {
  /** Image ID */
  imageId: number;
  /** Rotation quaternion [qw, qx, qy, qz] */
  rotation: [number, number, number, number];
  /** Translation [tx, ty, tz] */
  translation: [number, number, number];
  /** Registered (successfully reconstructed) */
  registered: boolean;
}

export interface COLMAP3DPoint {
  /** Point ID */
  id: number;
  /** 3D position */
  position: [number, number, number];
  /** Color */
  color: [number, number, number];
  /** Reprojection error */
  error: number;
  /** Track length (number of images observing this point) */
  trackLength: number;
}

export interface COLMAPOutput {
  /** Camera parameters */
  cameras: COLMAPCamera[];
  /** Image poses */
  poses: COLMAPPose[];
  /** Sparse 3D points */
  points3D: COLMAP3DPoint[];
  /** Feature matches */
  numMatches: number;
  /** Reconstruction statistics */
  stats: {
    numRegisteredImages: number;
    numPoints: number;
    meanTrackLength: number;
    meanReprojectionError: number;
  };
}

export interface COLMAPModel {
  initialize(config: COLMAPConfig): Promise<void>;
  reconstruct(input: COLMAPInput): Promise<InferenceResult<COLMAPOutput>>;
  destroy(): void;
}

// ============================================================================
// Face Embedding (for Identity Lock)
// ============================================================================

export interface FaceEmbeddingConfig extends ModelConfig {
  /** Embedding dimension */
  embeddingDim?: number;
  /** Face detection confidence threshold */
  detectionThreshold?: number;
}

export interface FaceEmbeddingInput {
  /** Image as RGB */
  image: Uint8Array;
  width: number;
  height: number;
  /** Optional crop region */
  cropRegion?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface FaceEmbeddingOutput {
  /** Face detected */
  faceDetected: boolean;
  /** Face bounding box */
  faceBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  /** Face embedding vector */
  embedding?: Float32Array;
  /** Detection confidence */
  confidence: number;
}

export interface FaceEmbeddingModel {
  initialize(config: FaceEmbeddingConfig): Promise<void>;
  embed(input: FaceEmbeddingInput): Promise<InferenceResult<FaceEmbeddingOutput>>;
  computeDistance(embedding1: Float32Array, embedding2: Float32Array): number;
  destroy(): void;
}

// ============================================================================
// Video Diffusion (for Refinement)
// ============================================================================

export interface VideoDiffusionConfig extends ModelConfig {
  /** Number of denoising steps */
  numSteps?: number;
  /** Guidance scale */
  guidanceScale?: number;
  /** Temporal consistency strength */
  temporalWeight?: number;
}

export interface VideoDiffusionInput {
  /** Base render frames */
  frames: Array<{
    data: Uint8Array;
    width: number;
    height: number;
  }>;
  /** Condition images (anchors) */
  conditionFrames?: Array<{
    data: Uint8Array;
    frameIndex: number;
  }>;
  /** Text prompt (optional) */
  prompt?: string;
}

export interface VideoDiffusionOutput {
  /** Refined frames */
  frames: Array<{
    data: Uint8Array;
    width: number;
    height: number;
  }>;
}

export interface VideoDiffusionModel {
  initialize(config: VideoDiffusionConfig): Promise<void>;
  refine(input: VideoDiffusionInput): Promise<InferenceResult<VideoDiffusionOutput>>;
  destroy(): void;
}

// ============================================================================
// Model Factory
// ============================================================================

export interface ModelFactory {
  createSAM2(): SAM2Model;
  createTAPIR(): TAPIRModel;
  createDepthCrafter(): DepthCrafterModel;
  createCOLMAP(): COLMAPModel;
  createFaceEmbedding(): FaceEmbeddingModel;
  createVideoDiffusion(): VideoDiffusionModel;
}

/**
 * Get the default model factory
 * Returns stub implementations by default
 */
export function getModelFactory(): ModelFactory {
  return {
    createSAM2: () => new StubSAM2Model(),
    createTAPIR: () => new StubTAPIRModel(),
    createDepthCrafter: () => new StubDepthCrafterModel(),
    createCOLMAP: () => new StubCOLMAPModel(),
    createFaceEmbedding: () => new StubFaceEmbeddingModel(),
    createVideoDiffusion: () => new StubVideoDiffusionModel(),
  };
}

// ============================================================================
// Stub Implementations (for testing)
// ============================================================================

class StubSAM2Model implements SAM2Model {
  async initialize(_config: SAM2Config): Promise<void> {
    console.log('[StubSAM2] Initialized');
  }

  async segment(input: SAM2Input): Promise<InferenceResult<SAM2Output>> {
    // Generate a simple circular mask in center
    const mask = new Uint8Array(input.width * input.height);
    const cx = input.width / 2;
    const cy = input.height / 2;
    const radius = Math.min(input.width, input.height) * 0.3;

    for (let y = 0; y < input.height; y++) {
      for (let x = 0; x < input.width; x++) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy < radius * radius) {
          mask[y * input.width + x] = 255;
        }
      }
    }

    return {
      success: true,
      data: { mask, score: 0.95, iouPrediction: 0.92 },
      inferenceTime: 100,
    };
  }

  async segmentVideo(frames: SAM2Input[]): Promise<InferenceResult<SAM2Output[]>> {
    const results: SAM2Output[] = [];
    for (const frame of frames) {
      const result = await this.segment(frame);
      if (result.success && result.data) {
        results.push(result.data);
      }
    }
    return { success: true, data: results, inferenceTime: frames.length * 100 };
  }

  destroy(): void {
    console.log('[StubSAM2] Destroyed');
  }
}

class StubTAPIRModel implements TAPIRModel {
  async initialize(_config: TAPIRConfig): Promise<void> {
    console.log('[StubTAPIR] Initialized');
  }

  async track(input: TAPIRInput): Promise<InferenceResult<TAPIROutput>> {
    const tracks: TAPIRTrack[] = input.queryPoints.map((point, idx) => ({
      id: idx,
      points: input.frames.map((_, frameIdx) => ({
        frameIndex: frameIdx,
        x: point.x + (Math.random() - 0.5) * 5, // Small random motion
        y: point.y + (Math.random() - 0.5) * 5,
        visible: true,
        confidence: 0.8 + Math.random() * 0.15,
      })),
    }));

    return {
      success: true,
      data: { tracks },
      inferenceTime: input.frames.length * 50,
    };
  }

  destroy(): void {
    console.log('[StubTAPIR] Destroyed');
  }
}

class StubDepthCrafterModel implements DepthCrafterModel {
  async initialize(_config: DepthCrafterConfig): Promise<void> {
    console.log('[StubDepthCrafter] Initialized');
  }

  async estimateDepth(input: DepthCrafterInput): Promise<InferenceResult<DepthCrafterOutput>> {
    const depthMaps: Float32Array[] = [];
    const medianDepths: number[] = [];

    for (const frame of input.frames) {
      const depth = new Float32Array(frame.width * frame.height);
      const baseDepth = 3 + Math.random() * 2;

      for (let i = 0; i < depth.length; i++) {
        depth[i] = baseDepth + (Math.random() - 0.5) * 0.5;
      }

      depthMaps.push(depth);
      medianDepths.push(baseDepth);
    }

    return {
      success: true,
      data: { depthMaps, medianDepths },
      inferenceTime: input.frames.length * 200,
    };
  }

  destroy(): void {
    console.log('[StubDepthCrafter] Destroyed');
  }
}

class StubCOLMAPModel implements COLMAPModel {
  async initialize(_config: COLMAPConfig): Promise<void> {
    console.log('[StubCOLMAP] Initialized');
  }

  async reconstruct(input: COLMAPInput): Promise<InferenceResult<COLMAPOutput>> {
    const poses: COLMAPPose[] = input.images.map((img, idx) => ({
      imageId: img.id,
      rotation: [1, 0, 0, 0], // Identity rotation
      translation: [idx * 0.1, 0, 0], // Small translation per frame
      registered: true,
    }));

    const camera: COLMAPCamera = {
      model: 'SIMPLE_PINHOLE',
      width: input.images[0].width,
      height: input.images[0].height,
      params: [
        input.intrinsics?.fx || input.images[0].width,
        input.intrinsics?.cx || input.images[0].width / 2,
        input.intrinsics?.cy || input.images[0].height / 2,
      ],
    };

    return {
      success: true,
      data: {
        cameras: [camera],
        poses,
        points3D: [],
        numMatches: input.images.length * 100,
        stats: {
          numRegisteredImages: input.images.length,
          numPoints: 0,
          meanTrackLength: 0,
          meanReprojectionError: 1.0,
        },
      },
      inferenceTime: input.images.length * 1000,
    };
  }

  destroy(): void {
    console.log('[StubCOLMAP] Destroyed');
  }
}

class StubFaceEmbeddingModel implements FaceEmbeddingModel {
  async initialize(_config: FaceEmbeddingConfig): Promise<void> {
    console.log('[StubFaceEmbedding] Initialized');
  }

  async embed(_input: FaceEmbeddingInput): Promise<InferenceResult<FaceEmbeddingOutput>> {
    // Generate random embedding
    const embedding = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      embedding[i] = Math.random() * 2 - 1;
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
    for (let i = 0; i < 512; i++) {
      embedding[i] /= norm;
    }

    return {
      success: true,
      data: {
        faceDetected: true,
        faceBox: { x: 100, y: 100, width: 200, height: 200 },
        embedding,
        confidence: 0.95,
      },
      inferenceTime: 50,
    };
  }

  computeDistance(embedding1: Float32Array, embedding2: Float32Array): number {
    let dot = 0;
    for (let i = 0; i < embedding1.length; i++) {
      dot += embedding1[i] * embedding2[i];
    }
    return 1 - dot; // Cosine distance
  }

  destroy(): void {
    console.log('[StubFaceEmbedding] Destroyed');
  }
}

class StubVideoDiffusionModel implements VideoDiffusionModel {
  async initialize(_config: VideoDiffusionConfig): Promise<void> {
    console.log('[StubVideoDiffusion] Initialized');
  }

  async refine(input: VideoDiffusionInput): Promise<InferenceResult<VideoDiffusionOutput>> {
    // Just return the input frames (no actual refinement in stub)
    return {
      success: true,
      data: { frames: input.frames },
      inferenceTime: input.frames.length * 500,
    };
  }

  destroy(): void {
    console.log('[StubVideoDiffusion] Destroyed');
  }
}
