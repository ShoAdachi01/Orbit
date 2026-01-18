/**
 * Orbit Core Type Definitions
 *
 * Data Contracts (from PRD):
 * - Pose convention: T_world_from_cam, right-handed, meters, row-major 4×4
 * - Intrinsics: fx, fy, cx, cy in pixels + distortion model (brown-conrady default)
 * - Masks: lossless PNG sequence (uint8 0/255)
 * - Camera path resampling: slerp rotation + linear translation + linear fov to render FPS
 */

// ============================================================================
// Core Math Types
// ============================================================================

/** Row-major 4x4 transformation matrix */
export type Matrix4x4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number
];

/** 3D vector */
export type Vector3 = [number, number, number];

/** Quaternion [x, y, z, w] */
export type Quaternion = [number, number, number, number];

// ============================================================================
// Camera Types
// ============================================================================

/** Brown-Conrady distortion model */
export interface BrownConradyDistortion {
  model: 'brown-conrady';
  k1: number;
  k2: number;
  k3: number;
  p1: number;
  p2: number;
}

/** Camera intrinsics in pixels */
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  distortion?: BrownConradyDistortion;
}

/**
 * Camera pose: T_world_from_cam
 * Transforms points from camera space to world space
 * Right-handed coordinate system, units in meters
 */
export interface CameraPose {
  /** Row-major 4x4 transformation matrix */
  transform: Matrix4x4;
  /** Timestamp in seconds from video start */
  timestamp: number;
  /** Frame index */
  frameIndex: number;
}

/** Camera keyframe for path authoring */
export interface CameraKeyframe {
  timestamp: number;
  position: Vector3;
  rotation: Quaternion;
  fov?: number;
}

/** Camera path for orbit animation */
export interface CameraPath {
  keyframes: CameraKeyframe[];
  /** Interpolation settings */
  interpolation: {
    rotation: 'slerp';
    translation: 'linear';
    fov: 'linear';
  };
  /** Target FPS for path resampling */
  targetFps: number;
}

// ============================================================================
// Orbit Bounds (Enforced Constraints)
// ============================================================================

export interface OrbitBounds {
  /** Maximum yaw in degrees (default ±20°) */
  maxYaw: number;
  /** Maximum pitch in degrees (default ±10°) */
  maxPitch: number;
  /** Maximum roll in degrees (default ±3°) */
  maxRoll: number;
  /** Maximum translation in meters */
  maxTranslation: number;
  /** Maximum translation as percentage of median scene depth */
  maxTranslationDepthPercent: number;
  /** Whether to clamp to parallax limits */
  clampToParallax: boolean;
}

export const DEFAULT_ORBIT_BOUNDS: OrbitBounds = {
  maxYaw: 20,
  maxPitch: 10,
  maxRoll: 3,
  maxTranslation: 0.10,
  maxTranslationDepthPercent: 2,
  clampToParallax: true,
};

// ============================================================================
// Quality Metrics
// ============================================================================

/** Segmentation/mask quality metrics */
export interface MaskQualityMetrics {
  /** Overall mask quality score (0-1) */
  score: number;
  /** Subject area ratio per frame (0-1) */
  subjectAreaRatios: number[];
  /** Edge jitter score */
  edgeJitter: number;
  /** Leak score (high-frequency fragments outside main component) */
  leakScore: number;
  /** Frames where subject covers >65% */
  highCoverageFrameCount: number;
  /** Total frame count */
  totalFrames: number;
}

/** Pose estimation quality metrics */
export interface PoseQualityMetrics {
  /** Overall pose quality score (0-1) */
  score: number;
  /** Inlier ratio from RANSAC */
  inlierRatio: number;
  /** Median reprojection error in pixels */
  medianReprojectionError: number;
  /** Track coverage across image quadrants */
  quadrantCoverage: [boolean, boolean, boolean, boolean];
  /** Percentage of frames with >=3 quadrants populated */
  goodCoverageFramePercent: number;
  /** High-frequency pose jitter energy */
  jitterScore: number;
}

/** Track quality metrics (TAPIR foreground tracking) */
export interface TrackQualityMetrics {
  /** Number of tracks kept after filtering */
  numTracksKept: number;
  /** Number of tracks discarded */
  numTracksDiscarded: number;
  /** Median track lifespan in frames */
  medianLifespan: number;
  /** Median track confidence */
  medianConfidence: number;
}

/** Depth estimation quality metrics */
export interface DepthQualityMetrics {
  /** Per-frame depth confidence scores */
  frameConfidences: number[];
  /** Temporal consistency score */
  temporalConsistency: number;
  /** Edge stability around subject boundary */
  edgeStability: number;
}

/** Combined quality report */
export interface QualityReport {
  mask: MaskQualityMetrics;
  pose: PoseQualityMetrics;
  track: TrackQualityMetrics;
  depth: DepthQualityMetrics;
  /** Selected rendering mode */
  mode: OrbitMode;
  /** Enforced orbit bounds for this scene */
  enforcedBounds: OrbitBounds;
  /** Reasons for mode selection */
  modeReasons: string[];
}

// ============================================================================
// Orbit Modes (Fallback System)
// ============================================================================

export type OrbitMode =
  | 'full-orbit'      // Normal full parallax
  | 'micro-parallax'  // Reduced orbit clamps
  | '2.5d-subject'    // Subject as billboard
  | 'render-only';    // No interactive 3D, stabilized export only

// ============================================================================
// OrbitScene Pack (Output Artifacts)
// ============================================================================

/** Camera configuration for the scene */
export interface SceneCamera {
  intrinsics: CameraIntrinsics;
  /** Reference frame pose */
  referencePose: CameraPose;
  /** All estimated poses */
  poses: CameraPose[];
}

/** OrbitScene pack manifest */
export interface OrbitSceneManifest {
  version: '1.0';
  /** Scene identifier */
  id: string;
  /** Creation timestamp */
  createdAt: string;
  /** Source video info */
  source: {
    filename: string;
    duration: number;
    fps: number;
    width: number;
    height: number;
  };
  /** Asset paths relative to pack root */
  assets: {
    backgroundSplat: string;      // bg.splat
    subjectSplat: string;         // subject_4d.splat
    camera: string;               // camera.json
    quality: string;              // quality.json
    lod?: string;                 // lod/ directory
  };
  /** Quality report */
  quality: QualityReport;
}

// ============================================================================
// Pipeline Types
// ============================================================================

/** Job status */
export type JobStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed';

/** Pipeline stage */
export type PipelineStage =
  | 'upload'
  | 'segmentation'
  | 'pose-estimation'
  | 'depth-estimation'
  | 'tracking'
  | 'background-reconstruction'
  | 'subject-reconstruction'
  | 'base-render'
  | 'refinement'
  | 'export';

/** Progress event */
export interface ProgressEvent {
  jobId: string;
  stage: PipelineStage;
  progress: number;  // 0-100
  message?: string;
  timestamp: number;
}

/** Job result */
export interface JobResult {
  jobId: string;
  status: JobStatus;
  /** OrbitScene pack path (if successful) */
  scenePackPath?: string;
  /** Base render path */
  baseRenderPath?: string;
  /** Final refined render path */
  finalRenderPath?: string;
  /** Quality report */
  quality?: QualityReport;
  /** Error message (if failed) */
  error?: string;
}

// ============================================================================
// Anchor Frame System (Identity Lock)
// ============================================================================

/** Anchor frame for identity preservation */
export interface AnchorFrame {
  frameIndex: number;
  timestamp: number;
  /** Sharpness score */
  sharpness: number;
  /** Face/logo visibility score */
  identityVisibility: number;
  /** Motion blur score (lower is better) */
  motionBlur: number;
  /** Face embedding if detected */
  faceEmbedding?: Float32Array;
  /** Crop region for identity features */
  cropRegion?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

/** Identity drift detection result */
export interface IdentityDriftResult {
  /** Maximum drift from anchor embeddings */
  maxDrift: number;
  /** Mean drift from anchor embeddings */
  meanDrift: number;
  /** Whether drift exceeds threshold */
  driftExceeded: boolean;
  /** Frame indices with high drift */
  highDriftFrames: number[];
}
