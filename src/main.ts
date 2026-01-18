import { ModalClient } from './api/ModalClient';
import { quatFromEuler } from './utils/math';
import { SAM3DViewer } from './viewer/SAM3DViewer';

type Step = 'upload' | 'camera' | 'results';
type Stage = 'upload' | 'snapshot' | 'reconstruction' | 'preview-generation' | 'veo-generation';

type StatusState = 'idle' | 'running' | 'done' | 'error';

type StageStatus = {
  stage?: string;
  progress: number;
  message?: string;
};

const modalClient = new ModalClient();
let currentJobId: string | null = null;
let videoId: string | null = null;
let orientation: 'horizontal' | 'vertical' = 'horizontal';
let snapshotTimeSec = 1;
let isProcessing = false;
let reconstructionReady = false;
let lastClick: { x: number; y: number } | null = null;
let viewer: SAM3DViewer | null = null;
let viewerFailed = false;
let reconstructionResult: {
  snapshotUrl?: string;
  reconstructionUrl?: string;
  orientation?: 'horizontal' | 'vertical';
} | null = null;
let videoDurationSec = 0;
let segmentStartSec = 0;
let segmentEndSec = 0;

let isDragging = false;
let dragMoved = false;
let dragStart = { x: 0, y: 0 };
let dragYaw = 0;
let dragPitch = 0;

const stageOrder: Stage[] = [
  'upload',
  'snapshot',
  'reconstruction',
  'preview-generation',
  'veo-generation',
];

const stageMap: Record<string, Stage> = {
  upload: 'upload',
  snapshot: 'snapshot',
  reconstruction: 'reconstruction',
  'preview-generation': 'preview-generation',
  'veo-generation': 'veo-generation',
  segmentation: 'reconstruction',
  'pose-estimation': 'reconstruction',
  'depth-estimation': 'reconstruction',
  tracking: 'reconstruction',
  'background-reconstruction': 'reconstruction',
  'subject-reconstruction': 'reconstruction',
  'base-render': 'preview-generation',
  refinement: 'veo-generation',
  export: 'veo-generation',
};

const statusPill = getEl<HTMLDivElement>('status-pill');
const errorMessage = getEl<HTMLDivElement>('error-message');
const stepChips = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-step]'));
const stepPanels = Array.from(document.querySelectorAll<HTMLElement>('[data-step-panel]'));

const uploadZone = getEl<HTMLDivElement>('upload-zone');
const fileInput = getEl<HTMLInputElement>('file-input');
const fileMeta = getEl<HTMLDivElement>('file-meta');
const uploadProgress = getEl<HTMLDivElement>('upload-progress');
const uploadProgressFill = getEl<HTMLDivElement>('upload-progress-fill');
const uploadStatusText = getEl<HTMLDivElement>('upload-status-text');
const continueBtn = getEl<HTMLButtonElement>('continue-btn');

const orientationBadge = getEl<HTMLDivElement>('orientation-badge');
const orientationButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-orientation]'));

const segmentStartSlider = getEl<HTMLInputElement>('segment-start-slider');
const segmentEndSlider = getEl<HTMLInputElement>('segment-end-slider');
const segmentStartValue = getEl<HTMLDivElement>('segment-start-value');
const segmentEndValue = getEl<HTMLDivElement>('segment-end-value');
const segmentLengthValue = getEl<HTMLDivElement>('segment-length-value');

const snapshotSlider = getEl<HTMLInputElement>('snapshot-slider');
const snapshotValue = getEl<HTMLDivElement>('snapshot-value');

const canvasShell = getEl<HTMLDivElement>('canvas-shell');
const canvas = getEl<HTMLCanvasElement>('orbit-canvas');
const canvasOverlay = getEl<HTMLDivElement>('canvas-overlay');
const overlayTitle = getEl<HTMLDivElement>('overlay-title');
const overlaySub = getEl<HTMLDivElement>('overlay-sub');

const yawSlider = getEl<HTMLInputElement>('yaw-slider');
const pitchSlider = getEl<HTMLInputElement>('pitch-slider');
const rollSlider = getEl<HTMLInputElement>('roll-slider');
const distanceSlider = getEl<HTMLInputElement>('distance-slider');
const yawValue = getEl<HTMLDivElement>('yaw-value');
const pitchValue = getEl<HTMLDivElement>('pitch-value');
const rollValue = getEl<HTMLDivElement>('roll-value');
const distanceValue = getEl<HTMLDivElement>('distance-value');

const pipelineProgress = getEl<HTMLDivElement>('pipeline-progress');
const pipelineProgressFill = getEl<HTMLDivElement>('pipeline-progress-fill');
const pipelineStatusText = getEl<HTMLDivElement>('pipeline-status-text');
const stageList = Array.from(document.querySelectorAll<HTMLLIElement>('[data-stage]'));

const backBtn = getEl<HTMLButtonElement>('back-btn');
const generateBtn = getEl<HTMLButtonElement>('generate-btn');

const snapshotImage = getEl<HTMLImageElement>('snapshot-image');
const snapshotPlaceholder = getEl<HTMLDivElement>('snapshot-placeholder');
const previewImage = getEl<HTMLImageElement>('preview-image');
const previewPlaceholder = getEl<HTMLDivElement>('preview-placeholder');
const veoVideo = getEl<HTMLVideoElement>('veo-video');
const videoPlaceholder = getEl<HTMLDivElement>('video-placeholder');
const downloadBtn = getEl<HTMLAnchorElement>('download-btn');
const reconstructionLink = getEl<HTMLAnchorElement>('reconstruction-link');
const adjustCameraBtn = getEl<HTMLButtonElement>('adjust-camera-btn');
const startOverBtn = getEl<HTMLButtonElement>('start-over-btn');

function getEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element: ${id}`);
  }
  return el as T;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function setStatus(text: string, state: StatusState = 'idle') {
  statusPill.textContent = text;
  statusPill.dataset.state = state;
}

function showError(message: string) {
  errorMessage.textContent = message;
  errorMessage.classList.remove('hidden');
  setStatus('Error', 'error');
}

function hideError() {
  errorMessage.classList.add('hidden');
}

function setCanvasOverlay(visible: boolean, title: string, subtitle?: string) {
  canvasOverlay.classList.toggle('hidden', !visible);
  overlayTitle.textContent = title;
  overlaySub.textContent = subtitle || '';
}

function updateActionState() {
  generateBtn.disabled = isProcessing || !reconstructionReady;
  backBtn.disabled = isProcessing;
}

function goToStep(step: Step) {
  stepPanels.forEach((panel) => {
    const panelStep = panel.dataset.stepPanel as Step;
    panel.classList.toggle('hidden', panelStep !== step);
  });

  stepChips.forEach((chip) => {
    const chipStep = chip.dataset.step as Step;
    chip.setAttribute('aria-current', chipStep === step ? 'step' : 'false');
  });

  if (step === 'camera') {
    void ensureViewer();
    updateActionState();
  }
}

function updateOrientationUI() {
  orientationBadge.textContent = orientation.charAt(0).toUpperCase() + orientation.slice(1);
  orientationButtons.forEach((btn) => {
    btn.dataset.selected = btn.dataset.orientation === orientation ? 'true' : 'false';
  });
}

function updateSnapshotUI() {
  snapshotValue.textContent = `${snapshotTimeSec.toFixed(1)}s`;
}

function getSegmentDuration(): number {
  return Math.max(segmentEndSec - segmentStartSec, 0);
}

function updateSegmentUI() {
  segmentStartValue.textContent = `${segmentStartSec.toFixed(1)}s`;
  segmentEndValue.textContent = `${segmentEndSec.toFixed(1)}s`;
  segmentLengthValue.textContent = `${getSegmentDuration().toFixed(1)}s`;
}

function updateSnapshotRange() {
  const segmentDuration = getSegmentDuration();
  const maxSnapshot = Math.max(segmentDuration, 0.1);
  snapshotSlider.max = maxSnapshot.toFixed(1);
  snapshotTimeSec = clamp(snapshotTimeSec, 0, segmentDuration);
  snapshotSlider.value = snapshotTimeSec.toFixed(1);
  updateSnapshotUI();
}

function updateCameraUI() {
  const yaw = Number(yawSlider.value);
  const pitch = Number(pitchSlider.value);
  const roll = Number(rollSlider.value);
  const distance = Number(distanceSlider.value);

  yawValue.textContent = `${yaw.toFixed(1)} deg`;
  pitchValue.textContent = `${pitch.toFixed(1)} deg`;
  rollValue.textContent = `${roll.toFixed(1)} deg`;
  distanceValue.textContent = distance.toFixed(1);

  if (viewer && !viewerFailed) {
    viewer.setCamera(yaw, pitch, roll, distance);
  }
}

function getCameraPose() {
  const yaw = Number(yawSlider.value);
  const pitch = Number(pitchSlider.value);
  const roll = Number(rollSlider.value);
  const distance = Number(distanceSlider.value);

  const yawRad = (yaw * Math.PI) / 180;
  const pitchRad = (pitch * Math.PI) / 180;

  const x = distance * Math.sin(yawRad) * Math.cos(pitchRad);
  const y = distance * Math.sin(pitchRad);
  const z = distance * Math.cos(yawRad) * Math.cos(pitchRad);

  return {
    position: [x, y, z] as [number, number, number],
    rotation: quatFromEuler(yaw, pitch, roll),
    fov: 45,
  };
}

function updateStageList(stage: Stage, stageProgress: number) {
  const activeIndex = Math.max(stageOrder.indexOf(stage), 0);

  stageOrder.forEach((name, index) => {
    const item = stageList.find((el) => el.dataset.stage === name);
    if (!item) return;
    if (index < activeIndex) {
      item.dataset.state = 'done';
    } else if (index === activeIndex) {
      item.dataset.state = stageProgress >= 100 ? 'done' : 'running';
    } else {
      item.dataset.state = 'idle';
    }
  });

  const normalized = ((activeIndex + stageProgress / 100) / stageOrder.length) * 100;
  pipelineProgressFill.style.width = `${normalized}%`;
}

function resetMedia() {
  snapshotImage.classList.add('hidden');
  previewImage.classList.add('hidden');
  veoVideo.classList.add('hidden');
  snapshotPlaceholder.classList.remove('hidden');
  previewPlaceholder.classList.remove('hidden');
  videoPlaceholder.classList.remove('hidden');
  downloadBtn.setAttribute('href', '#');
  downloadBtn.setAttribute('aria-disabled', 'true');
  reconstructionLink.setAttribute('href', '#');
  reconstructionLink.classList.add('hidden');
}

function resetProgress() {
  uploadProgressFill.style.width = '0%';
  pipelineProgressFill.style.width = '0%';
  stageList.forEach((item) => {
    item.dataset.state = 'idle';
  });
}

function resetPipelineState() {
  reconstructionReady = false;
  reconstructionResult = null;
  currentJobId = null;
  lastClick = null;
  pipelineProgress.classList.add('hidden');
  pipelineStatusText.textContent = 'Processing...';
  if (viewer) {
    viewer.clearMesh();
  }
  setCanvasOverlay(true, 'Upload a clip to begin', 'We will build a 3D proxy from a snapshot.');
  updateActionState();
}

function invalidateReconstruction(reason: string) {
  if (!reconstructionReady) return;
  reconstructionReady = false;
  reconstructionResult = null;
  lastClick = null;
  if (viewer) {
    viewer.clearMesh();
  }
  setCanvasOverlay(true, 'Rebuild required', reason);
  updateActionState();
}

async function loadReconstructionMesh(url?: string) {
  if (!url) {
    throw new Error('Reconstruction asset missing.');
  }

  await ensureViewer();
  if (!viewer || viewerFailed) return;

  const bounds = await viewer.loadFromUrl(url);
  if (bounds) {
    const minDistance = Math.max(bounds.radius * 0.6, 0.8);
    const suggested = Math.max(bounds.radius * 2.5, minDistance + 0.1);
    const maxDistance = Math.max(bounds.radius * 6, suggested + 1);

    distanceSlider.min = minDistance.toFixed(1);
    distanceSlider.max = maxDistance.toFixed(1);
    distanceSlider.value = suggested.toFixed(1);
  }

  updateCameraUI();
}

function setProcessingState(state: boolean) {
  isProcessing = state;
  updateActionState();
}

function normalizeStage(stage?: string): Stage {
  if (!stage) return 'reconstruction';
  return stageMap[stage] || 'reconstruction';
}

function resizeCanvas() {
  const rect = canvasShell.getBoundingClientRect();
  canvas.width = Math.floor(rect.width);
  canvas.height = Math.floor(rect.height);
  if (viewer && !viewerFailed) {
    viewer.resize(canvas.width, canvas.height);
  } else {
    drawCanvas();
  }
}

function drawCanvas() {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = 'rgba(15, 118, 110, 0.15)';
  ctx.lineWidth = 1;

  const gridSize = 40;
  for (let x = 0; x <= width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = 0; y <= height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(31, 26, 20, 0.2)';
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (lastClick) {
    ctx.fillStyle = 'rgba(15, 118, 110, 0.8)';
    ctx.beginPath();
    ctx.arc(lastClick.x, lastClick.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function ensureViewer() {
  if (viewer || viewerFailed) return;
  try {
    viewer = new SAM3DViewer(canvas);
    await viewer.initialize();
    viewer.start();
    resizeCanvas();
    updateCameraUI();
  } catch (error) {
    viewerFailed = true;
    console.error('Viewer initialization failed:', error);
    showError('WebGL is not available. 3D preview is disabled.');
    setCanvasOverlay(true, '3D preview unavailable', 'WebGL is required to render the scene.');
    drawCanvas();
  }
}

async function loadVideoMetadata(file: File) {
  return new Promise<{ duration: number; width: number; height: number }>((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    const url = URL.createObjectURL(file);
    video.src = url;

    video.onloadedmetadata = () => {
      const meta = {
        duration: video.duration || 0,
        width: video.videoWidth || 0,
        height: video.videoHeight || 0,
      };
      URL.revokeObjectURL(url);
      resolve(meta);
    };

    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to read video metadata'));
    };
  });
}

async function handleFile(file: File) {
  if (!file.type.startsWith('video/')) {
    showError('Please upload a valid video file.');
    return;
  }

  hideError();
  resetMedia();
  resetProgress();
  resetPipelineState();
  continueBtn.disabled = true;
  uploadProgress.classList.remove('hidden');
  setStatus('Uploading', 'running');

  try {
    const meta = await loadVideoMetadata(file);
    videoDurationSec = Math.max(meta.duration || 10, 1);
    segmentStartSec = 0;
    segmentEndSec = videoDurationSec;

    segmentStartSlider.min = '0';
    segmentStartSlider.max = videoDurationSec.toFixed(1);
    segmentEndSlider.min = '0';
    segmentEndSlider.max = videoDurationSec.toFixed(1);
    segmentStartSlider.value = segmentStartSec.toFixed(1);
    segmentEndSlider.value = segmentEndSec.toFixed(1);
    updateSegmentUI();

    updateSnapshotRange();

    fileMeta.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB, ${videoDurationSec.toFixed(1)}s)`;
    uploadStatusText.textContent = 'Uploading video...';

    const result = await modalClient.uploadVideo(file, (p) => {
      uploadProgressFill.style.width = `${p.percent}%`;
      uploadStatusText.textContent = `Uploading video... ${p.percent}%`;
    });

    videoId = result.videoId;
    continueBtn.disabled = false;
    uploadStatusText.textContent = 'Upload complete.';
    setStatus('Uploaded', 'done');
  } catch (error) {
    console.error('Upload failed:', error);
    uploadProgress.classList.add('hidden');
    showError(`Upload failed: ${(error as Error).message}`);
  }
}

async function startReconstruction() {
  if (!videoId) {
    showError('Upload a video before building the 3D preview.');
    goToStep('upload');
    return;
  }

  if (isProcessing) return;

  if (reconstructionReady && reconstructionResult?.reconstructionUrl) {
    setStatus('Camera ready', 'done');
    if (!viewerFailed) {
      setCanvasOverlay(false, '', '');
    }
    updateActionState();
    return;
  }

  reconstructionReady = false;
  reconstructionResult = null;
  if (viewer) {
    viewer.clearMesh();
  }
  updateActionState();

  hideError();
  setProcessingState(true);
  pipelineProgress.classList.remove('hidden');
  setStatus('Reconstructing', 'running');
  updateStageList('snapshot', 0);
  pipelineStatusText.textContent = 'Extracting snapshot...';
  setCanvasOverlay(true, 'Building 3D proxy', 'Extracting snapshot and running SAM 3D Body.');

  try {
    const submitResult = await modalClient.submitJobWithVideo({
      videoId,
      maxFrames: 150,
      targetFps: 10,
      snapshotTimeSec,
      orientation,
      pipeline: 'reconstruction',
      segmentStartSec,
      segmentEndSec,
    });

    currentJobId = submitResult.jobId;

    const result = await modalClient.waitForCompletion(
      currentJobId,
      (status: StageStatus) => {
        const stage = normalizeStage(status.stage);
        const progress = status.progress || 0;
        updateStageList(stage, progress);
        pipelineStatusText.textContent = `${stage.replace(/-/g, ' ')}... ${Math.round(progress)}%`;
      },
      2000,
      3600000
    );

    await onReconstructionComplete(result);
  } catch (error) {
    console.error('Reconstruction failed:', error);
    reconstructionReady = false;
    reconstructionResult = null;
    showError(`Reconstruction failed: ${(error as Error).message}`);
    setStatus('Error', 'error');
    setCanvasOverlay(true, 'Reconstruction failed', 'Try a different clip or snapshot time.');
    pipelineProgress.classList.add('hidden');
  } finally {
    setProcessingState(false);
  }
}

async function startGeneration() {
  if (!videoId) {
    showError('Upload a video before generating a preview.');
    goToStep('upload');
    return;
  }

  if (!reconstructionReady) {
    showError('Complete 3D reconstruction before generating a preview.');
    return;
  }

  if (isProcessing) return;

  hideError();
  setProcessingState(true);
  pipelineProgress.classList.remove('hidden');
  setStatus('Processing', 'running');
  updateStageList('snapshot', 0);
  pipelineStatusText.textContent = 'Starting generation...';
  setCanvasOverlay(true, 'Generating preview', 'Nano Banana preview and Veo 3.1 render.');

  try {
    const submitResult = await modalClient.submitJobWithVideo({
      videoId,
      maxFrames: 150,
      targetFps: 10,
      snapshotTimeSec,
      orientation,
      cameraPose: getCameraPose(),
      pipeline: 'generation',
      segmentStartSec,
      segmentEndSec,
      options: {
        durationSec: getSegmentDuration(),
      },
    });

    currentJobId = submitResult.jobId;

    const result = await modalClient.waitForCompletion(
      currentJobId,
      (status: StageStatus) => {
        const stage = normalizeStage(status.stage);
        const progress = status.progress || 0;
        updateStageList(stage, progress);
        pipelineStatusText.textContent = `${stage.replace(/-/g, ' ')}... ${Math.round(progress)}%`;
      },
      2000,
      3600000
    );

    const mergedResult = {
      snapshotUrl: result.snapshotUrl || reconstructionResult?.snapshotUrl,
      reconstructionUrl: result.reconstructionUrl || reconstructionResult?.reconstructionUrl,
      previewImageUrl: result.previewImageUrl,
      veoVideoUrl: result.veoVideoUrl,
      orientation: result.orientation || reconstructionResult?.orientation,
    };

    await onGenerationComplete(mergedResult);
  } catch (error) {
    console.error('Processing failed:', error);
    showError(`Processing failed: ${(error as Error).message}`);
    setStatus('Error', 'error');
    setCanvasOverlay(true, 'Generation failed', 'Try another camera angle.');
    pipelineProgress.classList.add('hidden');
  } finally {
    setProcessingState(false);
  }
}

async function onReconstructionComplete(result: {
  snapshotUrl?: string;
  reconstructionUrl?: string;
  orientation?: 'horizontal' | 'vertical';
}) {
  reconstructionResult = result;

  updateStageList('reconstruction', 100);
  pipelineStatusText.textContent = 'Reconstruction complete';
  pipelineProgress.classList.add('hidden');

  if (result.orientation) {
    orientation = result.orientation;
    updateOrientationUI();
  }

  await loadReconstructionMesh(result.reconstructionUrl);
  reconstructionReady = true;

  setStatus('Camera ready', 'done');
  if (viewerFailed) {
    setCanvasOverlay(true, '3D preview unavailable', 'WebGL is required to render the scene.');
  } else {
    setCanvasOverlay(false, '', '');
  }
  updateActionState();
}

async function onGenerationComplete(result: {
  snapshotUrl?: string;
  previewImageUrl?: string;
  veoVideoUrl?: string;
  reconstructionUrl?: string;
  orientation?: 'horizontal' | 'vertical';
}) {
  setStatus('Complete', 'done');
  updateStageList('veo-generation', 100);
  pipelineStatusText.textContent = 'Complete';
  pipelineProgress.classList.add('hidden');

  if (result.snapshotUrl) {
    snapshotImage.src = result.snapshotUrl;
    snapshotImage.classList.remove('hidden');
    snapshotPlaceholder.classList.add('hidden');
  }

  if (result.previewImageUrl) {
    previewImage.src = result.previewImageUrl;
    previewImage.classList.remove('hidden');
    previewPlaceholder.classList.add('hidden');
  }

  if (result.veoVideoUrl) {
    veoVideo.src = result.veoVideoUrl;
    veoVideo.classList.remove('hidden');
    videoPlaceholder.classList.add('hidden');
    downloadBtn.setAttribute('href', result.veoVideoUrl);
    downloadBtn.setAttribute('aria-disabled', 'false');
  }

  if (result.reconstructionUrl) {
    reconstructionLink.setAttribute('href', result.reconstructionUrl);
    reconstructionLink.classList.remove('hidden');
  }

  if (result.orientation) {
    orientation = result.orientation;
    updateOrientationUI();
  }

  setCanvasOverlay(false, '', '');
  goToStep('results');
}

function resetState() {
  currentJobId = null;
  videoId = null;
  isProcessing = false;
  snapshotTimeSec = 1;
  videoDurationSec = 0;
  segmentStartSec = 0;
  segmentEndSec = 0;
  uploadProgress.classList.add('hidden');
  pipelineProgress.classList.add('hidden');
  fileMeta.textContent = 'No file selected';
  continueBtn.disabled = true;
  resetMedia();
  resetProgress();
  resetPipelineState();
  segmentStartSlider.value = '0';
  segmentEndSlider.value = '0';
  segmentStartSlider.min = '0';
  segmentStartSlider.max = '0';
  segmentEndSlider.min = '0';
  segmentEndSlider.max = '0';
  updateSegmentUI();
  updateSnapshotRange();
  setStatus('Idle', 'idle');
  goToStep('upload');
}

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragging');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragging');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragging');
  const file = e.dataTransfer?.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const target = e.target as HTMLInputElement;
  const file = target.files?.[0];
  if (file) handleFile(file);
});

orientationButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const next = btn.dataset.orientation as 'horizontal' | 'vertical';
    if (next !== orientation) {
      orientation = next;
      updateOrientationUI();
      invalidateReconstruction('Orientation changed. Rebuild the 3D proxy to continue.');
    }
  });
});

segmentStartSlider.addEventListener('input', (e) => {
  const value = Number((e.target as HTMLInputElement).value);
  segmentStartSec = clamp(value, 0, videoDurationSec);
  if (segmentStartSec >= segmentEndSec) {
    segmentEndSec = clamp(segmentStartSec + 0.1, 0, videoDurationSec);
    segmentEndSlider.value = segmentEndSec.toFixed(1);
  }

  segmentEndSlider.min = segmentStartSec.toFixed(1);
  updateSegmentUI();
  updateSnapshotRange();
  invalidateReconstruction('Segment changed. Rebuild the 3D proxy to continue.');
});

segmentEndSlider.addEventListener('input', (e) => {
  const value = Number((e.target as HTMLInputElement).value);
  segmentEndSec = clamp(value, 0, videoDurationSec);
  if (segmentEndSec <= segmentStartSec) {
    segmentStartSec = clamp(segmentEndSec - 0.1, 0, videoDurationSec);
    segmentStartSlider.value = segmentStartSec.toFixed(1);
  }

  segmentStartSlider.max = segmentEndSec.toFixed(1);
  updateSegmentUI();
  updateSnapshotRange();
  invalidateReconstruction('Segment changed. Rebuild the 3D proxy to continue.');
});

snapshotSlider.addEventListener('input', (e) => {
  snapshotTimeSec = Number((e.target as HTMLInputElement).value);
  updateSnapshotUI();
  invalidateReconstruction('Snapshot time changed. Rebuild the 3D proxy to continue.');
});

[yawSlider, pitchSlider, rollSlider, distanceSlider].forEach((slider) => {
  slider.addEventListener('input', () => {
    updateCameraUI();
  });
});

canvas.addEventListener('pointerdown', (e) => {
  if (!videoId || isProcessing || !reconstructionReady) return;
  canvas.setPointerCapture(e.pointerId);
  isDragging = true;
  dragMoved = false;
  dragStart = { x: e.clientX, y: e.clientY };
  dragYaw = Number(yawSlider.value);
  dragPitch = Number(pitchSlider.value);
});

canvas.addEventListener('pointermove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - dragStart.x;
  const dy = e.clientY - dragStart.y;
  if (Math.abs(dx) + Math.abs(dy) > 2) {
    dragMoved = true;
  }

  const yawMin = Number(yawSlider.min);
  const yawMax = Number(yawSlider.max);
  const pitchMin = Number(pitchSlider.min);
  const pitchMax = Number(pitchSlider.max);
  const sensitivity = 0.12;

  const nextYaw = clamp(dragYaw + dx * sensitivity, yawMin, yawMax);
  const nextPitch = clamp(dragPitch - dy * sensitivity, pitchMin, pitchMax);

  yawSlider.value = nextYaw.toFixed(1);
  pitchSlider.value = nextPitch.toFixed(1);
  updateCameraUI();
});

const endDrag = (e: PointerEvent) => {
  if (!isDragging) return;
  isDragging = false;
  canvas.releasePointerCapture(e.pointerId);

  if (!dragMoved) {
    const rect = canvas.getBoundingClientRect();
    lastClick = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    if (!viewer || viewerFailed) {
      drawCanvas();
    }
    startGeneration();
  }
};

canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

canvas.addEventListener('wheel', (e) => {
  if (!videoId || isProcessing || !reconstructionReady) return;
  e.preventDefault();
  const delta = Math.sign(e.deltaY) * 0.3;
  const distanceMin = Number(distanceSlider.min);
  const distanceMax = Number(distanceSlider.max);
  const current = Number(distanceSlider.value);
  const next = clamp(current + delta, distanceMin, distanceMax);
  distanceSlider.value = next.toFixed(1);
  updateCameraUI();
}, { passive: false });

continueBtn.addEventListener('click', () => {
  if (!videoId) return;
  goToStep('camera');
  startReconstruction();
});

backBtn.addEventListener('click', () => {
  if (isProcessing) return;
  goToStep('upload');
});

generateBtn.addEventListener('click', () => {
  startGeneration();
});

adjustCameraBtn.addEventListener('click', () => {
  goToStep('camera');
});

startOverBtn.addEventListener('click', () => {
  resetState();
});

const resizeObserver = new ResizeObserver(() => {
  resizeCanvas();
});
resizeObserver.observe(canvasShell);

updateOrientationUI();
updateSnapshotUI();
updateCameraUI();
updateSegmentUI();
resetMedia();
resetProgress();
resetPipelineState();
setStatus('Idle', 'idle');
goToStep('upload');
resizeCanvas();
