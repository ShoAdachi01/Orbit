/**
 * Orbit Configuration
 * Environment-specific settings for Modal GPU backend
 */

export interface OrbitConfig {
  /** Backend API URL (Modal endpoint) */
  backendUrl: string;
  /** Use stub models instead of real backend */
  useStubs: boolean;
  /** Debug mode */
  debug: boolean;
  /** Polling interval for job status (ms) */
  pollInterval: number;
  /** Maximum polling attempts before timeout */
  maxPollAttempts: number;
}

// Default Modal backend URL (ASGI app endpoint)
const DEFAULT_BACKEND_URL = 'https://shoadachi01--orbit-backend-fastapi-app.modal.run';

// Default config - uses Modal backend
const defaultConfig: OrbitConfig = {
  backendUrl: import.meta.env.VITE_BACKEND_URL || DEFAULT_BACKEND_URL,
  useStubs: false, // Use real backend by default
  debug: true,
  pollInterval: 2000,
  maxPollAttempts: 300, // 10 minutes at 2s interval
};

// Export current config
export const config: OrbitConfig = defaultConfig;

/**
 * Override config at runtime
 */
export function setConfig(overrides: Partial<OrbitConfig>): void {
  Object.assign(config, overrides);
}

/**
 * Modal endpoint paths
 * ASGI app serves all endpoints from a single base URL with path routing
 */
const MODAL_ENDPOINTS = {
  submitJob: 'submit_job',
  getJobStatus: 'get_job_status',
  getJobResult: 'get_job_result',
  cancelJob: 'cancel_job',
  listJobs: 'list_jobs',
  health: 'health',
  uploadVideo: 'upload_video',
  getSceneFile: 'get_scene_file',
} as const;

/**
 * Get the Modal endpoint URL for a specific function
 */
export function getEndpointUrl(endpoint: keyof typeof MODAL_ENDPOINTS): string {
  if (!config.backendUrl) {
    throw new Error('Backend URL not configured. Set VITE_BACKEND_URL or use setConfig()');
  }

  const cleanBase = config.backendUrl.replace(/\/$/, '');
  return `${cleanBase}/${MODAL_ENDPOINTS[endpoint]}`;
}

/**
 * Get all endpoint URLs for debugging
 */
export function getEndpoints(): Record<keyof typeof MODAL_ENDPOINTS, string> {
  if (!config.backendUrl) {
    return {} as Record<keyof typeof MODAL_ENDPOINTS, string>;
  }

  return {
    submitJob: getEndpointUrl('submitJob'),
    getJobStatus: getEndpointUrl('getJobStatus'),
    getJobResult: getEndpointUrl('getJobResult'),
    cancelJob: getEndpointUrl('cancelJob'),
    listJobs: getEndpointUrl('listJobs'),
    health: getEndpointUrl('health'),
    uploadVideo: getEndpointUrl('uploadVideo'),
    getSceneFile: getEndpointUrl('getSceneFile'),
  };
}

/**
 * Get the scene pack URL for loading scene assets
 */
export function getScenePackUrl(jobId: string): string {
  const cleanBase = config.backendUrl.replace(/\/$/, '');
  return `${cleanBase}/get_scene_file?job_id=${jobId}&filename=`;
}
