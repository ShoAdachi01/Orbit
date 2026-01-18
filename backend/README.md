# Orbit Backend

Modal serverless GPU backend for the Orbit video processing pipeline.

## Quick Start

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate with Modal
modal token new

# 3. Deploy to Modal
modal deploy backend/main.py
```

## Prerequisites

- Modal account at https://modal.com
- Modal CLI installed and authenticated
- Sufficient Modal credits for GPU usage

## Deployment

Deploy to Modal:

```bash
cd /path/to/Orbit
modal deploy backend/main.py
```

This will output the endpoint URLs:
```
✓ Created objects.
├── 🔨 Created submit_job => https://yourname--orbit-backend-submit-job.modal.run
├── 🔨 Created get_job_status => https://yourname--orbit-backend-get-job-status.modal.run
├── 🔨 Created get_job_result => https://yourname--orbit-backend-get-job-result.modal.run
├── 🔨 Created cancel_job => https://yourname--orbit-backend-cancel-job.modal.run
├── 🔨 Created list_jobs => https://yourname--orbit-backend-list-jobs.modal.run
├── 🔨 Created health => https://yourname--orbit-backend-health.modal.run
```

## Configuration

Set the backend URL in your frontend:

```bash
# In your frontend project
export VITE_BACKEND_URL="https://shoadachi01--orbit-backend"
npm run dev
```

Or configure at runtime:

```typescript
import { setConfig } from './config';

setConfig({
  backendUrl: 'https://shoadachi01--orbit-backend',
  useStubs: false,
});
```

## API Endpoints

### POST /submit_job

Submit a video for processing.

**Request:**
```json
{
  "frames": ["base64_frame_1", "base64_frame_2", ...],
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "prompt_points": [{"x": 960, "y": 540}]
}
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "pending",
  "message": "Job submitted successfully"
}
```

### GET /get_job_status?job_id=xxx

Get job status.

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "processing",
  "stage": "segmentation",
  "progress": 45.0,
  "message": "Running SAM2 segmentation...",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:01:00Z"
}
```

### GET /get_job_result?job_id=xxx

Get completed job result.

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "mode": "full-orbit",
  "bounds": {
    "maxYaw": 20,
    "maxPitch": 10,
    "maxRoll": 3,
    "maxTranslation": 0.1
  },
  "quality": {
    "mask": { ... },
    "pose": { ... },
    "track": { ... },
    "depth": { ... }
  },
  "assets": {
    "masks": ["/data/job_abc123/masks/mask_000000.png", ...],
    "poses": "/data/job_abc123/poses/poses.json",
    "depth": ["/data/job_abc123/depth/depth_000000.png", ...],
    "tracks": "/data/job_abc123/tracks/tracks.json"
  }
}
```

### POST /cancel_job?job_id=xxx

Cancel a running job.

**Response:**
```json
{
  "cancelled": true,
  "job_id": "job_abc123"
}
```

### GET /list_jobs

List recent jobs.

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_abc123",
      "status": "completed",
      "stage": "completed",
      "progress": 100.0,
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:02:00Z"
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "orbit-backend",
  "version": "1.0.0"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Modal Backend                            │
│                                                              │
│  ┌──────────────────┐     ┌──────────────────┐              │
│  │  Web Endpoints   │────▶│   Orchestrator   │              │
│  │  (FastAPI)       │     │   (CPU)          │              │
│  └──────────────────┘     └────────┬─────────┘              │
│                                    │                         │
│           ┌────────────────────────┼────────────────────┐   │
│           │                        │                    │   │
│           ▼                        ▼                    ▼   │
│  ┌────────────────┐   ┌────────────────┐   ┌──────────────┐│
│  │  SAM2 Service  │   │  TAPIR Service │   │DepthCrafter  ││
│  │  (A100 40GB)   │   │  (A100 40GB)   │   │ (A100 40GB)  ││
│  └────────────────┘   └────────────────┘   └──────────────┘│
│                                                              │
│  ┌────────────────┐   ┌────────────────────────────────────┐│
│  │ COLMAP Service │   │        Modal Volume                ││
│  │   (A10G)       │   │    (Persistent Storage)            ││
│  └────────────────┘   └────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Cost Estimate

| Model | GPU | Memory | Est. Cost/min |
|-------|-----|--------|---------------|
| SAM2 | A100-40GB | ~20GB | $0.03 |
| TAPIR | A100-40GB | ~15GB | $0.03 |
| DepthCrafter | A100-40GB | ~25GB | $0.03 |
| COLMAP | A10G | ~8GB | $0.01 |

**Per video (10s, 300 frames):** ~$0.50-1.00 estimated

## Local Testing

Test locally with Modal's local run:

```bash
modal run backend/main.py
```

## Troubleshooting

### Container build fails

The ML model containers are large. If builds fail:
1. Check Modal's build logs
2. Ensure you have sufficient credits
3. Try building incrementally (comment out unused services)

### Out of memory

Reduce batch size or use smaller models:
- SAM2: Use `sam2_hiera_base` instead of `large`
- DepthCrafter: Reduce `num_inference_steps`

### Slow cold starts

GPU containers have cold start times (~30-60s). To keep them warm:
- Use Modal's `keep_warm` option
- Process multiple videos in batch
