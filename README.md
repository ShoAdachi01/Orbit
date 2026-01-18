# Orbit

Create new camera angles from existing videos using 3D reconstruction and Veo 3.1 video generation.

## Overview

Orbit lets you pick a new camera angle from a video, preview that angle as a still image, and generate a new high-resolution video from that perspective using Veo 3.1.

Instead of full video reconstruction, Orbit builds a lightweight 3D proxy from a single anchor frame, allowing you to:
1. Choose a camera position interactively
2. Generate a preview still from that pose
3. Use the preview as the reference for Veo 3.1 video generation

## Features

- **Video Upload & Segment Selection** - Upload a video and select the segment you want to transform
- **Interactive 3D Viewer** - Point-and-click subject selection using SAM2 segmentation
- **Camera Orbit Controls** - Position the virtual camera within bounded yaw/pitch/roll
- **Preview Generation** - Generate still previews via Nano Banana Pro
- **Veo 3.1 Integration** - Generate new-angle videos with preserved context
- **Audio Overlay** - Re-lay original audio track over generated video

## Tech Stack

**Frontend**
- TypeScript
- Vite
- WebGL/WebGPU viewer

**Backend**
- Python/FastAPI
- Modal (serverless GPU)
- SAM2 for segmentation
- SAM 3D Body/Objects for 3D reconstruction

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- Python 3.10+
- Modal account (for GPU backend)

### Installation

```bash
# Clone the repository
git clone https://github.com/ShoAdachi01/Orbit.git
cd Orbit

# Install frontend dependencies
npm install

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### Configuration

Copy the environment template and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `NANO_BANANA_API_URL` / `NANO_BANANA_API_KEY` - For preview generation
- `VEO_API_URL` / `VEO_API_KEY` - For Veo 3.1 video generation

### Development

```bash
# Start frontend dev server
npm run dev

# Deploy backend to Modal
cd backend
modal deploy main.py
```

### Testing

```bash
npm test
```

## Project Structure

```
Orbit/
├── backend/           # Modal serverless backend
│   ├── main.py       # FastAPI routes and Veo integration
│   └── utils/        # Video, mesh, and remote utilities
├── src/
│   ├── api/          # API clients (Modal, Orbit)
│   ├── ml/           # ML model interfaces
│   ├── pipeline/     # Job orchestration
│   ├── schemas/      # TypeScript types
│   ├── viewer/       # SAM3D interactive viewer
│   └── legacy/       # Archived modules
├── index.html        # Main UI
└── veo-architecture.md  # Architecture documentation
```

## Pipeline

1. **Video Ingest** - Extract anchor frame with minimal motion blur
2. **3D Proxy Construction** - Run SAM2 masks, SAM 3D Body for humans, SAM 3D Objects for foreground
3. **Camera Selection** - User positions camera in 3D viewer
4. **Preview Still** - Generate new-angle still via Nano Banana Pro
5. **Context Preservation** - VLM scene summary + audio transcription
6. **Veo 3.1 Generation** - Generate video from preview still with context
7. **Audio Overlay** - Combine original audio with generated video

## License

MIT
