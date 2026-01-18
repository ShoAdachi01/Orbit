# Orbit vNext: Single-Frame 3D Preview + Veo 3.1 Generation

## Goal
Enable a creator to pick a new camera angle from a simple 3D proxy of a video,
preview that angle as a still, and generate a new high-resolution video from
that angle using Veo 3.1.

## Core idea
Instead of full video reconstruction, we build a lightweight 3D proxy from a
single anchor frame (or a short segment if needed), then:
1) let the user choose a camera position,
2) generate a preview still from that pose via Nano Banana Pro,
3) use the preview as the reference start frame for Veo 3.1 video generation.

## Inputs and outputs
Inputs
- Source video
- User-selected segment (optional)
- Anchor frame (auto or user pick)
- Camera pose (from the 3D viewer)
- Output format (vertical or horizontal)

Outputs
- Proxy 3D scene for camera selection
- Preview still from the chosen angle
- Veo 3.1 video (same duration as chosen segment)
- Optional: original audio track re-laid over the generated video

## SAM 3D Objects and SAM 3D Body summary (from README)
- SAM 3D Objects reconstructs 3D shape, texture, and layout from a single image.
  It expects an image plus object mask(s) and can output a gaussian splat.
- SAM 3D Body reconstructs a full-body human mesh from a single image.
  It supports optional prompts (2D keypoints, masks) and uses the Momentum Human
  Rig (MHR) parametric mesh for pose + shape.
- The SAM 3D team provides an alignment notebook that demonstrates how to align
  SAM 3D Body and SAM 3D Objects in a shared frame of reference.
- Best fit: use SAM 3D Body for humans and SAM 3D Objects for non-human
  foreground assets; background still needs a simpler 2.5D layer.

## High-level pipeline
1) Video ingest
   - User picks the whole clip or a sub-segment.
   - Extract an anchor frame (sharp, minimal motion blur).
2) Proxy 3D construction
   - Compute masks (SAM/SAM2 or similar) and detect humans vs objects.
   - If a primary human is present, run SAM 3D Body for the human mesh.
   - Run SAM 3D Objects for other foreground objects using masks.
   - Align human mesh and object splats in a shared frame (SAM 3D alignment
     notebook reference).
   - Build a lightweight background layer (single-image depth + inpaint) to
     prevent holes during camera motion.
3) Camera selection
   - User orbits the camera within bounded yaw/pitch/roll and translation.
   - Save the pose and output aspect ratio (vertical or horizontal).
4) Preview still
   - Use Nano Banana Pro to generate a new-angle still from the anchor image
     + camera pose constraints.
   - Compare the preview to the proxy render; if drift is large, reduce orbit
     bounds or prompt a re-roll.
5) Context preservation
   - Run a VLM on the chosen segment for scene summary, actions, and key details.
   - Transcribe audio to capture spoken content.
6) Veo 3.1 generation
   - Provide the preview still as the reference start frame.
   - Inject scene context, action summary, camera direction, aspect ratio, and
     desired duration.
7) Audio (MVP)
   - Overlay original audio for the selected segment.
   - Accept that lip sync may be imperfect in the MVP.

## UX flow
1) Upload video
2) Select segment (or accept auto-selection)
3) Choose anchor frame
4) Use 3D proxy viewer to place camera
5) Generate preview still (Nano Banana Pro)
6) Approve and generate Veo 3.1 video
7) Download video with original audio track (optional)

## Quality gates and fallbacks
- If SAM 3D reconstruction is unstable or the preview drifts:
  - Clamp orbit to micro-parallax
  - Offer a static angle (no orbit)
  - Allow re-rolls with tighter prompts
- If background artifacts appear:
  - Strengthen depth smoothing or rely on a flat background layer

## Implementation TODO (ordered)
1) Define data contracts: camera pose, aspect ratio, orbit bounds, asset paths
2) Frame selection + segment UI
3) Mask extraction pipeline (SAM/SAM2)
4) SAM 3D Body inference integration (human mesh) and export to viewer format
5) SAM 3D Objects inference integration (foreground splats)
6) Alignment of human mesh + object splats into shared camera frame
7) Background layer from single-image depth (2.5D fallback)
8) Camera placement UI + pose export
9) Nano Banana Pro preview still generator
10) VLM context + audio transcription
11) Veo 3.1 prompt builder (pose + context + duration + aspect)
12) Audio overlay and export

## Open questions and risks
- SAM 3D Body quality on partial occlusions or cropped subjects
- SAM 3D Objects quality on complex scenes and cluttered backgrounds
- Alignment stability between SAM 3D Body meshes and SAM 3D Objects splats
- Consistency between proxy render and Nano Banana Pro output
- Veo 3.1 fidelity to the reference still and camera intent
- Lip sync quality (deferred from MVP)
