"""
TAPIR Point Tracking Service for Orbit.
Runs on Modal with A100 GPU.
"""

import modal
from typing import List, Optional
import numpy as np

from ..config import app, TAPIR_IMAGE, GPU_CONFIG, volume, VOLUME_PATH


@app.cls(
    image=TAPIR_IMAGE,
    gpu=GPU_CONFIG["tapir"],
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
class TAPIRService:
    """TAPIR point tracking service."""

    @modal.enter()
    def load_model(self):
        """Load TAPIR model on container startup."""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load TAPIR model
        # Using the PyTorch implementation
        self.model = torch.hub.load(
            "google-deepmind/tapnet",
            "tapir_checkpoint",
            pretrained=True,
        ).to(self.device).eval()

        print(f"[TAPIR] Model loaded on {self.device}")

    @modal.method()
    def track_points(
        self,
        job_id: str,
        frames: List[bytes],
        width: int,
        height: int,
        query_points: List[dict],
        query_frame: int = 0,
        mask_frames: Optional[List[bytes]] = None,
    ) -> dict:
        """
        Track points through video frames.

        Args:
            job_id: Unique job identifier
            frames: List of RGB frame bytes
            width: Frame width
            height: Frame height
            query_points: List of {"x": int, "y": int} points to track
            query_frame: Frame index where points are defined
            mask_frames: Optional masks to sample points from foreground

        Returns:
            dict with tracks and quality metrics
        """
        import torch
        import numpy as np
        from pathlib import Path
        import json

        print(f"[TAPIR] Tracking {len(query_points)} points through {len(frames)} frames")

        # Convert bytes to numpy arrays
        np_frames = []
        for frame_bytes in frames:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
            np_frames.append(frame)

        # Stack frames into video tensor [T, H, W, C]
        video = np.stack(np_frames, axis=0)

        # Normalize to [0, 1]
        video_tensor = torch.from_numpy(video).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        video_tensor = video_tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        # If no query points provided and we have masks, sample from foreground
        if not query_points and mask_frames:
            query_points = self._sample_points_from_mask(
                mask_frames[query_frame], width, height, num_points=100
            )

        # Prepare query points tensor
        points = torch.tensor(
            [[query_frame, p["y"], p["x"]] for p in query_points],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, N, 3]

        # Run tracking
        with torch.no_grad():
            outputs = self.model(video_tensor, points)

        # Extract tracks and visibility
        tracks = outputs["tracks"][0].cpu().numpy()  # [N, T, 2]
        visibility = outputs["visibles"][0].cpu().numpy()  # [N, T]
        confidence = outputs.get("expected_dist", torch.ones_like(outputs["visibles"]))[0].cpu().numpy()

        # Convert to confidence (inverse of expected distance)
        confidence = 1.0 / (1.0 + confidence)

        # Build track list
        track_list = []
        for i in range(len(query_points)):
            track_points = []
            for t in range(len(frames)):
                track_points.append({
                    "frame_index": t,
                    "x": float(tracks[i, t, 0]),
                    "y": float(tracks[i, t, 1]),
                    "visible": bool(visibility[i, t]),
                    "confidence": float(confidence[i, t]),
                })
            track_list.append({
                "id": i,
                "points": track_points,
            })

        # Compute quality metrics
        quality = self._compute_track_quality(track_list)

        # Filter tracks based on quality
        filtered_tracks = self._filter_tracks(track_list)

        # Save tracks to volume
        output_dir = Path(VOLUME_PATH) / job_id / "tracks"
        output_dir.mkdir(parents=True, exist_ok=True)

        tracks_path = output_dir / "tracks.json"
        with open(tracks_path, "w") as f:
            json.dump(filtered_tracks, f)

        volume.commit()

        return {
            "job_id": job_id,
            "num_tracks": len(filtered_tracks),
            "tracks_path": str(tracks_path),
            "tracks": filtered_tracks,
            "quality": quality,
        }

    def _sample_points_from_mask(
        self,
        mask_bytes: bytes,
        width: int,
        height: int,
        num_points: int = 100,
    ) -> List[dict]:
        """Sample points from foreground mask."""
        import numpy as np

        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(height, width)

        # Find foreground pixels
        fg_y, fg_x = np.where(mask > 127)

        if len(fg_x) == 0:
            # Fallback to grid sampling
            return self._grid_sample_points(width, height, num_points)

        # Random sample from foreground
        indices = np.random.choice(len(fg_x), min(num_points, len(fg_x)), replace=False)

        return [{"x": int(fg_x[i]), "y": int(fg_y[i])} for i in indices]

    def _grid_sample_points(self, width: int, height: int, num_points: int) -> List[dict]:
        """Sample points on a grid."""
        grid_size = int(np.sqrt(num_points))
        step_x = width / (grid_size + 1)
        step_y = height / (grid_size + 1)

        points = []
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                points.append({"x": int(i * step_x), "y": int(j * step_y)})

        return points[:num_points]

    def _compute_track_quality(self, tracks: List[dict]) -> dict:
        """Compute track quality metrics."""
        if not tracks:
            return {
                "num_tracks_kept": 0,
                "num_tracks_discarded": 0,
                "median_lifespan": 0,
                "median_confidence": 0.0,
            }

        lifespans = []
        confidences = []

        for track in tracks:
            # Lifespan = number of visible frames
            visible_count = sum(1 for p in track["points"] if p["visible"])
            lifespans.append(visible_count)

            # Average confidence
            avg_conf = np.mean([p["confidence"] for p in track["points"] if p["visible"]])
            confidences.append(avg_conf)

        return {
            "num_tracks_kept": len(tracks),
            "num_tracks_discarded": 0,  # Will be updated after filtering
            "median_lifespan": float(np.median(lifespans)),
            "median_confidence": float(np.median(confidences)),
        }

    def _filter_tracks(
        self,
        tracks: List[dict],
        min_confidence: float = 0.6,
        min_lifespan: int = 15,
    ) -> List[dict]:
        """Filter tracks based on quality thresholds."""
        filtered = []

        for track in tracks:
            # Check lifespan
            visible_count = sum(1 for p in track["points"] if p["visible"])
            if visible_count < min_lifespan:
                continue

            # Check confidence
            avg_conf = np.mean([p["confidence"] for p in track["points"] if p["visible"]])
            if avg_conf < min_confidence:
                continue

            filtered.append(track)

        return filtered


@app.function(
    image=TAPIR_IMAGE,
    gpu=GPU_CONFIG["tapir"],
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def track_points_standalone(
    job_id: str,
    frames: List[bytes],
    width: int,
    height: int,
    query_points: List[dict],
    mask_frames: Optional[List[bytes]] = None,
) -> dict:
    """Standalone function to track points."""
    service = TAPIRService()
    return service.track_points(job_id, frames, width, height, query_points, 0, mask_frames)
