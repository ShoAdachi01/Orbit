"""
COLMAP Pose Estimation Service for Orbit.
Runs on Modal with A10G GPU.
"""

import modal
from typing import List, Optional, Tuple
import numpy as np

from ..config import app, COLMAP_IMAGE, GPU_CONFIG, volume, VOLUME_PATH


@app.cls(
    image=COLMAP_IMAGE,
    gpu=GPU_CONFIG["colmap"],
    volumes={VOLUME_PATH: volume},
    timeout=1200,  # COLMAP can take a while
)
class COLMAPService:
    """COLMAP structure-from-motion service."""

    @modal.enter()
    def setup(self):
        """Setup COLMAP environment."""
        import pycolmap
        print("[COLMAP] Service initialized")

    @modal.method()
    def reconstruct(
        self,
        job_id: str,
        frames: List[bytes],
        width: int,
        height: int,
        masks: Optional[List[bytes]] = None,
        camera_model: str = "SIMPLE_PINHOLE",
    ) -> dict:
        """
        Run structure-from-motion on video frames.

        Args:
            job_id: Unique job identifier
            frames: List of RGB frame bytes (background-masked)
            width: Frame width
            height: Frame height
            masks: Optional masks to exclude foreground
            camera_model: COLMAP camera model

        Returns:
            dict with poses, points, and quality metrics
        """
        import pycolmap
        import numpy as np
        from pathlib import Path
        import cv2
        import tempfile
        import shutil
        import json

        print(f"[COLMAP] Reconstructing from {len(frames)} frames for job {job_id}")

        # Create temp directory for COLMAP
        work_dir = Path(tempfile.mkdtemp())
        images_dir = work_dir / "images"
        images_dir.mkdir()

        try:
            # Save frames as images, applying masks if provided
            for i, frame_bytes in enumerate(frames):
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)

                # Apply mask to exclude foreground (for background-only reconstruction)
                if masks and i < len(masks):
                    mask = np.frombuffer(masks[i], dtype=np.uint8).reshape(height, width)
                    # Invert mask: we want background (mask=0) not foreground (mask=255)
                    bg_mask = mask < 127
                    # Set foreground to neutral gray (won't affect feature matching)
                    frame = frame.copy()
                    frame[~bg_mask] = 128

                frame_path = images_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Run COLMAP reconstruction
            database_path = work_dir / "database.db"
            sparse_dir = work_dir / "sparse"
            sparse_dir.mkdir()

            # Feature extraction
            pycolmap.extract_features(
                database_path=str(database_path),
                image_path=str(images_dir),
                camera_model=camera_model,
                sift_options=pycolmap.SiftExtractionOptions(),
            )

            # Feature matching (sequential for video)
            pycolmap.match_sequential(
                database_path=str(database_path),
                matching_options=pycolmap.SequentialMatchingOptions(
                    overlap=10,
                    quadratic_overlap=True,
                ),
            )

            # Incremental reconstruction
            maps = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(images_dir),
                output_path=str(sparse_dir),
            )

            if not maps:
                return self._create_failed_result(job_id, "Reconstruction failed")

            # Get the largest reconstruction
            reconstruction = max(maps.values(), key=lambda r: r.num_reg_images())

            # Extract poses
            poses = []
            registered_frames = set()

            for image_id, image in reconstruction.images.items():
                frame_idx = int(image.name.split("_")[1].split(".")[0])
                registered_frames.add(frame_idx)

                # Extract rotation quaternion (COLMAP uses qw, qx, qy, qz)
                qvec = image.cam_from_world.rotation.quat
                tvec = image.cam_from_world.translation

                poses.append({
                    "image_id": frame_idx,
                    "rotation": [float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])],
                    "translation": [float(tvec[0]), float(tvec[1]), float(tvec[2])],
                    "registered": True,
                })

            # Add unregistered frames
            for i in range(len(frames)):
                if i not in registered_frames:
                    poses.append({
                        "image_id": i,
                        "rotation": [1, 0, 0, 0],
                        "translation": [0, 0, 0],
                        "registered": False,
                    })

            # Sort by frame index
            poses.sort(key=lambda p: p["image_id"])

            # Extract 3D points
            points_3d = []
            for point_id, point in reconstruction.points3D.items():
                points_3d.append({
                    "id": int(point_id),
                    "position": [float(point.xyz[0]), float(point.xyz[1]), float(point.xyz[2])],
                    "color": [int(point.color[0]), int(point.color[1]), int(point.color[2])],
                    "error": float(point.error),
                    "track_length": len(point.track.elements),
                })

            # Extract camera
            camera = list(reconstruction.cameras.values())[0]
            camera_info = {
                "model": camera.model.name,
                "width": camera.width,
                "height": camera.height,
                "params": [float(p) for p in camera.params],
            }

            # Compute quality metrics
            quality = self._compute_quality_metrics(reconstruction, poses, len(frames))

            # Save results to volume
            output_dir = Path(VOLUME_PATH) / job_id / "poses"
            output_dir.mkdir(parents=True, exist_ok=True)

            poses_path = output_dir / "poses.json"
            with open(poses_path, "w") as f:
                json.dump({
                    "camera": camera_info,
                    "poses": poses,
                    "points_3d": points_3d[:1000],  # Limit for JSON size
                }, f)

            volume.commit()

            return {
                "job_id": job_id,
                "num_registered": len(registered_frames),
                "num_points": len(points_3d),
                "poses_path": str(poses_path),
                "camera": camera_info,
                "poses": poses,
                "quality": quality,
            }

        finally:
            # Cleanup
            shutil.rmtree(work_dir, ignore_errors=True)

    def _compute_quality_metrics(
        self,
        reconstruction,
        poses: List[dict],
        total_frames: int,
    ) -> dict:
        """Compute pose quality metrics."""
        import numpy as np

        registered_count = sum(1 for p in poses if p["registered"])
        registration_ratio = registered_count / total_frames

        # Compute reprojection errors
        errors = [p.error for p in reconstruction.points3D.values()]
        median_error = float(np.median(errors)) if errors else 0.0

        # Compute coverage (which quadrants have features)
        # This would need image dimensions, simplified here
        quadrant_coverage = [True, True, True, True]
        if registration_ratio < 0.5:
            quadrant_coverage = [True, True, False, False]

        # Compute jitter score (pose smoothness)
        jitter_score = self._compute_jitter_score(poses)

        # Overall score
        score = min(1.0, registration_ratio * 0.4 + (1 - median_error / 5) * 0.3 + (1 - jitter_score) * 0.3)

        return {
            "score": float(score),
            "inlier_ratio": float(registration_ratio),
            "median_reprojection_error": median_error,
            "quadrant_coverage": quadrant_coverage,
            "good_coverage_frame_percent": float(registration_ratio),
            "jitter_score": jitter_score,
        }

    def _compute_jitter_score(self, poses: List[dict]) -> float:
        """Compute pose jitter (high-frequency motion)."""
        import numpy as np

        registered_poses = [p for p in poses if p["registered"]]
        if len(registered_poses) < 3:
            return 0.0

        # Compute translation differences
        translations = np.array([p["translation"] for p in registered_poses])
        velocities = np.diff(translations, axis=0)
        accelerations = np.diff(velocities, axis=0)

        # Jitter = RMS of accelerations
        jitter = np.sqrt(np.mean(accelerations ** 2))

        # Normalize to [0, 1]
        return min(1.0, float(jitter))

    def _create_failed_result(self, job_id: str, reason: str) -> dict:
        """Create a failed result."""
        return {
            "job_id": job_id,
            "num_registered": 0,
            "num_points": 0,
            "poses_path": None,
            "camera": None,
            "poses": [],
            "quality": {
                "score": 0.0,
                "inlier_ratio": 0.0,
                "median_reprojection_error": float("inf"),
                "quadrant_coverage": [False, False, False, False],
                "good_coverage_frame_percent": 0.0,
                "jitter_score": 1.0,
            },
            "error": reason,
        }


@app.function(
    image=COLMAP_IMAGE,
    gpu=GPU_CONFIG["colmap"],
    volumes={VOLUME_PATH: volume},
    timeout=1200,
)
def reconstruct_standalone(
    job_id: str,
    frames: List[bytes],
    width: int,
    height: int,
    masks: Optional[List[bytes]] = None,
) -> dict:
    """Standalone function to run reconstruction."""
    service = COLMAPService()
    return service.reconstruct(job_id, frames, width, height, masks)
