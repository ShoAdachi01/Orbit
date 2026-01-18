"""
SAM2 Segmentation Service for Orbit.
Runs on Modal with A100 GPU.
"""

import modal
from typing import List, Optional
import numpy as np

from ..config import app, SAM2_IMAGE, GPU_CONFIG, volume, VOLUME_PATH


@app.cls(
    image=SAM2_IMAGE,
    gpu=GPU_CONFIG["sam2"],
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
class SAM2Service:
    """SAM2 video segmentation service."""

    @modal.enter()
    def load_model(self):
        """Load SAM2 model on container startup."""
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use the large model for best quality
        self.predictor = build_sam2_video_predictor(
            "sam2_hiera_large.yaml",
            "sam2_hiera_large.pt",
            device=self.device,
        )
        print(f"[SAM2] Model loaded on {self.device}")

    @modal.method()
    def segment_video(
        self,
        job_id: str,
        frames: List[bytes],
        width: int,
        height: int,
        prompt_points: Optional[List[dict]] = None,
        prompt_frame: int = 0,
    ) -> dict:
        """
        Segment subject from video frames.

        Args:
            job_id: Unique job identifier
            frames: List of RGB frame bytes
            width: Frame width
            height: Frame height
            prompt_points: Optional list of {"x": int, "y": int, "label": 0|1}
            prompt_frame: Frame index for prompts

        Returns:
            dict with masks, scores, and quality metrics
        """
        import torch
        import numpy as np
        from pathlib import Path
        import cv2

        print(f"[SAM2] Processing {len(frames)} frames for job {job_id}")

        # Convert bytes to numpy arrays
        np_frames = []
        for frame_bytes in frames:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
            np_frames.append(frame)

        # Initialize video state
        inference_state = self.predictor.init_state(video_path=None, images=np_frames)

        # Add prompts if provided, otherwise use center point
        if prompt_points:
            points = np.array([[p["x"], p["y"]] for p in prompt_points])
            labels = np.array([p.get("label", 1) for p in prompt_points])
        else:
            # Default: center of frame as foreground
            points = np.array([[width // 2, height // 2]])
            labels = np.array([1])

        # Add points to predictor
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_frame,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate through video
        masks = []
        scores = []

        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0).cpu().numpy().astype(np.uint8) * 255
            masks.append(mask.squeeze())

            # Compute confidence score
            score = float(torch.sigmoid(mask_logits[0]).mean().cpu())
            scores.append(score)

        # Compute quality metrics
        subject_area_ratios = []
        for mask in masks:
            ratio = np.sum(mask > 127) / (width * height)
            subject_area_ratios.append(float(ratio))

        # Compute edge jitter (boundary displacement between frames)
        edge_jitter = self._compute_edge_jitter(masks)

        # Compute leak score (fragments outside main component)
        leak_score = self._compute_leak_score(masks)

        # Save masks to volume
        output_dir = Path(VOLUME_PATH) / job_id / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_paths = []
        for i, mask in enumerate(masks):
            mask_path = output_dir / f"mask_{i:06d}.png"
            cv2.imwrite(str(mask_path), mask)
            mask_paths.append(str(mask_path))

        volume.commit()

        return {
            "job_id": job_id,
            "num_frames": len(masks),
            "mask_paths": mask_paths,
            "scores": scores,
            "quality": {
                "score": float(np.mean(scores)),
                "subject_area_ratios": subject_area_ratios,
                "edge_jitter": edge_jitter,
                "leak_score": leak_score,
                "high_coverage_frame_count": sum(1 for r in subject_area_ratios if r > 0.65),
                "total_frames": len(masks),
            },
        }

    def _compute_edge_jitter(self, masks: List[np.ndarray]) -> float:
        """Compute mask boundary displacement over time."""
        if len(masks) < 2:
            return 0.0

        import cv2

        total_jitter = 0
        for i in range(1, len(masks)):
            # Find contours
            prev_contours, _ = cv2.findContours(masks[i-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            curr_contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if prev_contours and curr_contours:
                # Compare centroids
                prev_moments = cv2.moments(prev_contours[0])
                curr_moments = cv2.moments(curr_contours[0])

                if prev_moments["m00"] > 0 and curr_moments["m00"] > 0:
                    prev_cx = prev_moments["m10"] / prev_moments["m00"]
                    prev_cy = prev_moments["m01"] / prev_moments["m00"]
                    curr_cx = curr_moments["m10"] / curr_moments["m00"]
                    curr_cy = curr_moments["m01"] / curr_moments["m00"]

                    displacement = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                    total_jitter += displacement

        # Normalize by frame dimensions
        avg_dim = (masks[0].shape[0] + masks[0].shape[1]) / 2
        return total_jitter / ((len(masks) - 1) * avg_dim)

    def _compute_leak_score(self, masks: List[np.ndarray]) -> float:
        """Compute leak score from mask fragments."""
        import cv2

        total_leak = 0
        for mask in masks:
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            if num_labels <= 2:  # Background + 1 component
                continue

            # Find largest component (excluding background)
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            total_area = np.sum(areas)
            largest_area = np.max(areas)

            # Leak = pixels not in largest component
            leak_pixels = total_area - largest_area
            leak_ratio = leak_pixels / total_area if total_area > 0 else 0
            total_leak += leak_ratio

        return total_leak / len(masks) if masks else 0.0


# Standalone function for direct invocation
@app.function(
    image=SAM2_IMAGE,
    gpu=GPU_CONFIG["sam2"],
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def segment_video_standalone(
    job_id: str,
    frames: List[bytes],
    width: int,
    height: int,
    prompt_points: Optional[List[dict]] = None,
) -> dict:
    """Standalone function to segment video."""
    service = SAM2Service()
    return service.segment_video(job_id, frames, width, height, prompt_points)
