"""
DepthCrafter Depth Estimation Service for Orbit.
Runs on Modal with A100 GPU.
"""

import modal
from typing import List, Optional
import numpy as np

from ..config import app, DEPTH_IMAGE, GPU_CONFIG, volume, VOLUME_PATH


@app.cls(
    image=DEPTH_IMAGE,
    gpu=GPU_CONFIG["depth"],
    volumes={VOLUME_PATH: volume},
    timeout=900,  # Longer timeout for diffusion-based depth
)
class DepthCrafterService:
    """DepthCrafter video depth estimation service."""

    @modal.enter()
    def load_model(self):
        """Load DepthCrafter model on container startup."""
        import torch
        from diffusers import DiffusionPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load DepthCrafter pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            "tencent/DepthCrafter",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

        print(f"[DepthCrafter] Model loaded on {self.device}")

    @modal.method()
    def estimate_depth(
        self,
        job_id: str,
        frames: List[bytes],
        width: int,
        height: int,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.0,
    ) -> dict:
        """
        Estimate depth for video frames.

        Args:
            job_id: Unique job identifier
            frames: List of RGB frame bytes
            width: Frame width
            height: Frame height
            num_inference_steps: Diffusion steps (lower = faster)
            guidance_scale: Classifier-free guidance scale

        Returns:
            dict with depth maps and quality metrics
        """
        import torch
        import numpy as np
        from pathlib import Path
        from PIL import Image
        import cv2

        print(f"[DepthCrafter] Processing {len(frames)} frames for job {job_id}")

        # Convert bytes to PIL images
        pil_frames = []
        for frame_bytes in frames:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
            pil_frames.append(Image.fromarray(frame))

        # Process in chunks to manage memory
        chunk_size = 16
        all_depths = []
        all_confidences = []

        for i in range(0, len(pil_frames), chunk_size):
            chunk = pil_frames[i:i + chunk_size]

            with torch.no_grad():
                outputs = self.pipe(
                    chunk,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_type="np",
                )

            # outputs.depth is a list of depth arrays
            for depth in outputs.depth:
                # Normalize depth to [0, 1]
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                all_depths.append(depth_norm.astype(np.float32))

                # Estimate confidence from depth variance
                confidence = self._estimate_confidence(depth_norm)
                all_confidences.append(confidence)

        # Compute temporal consistency
        temporal_consistency = self._compute_temporal_consistency(all_depths)

        # Compute edge stability (with masks if available)
        edge_stability = 1.0  # Default, would need masks for proper computation

        # Save depth maps to volume
        output_dir = Path(VOLUME_PATH) / job_id / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_paths = []
        for i, depth in enumerate(all_depths):
            # Save as 16-bit PNG for precision
            depth_16bit = (depth * 65535).astype(np.uint16)
            depth_path = output_dir / f"depth_{i:06d}.png"
            cv2.imwrite(str(depth_path), depth_16bit)
            depth_paths.append(str(depth_path))

        volume.commit()

        return {
            "job_id": job_id,
            "num_frames": len(all_depths),
            "depth_paths": depth_paths,
            "quality": {
                "frame_confidences": all_confidences,
                "temporal_consistency": temporal_consistency,
                "edge_stability": edge_stability,
            },
        }

    def _estimate_confidence(self, depth: np.ndarray) -> float:
        """Estimate depth confidence from statistics."""
        # Confidence based on:
        # - Reasonable variance (not flat, not noisy)
        # - No extreme values
        variance = np.var(depth)
        valid_ratio = np.sum((depth > 0.01) & (depth < 0.99)) / depth.size

        variance_score = 1.0 if 0.01 < variance < 0.3 else 0.5
        confidence = valid_ratio * 0.6 + variance_score * 0.4

        return float(confidence)

    def _compute_temporal_consistency(self, depths: List[np.ndarray]) -> float:
        """Compute temporal consistency across depth frames."""
        if len(depths) < 2:
            return 1.0

        consistencies = []
        for i in range(1, len(depths)):
            # Compute relative depth change
            diff = np.abs(depths[i] - depths[i-1])
            avg_diff = np.mean(diff)
            consistency = max(0, 1 - avg_diff * 5)
            consistencies.append(consistency)

        return float(np.mean(consistencies))


@app.function(
    image=DEPTH_IMAGE,
    gpu=GPU_CONFIG["depth"],
    volumes={VOLUME_PATH: volume},
    timeout=900,
)
def estimate_depth_standalone(
    job_id: str,
    frames: List[bytes],
    width: int,
    height: int,
) -> dict:
    """Standalone function to estimate depth."""
    service = DepthCrafterService()
    return service.estimate_depth(job_id, frames, width, height)
