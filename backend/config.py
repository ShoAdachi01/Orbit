"""
Configuration for Modal backend services.
"""

import modal

# Modal app definition
app = modal.App("orbit-backend")

# Shared volume for storing job results
volume = modal.Volume.from_name("orbit-jobs", create_if_missing=True)
VOLUME_PATH = "/data"

# GPU configurations for different models
GPU_CONFIG = {
    "sam2": modal.gpu.A100(size="40GB"),
    "tapir": modal.gpu.A100(size="40GB"),
    "depth": modal.gpu.A100(size="40GB"),
    "colmap": modal.gpu.A10G(),
}

# Container images
SAM2_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "opencv-python",
        "pillow",
        "hydra-core",
        "iopath",
    )
    .run_commands(
        "pip install git+https://github.com/facebookresearch/sam2.git"
    )
)

TAPIR_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "opencv-python",
        "mediapy",
        "jax[cuda12]",
        "flax",
        "chex",
    )
)

DEPTH_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "numpy",
        "opencv-python",
        "pillow",
    )
)

COLMAP_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("colmap", "ffmpeg", "libgl1-mesa-glx")
    .pip_install(
        "pycolmap",
        "numpy",
        "opencv-python",
        "pillow",
    )
)

ORCHESTRATOR_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "numpy",
        "opencv-python",
        "pillow",
        "pydantic",
        "fastapi",
    )
)
