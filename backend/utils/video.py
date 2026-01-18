"""
Video I/O utilities for Orbit backend.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import io


def decode_frames(
    frame_bytes_list: List[bytes],
    width: int,
    height: int,
    channels: int = 3,
) -> List[np.ndarray]:
    """
    Decode raw frame bytes to numpy arrays.

    Args:
        frame_bytes_list: List of raw RGB/RGBA bytes
        width: Frame width
        height: Frame height
        channels: Number of channels (3 for RGB, 4 for RGBA)

    Returns:
        List of numpy arrays with shape (height, width, channels)
    """
    frames = []
    for frame_bytes in frame_bytes_list:
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = frame.reshape(height, width, channels)
        frames.append(frame)
    return frames


def encode_frames(frames: List[np.ndarray]) -> List[bytes]:
    """
    Encode numpy frames to raw bytes.

    Args:
        frames: List of numpy arrays

    Returns:
        List of raw bytes
    """
    return [frame.tobytes() for frame in frames]


def load_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
) -> Tuple[List[np.ndarray], int, int, float]:
    """
    Load video frames from file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        target_fps: Resample to this FPS (None = use source FPS)

    Returns:
        Tuple of (frames, width, height, fps)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame sampling
    fps = target_fps if target_fps else source_fps
    frame_step = source_fps / fps if target_fps else 1.0

    frames = []
    frame_idx = 0
    next_sample = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_sample:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            next_sample += frame_step

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames, width, height, fps


def save_video_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> str:
    """
    Save frames as video file.

    Args:
        frames: List of RGB numpy arrays
        output_path: Output video path
        fps: Output frame rate
        codec: Video codec (mp4v, avc1, etc.)

    Returns:
        Path to saved video
    """
    import cv2

    if not frames:
        raise ValueError("No frames to save")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path


def resize_frames(
    frames: List[np.ndarray],
    target_width: int,
    target_height: int,
    interpolation: str = "bilinear",
) -> List[np.ndarray]:
    """
    Resize frames to target dimensions.

    Args:
        frames: List of numpy arrays
        target_width: Target width
        target_height: Target height
        interpolation: Interpolation method (nearest, bilinear, bicubic)

    Returns:
        List of resized frames
    """
    import cv2

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

    resized = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=interp)
        resized.append(resized_frame)

    return resized


def apply_masks(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    background_color: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply masks to separate foreground and background.

    Args:
        frames: List of RGB frames
        masks: List of binary masks (255 = foreground, 0 = background)
        background_color: Color to fill masked regions

    Returns:
        Tuple of (foreground_frames, background_frames)
    """
    import cv2

    foreground_frames = []
    background_frames = []

    for frame, mask in zip(frames, masks):
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Create binary mask
        fg_mask = mask > 127

        # Foreground: keep subject, fill background with color
        fg_frame = frame.copy()
        fg_frame[~fg_mask] = background_color

        # Background: keep background, fill subject with color
        bg_frame = frame.copy()
        bg_frame[fg_mask] = background_color

        foreground_frames.append(fg_frame)
        background_frames.append(bg_frame)

    return foreground_frames, background_frames


def extract_subject_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    padding: float = 0.1,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a cropped region around the masked subject.

    Args:
        frame: RGB frame
        mask: Binary mask
        padding: Padding ratio around bounding box

    Returns:
        Tuple of (cropped_frame, bbox (x1, y1, x2, y2))
    """
    import cv2

    # Find bounding box of mask
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return frame, (0, 0, frame.shape[1], frame.shape[0])

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Add padding
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding)
    pad_y = int(height * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)

    cropped = frame[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def frames_to_base64(frames: List[np.ndarray], format: str = "png") -> List[str]:
    """
    Convert frames to base64-encoded strings.

    Args:
        frames: List of RGB frames
        format: Image format (png, jpg)

    Returns:
        List of base64-encoded strings
    """
    import base64
    import cv2

    encoded = []
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Encode to format
        ext = ".png" if format == "png" else ".jpg"
        _, buffer = cv2.imencode(ext, frame_bgr)

        # Base64 encode
        b64 = base64.b64encode(buffer).decode("utf-8")
        encoded.append(b64)

    return encoded


def base64_to_frames(encoded: List[str]) -> List[np.ndarray]:
    """
    Convert base64-encoded strings to frames.

    Args:
        encoded: List of base64-encoded strings

    Returns:
        List of RGB frames
    """
    import base64
    import cv2

    frames = []
    for b64 in encoded:
        # Decode base64
        buffer = base64.b64decode(b64)
        nparr = np.frombuffer(buffer, np.uint8)

        # Decode image
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    return frames


def compute_optical_flow(
    frames: List[np.ndarray],
    method: str = "farneback",
) -> List[np.ndarray]:
    """
    Compute optical flow between consecutive frames.

    Args:
        frames: List of RGB frames
        method: Flow method (farneback, dis)

    Returns:
        List of flow fields (N-1 flows for N frames)
    """
    import cv2

    flows = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

        if method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        else:
            # DIS optical flow (faster)
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(gray1, gray2, None)

        flows.append(flow)

    return flows


def warp_frame(
    frame: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    """
    Warp frame using optical flow.

    Args:
        frame: RGB frame
        flow: Optical flow field

    Returns:
        Warped frame
    """
    import cv2

    h, w = frame.shape[:2]
    flow_map = np.column_stack((
        np.tile(np.arange(w), h) + flow[:, :, 0].flatten(),
        np.repeat(np.arange(h), w) + flow[:, :, 1].flatten(),
    )).reshape(h, w, 2).astype(np.float32)

    warped = cv2.remap(frame, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR)
    return warped
