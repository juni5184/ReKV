"""
Streaming video processing module optimized for fixed FPS sampling with video_chunk_frames support.

This module provides precise FPS-based frame sampling and consistent spatial resolution
for chunked/streaming video processing workflows.

Key features:
- Precise FPS-based frame sampling (exact intervals, not uniform distribution)
- video_chunk_frames support for consistent spatial resolution across chunks
- Decord backend only (optimized for your use case)
- Standalone module for streaming workflows
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional
import decord
import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model.constants import VIDEO_MAX_TOKEN_NUM


logger = logging.getLogger(__name__)

# Image processing constants
MAX_RATIO = 200
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

# Video processing constants
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * 28 * 28 # 100352
VIDEO_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * 28 * 28 # 602112

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 768))

VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))

logger.info(f"Streaming video processing initialized: FPS_MAX_FRAMES={FPS_MAX_FRAMES}, VIDEO_TOTAL_PIXELS={VIDEO_TOTAL_PIXELS}")


# ========== Helper Functions ==========

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert image to RGB, handling RGBA with white background."""
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


# ========== Frame Sampling Functions ==========

def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
    start_frame: int = 0,
    return_indices: bool = False,
) -> int | tuple[int, list[int]]:
    """Calculate frame count and indices for fixed FPS streaming video processing.

    Handles non-integer FPS values (e.g., 29.97, 24.97) precisely by computing
    each frame index directly: frame_idx = start_frame + i * (video_fps / target_fps)

    This avoids cumulative rounding errors from incremental addition.

    Args:
        ele (dict): Configuration dictionary with 'fps' key:
            - fps: Target FPS for sampling (default: 2.0)
            - min_frames: Minimum frames to sample (default: 4)
        total_frames (int): Total frames available in the sampling range
        video_fps (int | float): Original video FPS (can be non-integer like 29.97)
        start_frame (int): Starting frame index (default: 0)
        return_indices (bool): If True, return (nframes, indices) tuple with precise frame positions

    Returns:
        int: Number of frames to sample (if return_indices=False)
        tuple[int, list[int]]: (nframes, frame_indices) with precise FPS-aligned positions (if return_indices=True)

    Raises:
        ValueError: If nframes is not in interval [FRAME_FACTOR, total_frames]
    """
    # Fixed FPS-based sampling only (simplified for streaming use case)
    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    end_frame = start_frame + total_frames - 1

    # Compute frame indices directly to handle non-integer FPS precisely
    # Example: 29.97 FPS video, 2 FPS target → indices at [0, 14.985, 29.97, 44.955, ...]
    # Using direct calculation: frame_idx = start_frame + i * (video_fps / fps)
    indices = []
    i = 0
    while True:
        frame_idx = start_frame + i * (video_fps / fps)
        if frame_idx > end_frame:
            break
        indices.append(int(round(frame_idx)))
        i += 1

    # Apply min_frames constraint
    if len(indices) < min_frames:
        logger.warning(f"Not enough frames for FPS sampling ({len(indices)} < {min_frames}), using uniform sampling")
        nframes = min_frames
        indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    else:
        # Round down to FRAME_FACTOR
        nframes = floor_by_factor(len(indices), FRAME_FACTOR)
        indices = indices[:nframes]

    # Validation
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], got {nframes}")

    if return_indices:
        return nframes, indices
    return nframes


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(
        f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}"
    )
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> tuple[torch.Tensor, float]:
    """Read video using decord.VideoReader with precise FPS-based frame sampling.

    This function uses smart_nframes with return_indices=True to get exact frame indices
    based on target FPS, ensuring temporal precision instead of uniform sampling.

    Args:
        ele (dict): Configuration dictionary containing:
            - video: Path to video file (local path, file://, http://, https://)
            - video_start (optional): Start time in seconds
            - video_end (optional): End time in seconds
            - fps: Target sampling FPS (default: 2.0)
            - frames_per_video_chunk (optional): Enables streaming mode

    Returns:
        tuple[torch.Tensor, float]: Video tensor (T, C, H, W) and effective sample FPS
    """

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)

    # 18500, 25.0
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    # Calculate frame range (handles video_start/video_end)
    start_frame, end_frame, frame_count = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )

    # Get precise FPS-based frame indices
    nframes, frame_idxs = smart_nframes(
        ele,
        total_frames=frame_count,
        video_fps=video_fps,
        start_frame=start_frame,
        return_indices=True
    )

    # Load frames at exact indices
    frames = vr.get_batch(frame_idxs).asnumpy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # Convert to TCHW format

    # Calculate effective sample FPS
    sample_fps = nframes / max(frame_count, 1e-6) * video_fps

    logger.info(
        f"decord (streaming): {video_path=}, total_frames={total_frames}, "
        f"sampled_frames={nframes}, {video_fps=:.2f}, {sample_fps=:.2f}, "
        f"time={time.time() - st:.3f}s"
    )

    return frames, sample_fps


# ========== Main Video Processing Functions ==========

def fetch_video(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
    video_chunk_frames: int = -1
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Fetch and process video with support for video_chunk_frames for consistent spatial resolution.

    Args:
        ele (dict): Video configuration dictionary
        image_factor (int): Factor for dimension alignment (default: 28)
        return_video_sample_fps (bool): If True, return (video, sample_fps) tuple
        video_chunk_frames (int): Number of frames per chunk for resize computation.
            If > 0, computes max_pixels based on this value instead of total nframes,
            ensuring consistent spatial resolution across video chunks.

    Returns:
        torch.Tensor: Processed video tensor (T, C, H, W)
        tuple[torch.Tensor, float]: (video, sample_fps) if return_video_sample_fps=True
    """
    if not isinstance(ele["video"], str):
        raise ValueError("Streaming video processing only supports video file paths, not frame lists")

    # Propagate video_chunk_frames to ele dict for streaming mode
    if video_chunk_frames > 0:
        ele["frames_per_video_chunk"] = video_chunk_frames

    # Read video with precise FPS sampling
    video, sample_fps = _read_video_decord(ele)

    nframes, _, height, width = video.shape

    # Compute effective_nframes for resize calculation
    # This is the key optimization: use video_chunk_frames instead of total nframes
    if video_chunk_frames > 0:
        effective_nframes = video_chunk_frames
        logger.info(f"Using video_chunk_frames={video_chunk_frames} for resize (actual nframes={nframes})")
    else:
        effective_nframes = nframes

    # Calculate max_pixels based on effective_nframes
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS) # 128 * 28 * 28 = 100352
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS) # 128000 * 28 * 28 * 0.9 = 90316800
    # max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / effective_nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max(min(VIDEO_MAX_PIXELS, total_pixels / effective_nframes * FRAME_FACTOR), int(min_pixels * 1.05)))
    print("min, max_pixels:", min_pixels, max_pixels)

    # Allow user override but warn if exceeding limit
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]")
    max_pixels = min(max_pixels_supposed, max_pixels)

    # Compute resize dimensions
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    logger.info(
        f"Video resize: {height}x{width} → {resized_height}x{resized_width}, "
        f"nframes={nframes}, effective_nframes={effective_nframes}, max_pixels={max_pixels}"
    )

    # Resize video
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    if return_video_sample_fps:
        return video, sample_fps
    return video


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    """Extract vision information (images/videos) from conversation messages."""
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    video_chunk_frames: int = 60,
) -> tuple[list[Image.Image] | None, list[torch.Tensor] | None, Optional[dict]]:
    """Process vision information from conversations for streaming video workflows.

    Args:
        conversations: Conversation messages containing vision content
        return_video_kwargs (bool): If True, return video metadata dict
        video_chunk_frames (int): Number of frames per chunk for consistent spatial resolution

    Returns:
        tuple: (image_inputs, video_inputs, video_kwargs)
            - image_inputs: List of PIL Images or None
            - video_inputs: List of video tensors or None
            - video_kwargs: Dict with 'fps' list if return_video_kwargs=True, else None
    """
    vision_infos = extract_vision_info(conversations)

    # Note: This streaming module focuses on video processing
    # For image processing, you may want to import fetch_image from the original vision_process
    video_inputs = []
    video_sample_fps_list = []

    for vision_info in vision_infos:
        if "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info,
                return_video_sample_fps=True,
                video_chunk_frames=video_chunk_frames
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        elif "image" in vision_info or "image_url" in vision_info:
            logger.warning("Image processing not implemented in streaming module. Import fetch_image from vision_process if needed.")

    video_inputs = None if len(video_inputs) == 0 else video_inputs

    if return_video_kwargs:
        return None, video_inputs, {'fps': video_sample_fps_list}
    return None, video_inputs
