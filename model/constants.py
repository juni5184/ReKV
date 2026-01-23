"""
Shared constants for video processing and model configuration.
"""

# Video processing constants for Qwen2.5-VL
VIDEO_MAXFRAMES = 130
VIDEO_MAX_TOKEN_NUM = 130  # Alias for compatibility
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

# Default FPS settings
DEFAULT_FPS = 2.0
FPS_MIN_FRAMES = 4

# Frame tokens per frame (for KV cache calculations)
N_FRAME_TOKENS = 196
