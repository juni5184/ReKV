import math
import torch
from logzero import logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from model.patch import patch_hf
from model.streaming_vision_process import (
    fetch_video,
    FPS,
    FPS_MIN_FRAMES,
    FPS_MAX_FRAMES,
    IMAGE_FACTOR,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    FRAME_FACTOR,
)

# System prompt template - follows Qwen2.5-VL chat format
SYSTEM_PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

# Qwen2.5-VL vision constants
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2


def compute_block_size(height, width, num_frames=64, temporal_patch_size=2):
    """Estimate tokens per frame after spatial merge."""
    h_patches = height // PATCH_SIZE // SPATIAL_MERGE_SIZE
    w_patches = width // PATCH_SIZE // SPATIAL_MERGE_SIZE
    tokens_per_frame = h_patches * w_patches

    logger.info(
        f"Block size calculation: {height}x{width} -> {h_patches}x{w_patches} "
        f"= {tokens_per_frame} tokens/frame"
    )
    return tokens_per_frame


def estimate_video_resolution(
    num_frames,
    min_pixels=VIDEO_MIN_PIXELS,
    max_pixels=VIDEO_MAX_PIXELS,
    total_pixels=VIDEO_TOTAL_PIXELS,
):
    """Estimate resized frame shape and tokens."""
    # Max pixel budget scales with frame count
    effective_max_pixels = max(
        min(max_pixels, total_pixels / num_frames * FRAME_FACTOR),
        int(min_pixels * 1.05)
    )

    # Assume roughly square frames for estimation
    side = int(math.sqrt(effective_max_pixels))
    side = (side // IMAGE_FACTOR) * IMAGE_FACTOR
    side = max(side, IMAGE_FACTOR)

    tokens_per_frame = compute_block_size(side, side, num_frames)
    return side, side, tokens_per_frame


class Qwen2_5VL_ReKV(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        # ReKV attributes will be initialized via init_rekv() after from_pretrained
        self.processor = None
        self.n_frame_tokens = None
        self.init_prompt_ids = None
        self.n_local = None
        self.topk = None
        self.chunk_size = None
        self.kv_cache = None

        # For Qwen2.5-VL, tokens per frame depends on spatial resolution
        # This will be set dynamically during first chunk encoding
        self._tokens_per_frame = None
        self._spatial_merge_size = 2  # Qwen2.5-VL uses 2x2 spatial merge

    def init_rekv(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        """Initialize ReKV state after loading."""
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    def get_prompt(self, query, mc=False):
        # Close user turn and start assistant turn
        prompt = f"{query}<|im_end|>\n<|im_start|>assistant\n"
        if mc:
            prompt += "Best option: ("
        return prompt

    def _compute_tokens_per_frame(self, video_grid_thw):
        """Return tokens per frame from merged grid size."""
        t, h, w = video_grid_thw[0].tolist()
        logger.debug(f"video_grid_thw: t={t}, h={h}, w={w}")
        tokens_per_frame = (h // self._spatial_merge_size) * (w // self._spatial_merge_size)
        logger.debug(f"tokens_per_frame: {tokens_per_frame}")
        return tokens_per_frame

    def _get_video_features(self, pixel_values_videos, video_grid_thw):
        """Extract video embeddings for the language model."""
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds.unsqueeze(0)  # (N_tokens, D) -> (1, N_tokens, D)
    
    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
    
    def _encode_video_chunk(self, video_chunk):
        """Encode a chunk of frames into the KV cache."""
        inputs = self.processor(
            text=[VIDEO_PLACEHOLDER],
            videos=video_chunk,
            return_tensors="pt",
            padding=True,
        )

        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]

        pixel_dtype = self.visual.get_dtype() if hasattr(self.visual, "get_dtype") else self.dtype
        pixel_values_videos = pixel_values_videos.to(self.device, pixel_dtype)
        video_grid_thw = video_grid_thw.to(self.device)

        tokens_per_frame = self._compute_tokens_per_frame(video_grid_thw)
        if self._tokens_per_frame is None:
            self._tokens_per_frame = tokens_per_frame
            logger.info(f"Qwen2.5-VL tokens per frame: {tokens_per_frame}")
        elif self._tokens_per_frame != tokens_per_frame:
            logger.warning(
                f"Tokens per frame changed: {self._tokens_per_frame} -> {tokens_per_frame}. "
                "This may affect ReKV block alignment."
            )

        video_features = self._get_video_features(pixel_values_videos, video_grid_thw)

        num_frames = video_chunk.shape[0]
        logger.debug(f"video chunk frames: {num_frames}, tokens per frame: {tokens_per_frame}")
        expected_tokens = num_frames * tokens_per_frame
        actual_tokens = video_features.shape[1]
        logger.debug(f"Video chunk: {num_frames} frames, {actual_tokens} tokens (expected: {expected_tokens})")

        assert self.n_local >= video_features.shape[1], \
            f'n_local ({self.n_local}) must be >= video tokens ({video_features.shape[1]})'

        output = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True
        )
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=None):
        """Encode video into KV cache chunk by chunk."""
        num_frames = video.shape[0]

        # Auto-fit chunk size to n_local when unspecified
        if encode_chunk_size is None:
            h, w = video.shape[2], video.shape[3]
            h_patches = h // PATCH_SIZE // SPATIAL_MERGE_SIZE
            w_patches = w // PATCH_SIZE // SPATIAL_MERGE_SIZE
            est_tokens_per_frame = h_patches * w_patches
            logger.debug(f"Frame shape: {h}x{w}, patches: {h_patches}x{w_patches}, est_tokens_per_frame: {est_tokens_per_frame}")

            encode_chunk_size = max(1, (self.n_local - 100) // est_tokens_per_frame)
            logger.info(
                f'Auto encode_chunk_size={encode_chunk_size} '
                f'(est {est_tokens_per_frame} tokens/frame, n_local={self.n_local})'
            )

        num_chunks = num_frames // encode_chunk_size
        logger.debug(f"num_chunks: {num_chunks}")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'Chunk {chunk_idx + 1}/{num_chunks}: KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            remaining_video = video[start_idx:]
            self._encode_video_chunk(remaining_video)

        logger.debug(f'Total KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # === Reprod. Summary ===
        tokens_per_frame = self._tokens_per_frame or "N/A"
        h, w = video.shape[2], video.shape[3]

        # Check frame alignment (per CLAUDE.md: atomic unit is FRAME)
        frame_aligned = (
            isinstance(tokens_per_frame, int) and
            self.n_frame_tokens > 1 and
            tokens_per_frame % self.n_frame_tokens == 0
        )
        alignment_status = "FRAME-ALIGNED" if frame_aligned else "TOKEN-LEVEL"

        logger.info(
            f"\n{'='*60}\n"
            f"Qwen2.5-VL ReKV encode_video summary\n"
            f"  Sampling         : {num_frames} frames (chunk_size={encode_chunk_size})\n"
            f"  Frame shape      : {tuple(video.shape)} (T, C, H, W)\n"
            f"  Tokens/frame     : {tokens_per_frame}\n"
            f"  Block size       : {self.n_frame_tokens} ({alignment_status})\n"
            f"  Total vis tok    : {tokens_per_frame} * {num_frames} = {tokens_per_frame * num_frames if isinstance(tokens_per_frame, int) else 'N/A'}\n"
            f"  n_local          : {self.n_local}\n"
            f"  topk             : {self.topk}\n"
            f"  Budget (local)   : {self.n_local} tokens\n"
            f"{'='*60}"
        )

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        """Answer a question with retrieval-augmented KV cache."""
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []

        # Only the question is used for retrieval
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        for layer_kv in self.kv_cache:
            layer_kv.set_retrieval()

        output = self.language_model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=self.kv_cache.copy()
        )
        past_key_values = output.past_key_values

        for layer_kv in self.kv_cache:
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            else:  # decoding
                outputs = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = self.lm_head(hidden_states[0, -1, :])
            token = torch.argmax(logits, dim=-1).item()

            output_ids.append(token)

            if token in stop_token_ids:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        # === Reprod. Budget Summary ===
        final_kv_len = past_key_values[0][0].shape[2]
        logger.info(
            f"Qwen2.5-VL ReKV QA budget: "
            f"n_local={self.n_local}, topk={self.topk}, "
            f"tokens/frame={self._tokens_per_frame}, "
            f"final_kv={final_kv_len}"
        )

        return output

    def prepare_video_tensor(
        self,
        video_path,
        fps=None,
        min_frames=FPS_MIN_FRAMES,
        max_frames=FPS_MAX_FRAMES,
        video_start=None,
        video_end=None,
    ):
        """Load video with dynamic FPS sampling."""
        ele = {
            "video": video_path,
            "fps": FPS if fps is None else fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
        }
        if video_start is not None:
            ele["video_start"] = video_start
        if video_end is not None:
            ele["video_end"] = video_end

        video_chunk_frames = max_frames if max_frames is not None else -1

        video = fetch_video(
            ele,
            image_factor=IMAGE_FACTOR,
            return_video_sample_fps=False,
            video_chunk_frames=video_chunk_frames,
        )
        return video

    def calc_memory_usage(self):
        """Calculate CPU memory usage of KV cache."""
        if self.kv_cache is None:
            return 0
        n_layers = len(self.kv_cache)
        return n_layers * self.kv_cache[0].calculate_cpu_memory()

    def clear_cache(self):
        """Clear KV cache and reset dynamic state."""
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self._tokens_per_frame = None

def load_model(model_path='Qwen/Qwen2.5-VL-7B-Instruct',
               n_init=None, n_local=None, topk=64, chunk_size=1,
               block_size=None, expected_frames=64):
    """Load Qwen2.5-VL with ReKV attention patching."""
    device = 'cuda'

    # Use block_size=1 to avoid temporal alignment issues.
    if block_size is None:
        block_size = 1
        logger.info(f"Using block_size={block_size} for Qwen2.5-VL (temporal merge compatible)")

    n_frame_tokens = block_size

    processor = AutoProcessor.from_pretrained(model_path)
    init_prompt = SYSTEM_PROMPT_TEMPLATE
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    logger.debug(f"init_prompt_ids shape: {init_prompt_ids.shape}")

    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': block_size,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': block_size,
        'pin_memory': True,
    }

    model = Qwen2_5VL_ReKV.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    model.init_rekv(
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )

    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    model.eval()

    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'block_size (n_frame_tokens): {block_size}')

    return model, processor
