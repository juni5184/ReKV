import torch
from logzero import logger

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from model.constants import (
    VIDEO_MAXFRAMES,
    VIDEO_PLACEHOLDER,
    FPS_MIN_FRAMES,
)

# Re-export VIDEO_MAXFRAMES for backward compatibility with streaming_vision_process.py
__all__ = ['VIDEO_MAXFRAMES', 'VIDEO_PLACEHOLDER', 'Qwen2_5VL_Vanilla', 'load_model']

# System prompt template - <|im_start|>user will be followed by video
SYSTEM_PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"


class Qwen2_5VL_Vanilla(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        self.kv_cache = None
        self.processor = None
        self._rope_deltas = None  # Track rope deltas for position continuation

    def clear_cache(self):
        self.kv_cache = None
        self._rope_deltas = None
        # Also clear model's internal rope_deltas
        if hasattr(self.model, 'rope_deltas'):
            self.model.rope_deltas = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def get_prompt(self, query, mc=False):
        # The video is already followed by the user message in encode_video
        # So we just need to close user turn and start assistant turn
        prompt = f"{query}<|im_end|>\n<|im_start|>assistant\n"
        if mc:
            prompt += "Best option: ("
        return prompt

    @torch.inference_mode()
    def encode_init_prompt(self):
        """Encode system prompt into KV cache.

        This should be called before encode_video() to initialize the cache
        with the system prompt tokens.
        """
        # Tokenize system prompt
        system_tokens = self.processor.tokenizer(SYSTEM_PROMPT_TEMPLATE, return_tensors="pt")
        input_ids = system_tokens["input_ids"].to(self.device)

        seq_len = input_ids.shape[1]

        # For text-only, use 1D positions (same across all 3 dimensions)
        positions = torch.arange(seq_len, device=self.device)
        position_ids = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

        # Get embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # Encode through language model
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = output.past_key_values
        # Text-only encoding has no rope adjustment
        self._rope_deltas = torch.zeros(1, device=self.device, dtype=torch.long)
        self.model.rope_deltas = self._rope_deltas

    def prepare_video_tensor(
        self,
        video_path,
        fps=None,
        min_frames=FPS_MIN_FRAMES,
        max_frames=VIDEO_MAXFRAMES,
        image_patch_size=14,
        video_start=None,
        video_end=None,
    ):
        """Load and resize video for Qwen2.5-VL with fixed FPS sampling."""
        # Lazy import to avoid circular dependency
        from model.streaming_vision_process import fetch_video, FPS

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
        image_factor = image_patch_size * 2  # 14 * 2 = 28
        video = fetch_video(
            ele,
            image_factor=image_factor,
            return_video_sample_fps=False,
            video_chunk_frames=video_chunk_frames,
        )
        return video

    def _build_video_inputs_embeds(self, input_ids, pixel_values_videos, video_grid_thw):
        """Build input embeddings with video features injected."""
        inputs_embeds = self.get_input_embeddings()(input_ids)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
        video_mask = input_ids == self.config.video_token_id
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[video_mask] = video_embeds
        return inputs_embeds

    @torch.inference_mode()
    def encode_video(self, video, num_sampled_frames=64):
        """Encode video into KV cache for reuse across multiple questions.

        Assumes encode_init_prompt() was called first to encode the system prompt.
        This method computes 3D rope positions for Qwen2.5-VL's mrope and continues
        from the system prompt's KV cache.
        """
        if self.kv_cache is None:
            raise RuntimeError("kv_cache is empty. Call encode_init_prompt() before encode_video().")

        num_frames = video.shape[0]
        logger.debug(f"num_frames: {num_frames}")
        if num_frames <= num_sampled_frames:
            video_sampled = video
        else:
            sampling_indices = torch.linspace(0, num_frames - 1, steps=num_sampled_frames).long()
            video_sampled = video[sampling_indices]
        logger.debug(f"Encoding {video_sampled.shape[0]} frames (requested: {num_sampled_frames})")

        # Process just the video placeholder (system prompt already in cache)
        inputs = self.processor(
            text=[VIDEO_PLACEHOLDER],
            videos=video_sampled,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]

        pixel_dtype = self.visual.get_dtype() if hasattr(self.visual, "get_dtype") else self.dtype
        pixel_values_videos = pixel_values_videos.to(self.device, pixel_dtype)
        video_grid_thw = video_grid_thw.to(self.device)

        # Build embeddings with video features
        inputs_embeds = self._build_video_inputs_embeds(input_ids, pixel_values_videos, video_grid_thw)

        # Get past sequence length from system prompt encoding
        past_seq_len = self.kv_cache[0][0].shape[2]

        # Compute 3D rope position IDs for video
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
        )

        # Offset positions to continue from system prompt
        position_ids = position_ids + past_seq_len

        # Store rope_deltas for position continuation during generation
        self._rope_deltas = rope_deltas
        self.model.rope_deltas = rope_deltas

        # Encode through language model with existing system prompt cache
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = output.past_key_values

    def _compute_text_position_ids(self, seq_len, past_seq_len):
        """Compute 1D position IDs for text, continuing from past sequence.

        For text-only tokens after video encoding, we use 1D positions (same value
        across all 3 dimensions) that continue from the max position in the video.
        """
        # Get the offset from rope_deltas
        if self._rope_deltas is not None:
            # rope_deltas tells us how much to offset for subsequent tokens
            offset = past_seq_len + self._rope_deltas.item()
        else:
            offset = past_seq_len

        # Create 1D positions (same across all 3 rope dimensions)
        positions = torch.arange(offset, offset + seq_len, device=self.device)
        # Expand to 3D: (3, batch_size, seq_len)
        position_ids = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        return position_ids

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        """Answer a question using the cached video KV and a MCQA prompt."""
        if self.kv_cache is None:
            raise RuntimeError("kv_cache is empty. Call encode_video() before question_answering().")

        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []

        # Get current cache length for position continuation
        past_seq_len = self.kv_cache[0][0].shape[2]  # (batch, heads, seq_len, head_dim)

        # NOTE: Only input the question to perform retrieval (ReKV style)
        question_ids = self.processor.tokenizer(input_text["question"]).input_ids
        question_ids = torch.as_tensor([question_ids], device=device)

        # Compute position IDs continuing from video encoding
        question_pos_ids = self._compute_text_position_ids(question_ids.shape[1], past_seq_len)

        out = self.language_model(
            input_ids=question_ids,
            position_ids=question_pos_ids,
            use_cache=True,
            past_key_values=self.kv_cache,
        )
        past_key_values = out.past_key_values
        past_seq_len = past_key_values[0][0].shape[2]

        for i in range(max_new_tokens):
            if i == 0:  # prefill with full prompt
                prompt_ids = self.processor.tokenizer(input_text["prompt"]).input_ids
                prompt_ids = torch.as_tensor([prompt_ids], device=device)

                # Compute position IDs for prompt
                prompt_pos_ids = self._compute_text_position_ids(prompt_ids.shape[1], past_seq_len)

                inputs_embeds = self.get_input_embeddings()(prompt_ids)
                out = self.language_model(
                    inputs_embeds=inputs_embeds,
                    position_ids=prompt_pos_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                past_seq_len = past_key_values[0][0].shape[2]
                logits = self.lm_head(out.last_hidden_state)
            else:  # decoding one token at a time
                # Position for single token
                token_pos_ids = self._compute_text_position_ids(1, past_seq_len)

                out = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    position_ids=token_pos_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                past_seq_len = past_key_values[0][0].shape[2]
                logits = self.lm_head(out.last_hidden_state)

            last_token_logits = logits[0, -1, :]
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                break

            if i == max_new_tokens - 1:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return output


def load_model(model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    device = "cuda"
    processor = AutoProcessor.from_pretrained(model_path)

    model = Qwen2_5VL_Vanilla.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    model.processor = processor
    model.eval()

    return model, processor
