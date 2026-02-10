import torch
from logzero import logger
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from transformers.cache_utils import DynamicCache

SYSTEM_PROMPT = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'

class LlavaOneVision_Vanilla(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        self.processor = None
        self.kv_cache = None
        self.system_prompt = SYSTEM_PROMPT
        self.use_sliding_window = False
        self.n_local = 15000

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def set_sliding_window(self, enable: bool, n_local: int = 1024):
        """Enable or disable sliding window attention.

        Args:
            enable: True for sliding window, False for full attention
            n_local: Number of tokens to keep in KV cache when enabled
        """
        self.use_sliding_window = enable
        self.n_local = n_local
        mode = f"sliding window (size={n_local})" if enable else "full attention"
        logger.info(f"Attention mode: {mode}")

    @torch.inference_mode()
    def encode_init_prompt(self):
        system_tokens = self.processor.tokenizer(self.system_prompt, return_tensors="pt")
        input_ids = system_tokens["input_ids"].to(self.device)
        output = self.language_model(input_ids=input_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, num_sampled_frames=64):
        num_frames = video.shape[0]
        if num_frames <= num_sampled_frames:
            video_sampled = video
        else:
            sampling_indices = torch.linspace(0, num_frames - 1, steps=num_sampled_frames).long()
            video_sampled = video[sampling_indices]
        logger.debug(f"Encoding {video_sampled.shape[0]} frames (requested: {num_sampled_frames})")
        
        # Encode video frames
        pixel_values_videos = self.processor.video_processor(video_sampled, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, N, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, N*196, D)
        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _truncate_kv_cache(self, past_key_values):
        """Truncate KV cache to n_local if sliding window is enabled."""
        if not self.use_sliding_window or past_key_values is None:
            return past_key_values
        
        # Truncate the key-values to keep only the last n_local tokens along sequence dim
        truncated_cache = DynamicCache()
        for layer_idx in range(len(past_key_values)):
            k, v = past_key_values[layer_idx]
            truncated_cache.update(
                k[:, :, -self.n_local:, :] if k.shape[2] > self.n_local else k,
                v[:, :, -self.n_local:, :] if v.shape[2] > self.n_local else v,
                layer_idx,
            )
        return truncated_cache

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.model.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        
        out = self.language_model(
            input_ids=input_ids, 
            use_cache=True, 
            past_key_values=self._truncate_kv_cache(self.kv_cache)
        )
        past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(
                    inputs_embeds=inputs_embeds, 
                    use_cache=True, 
                    past_key_values=self._truncate_kv_cache(past_key_values)
                )
                past_key_values = out.past_key_values  # Update past_key_values
                logits = self.lm_head(out.last_hidden_state)
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=self._truncate_kv_cache(past_key_values),
                )
                past_key_values = out.past_key_values  # Update past_key_values
                logits = self.lm_head(out.last_hidden_state)

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='/scratch2/juni5184/model_zoo/llava-onevision-qwen2-7b-ov-hf'):
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)

    model = LlavaOneVision_Vanilla.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
    )
    model.processor = processor
    model.eval()

    return model, processor
