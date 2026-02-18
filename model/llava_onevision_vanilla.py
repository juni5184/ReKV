import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger


class LlavaOneVision_Vanilla(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)

    def init_vanilla(self, processor, n_frame_tokens=196):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens

        init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        self.init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(self.device)
        self.video_features = None

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|>\n<|im_start|>assistant\n"
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
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature

        video_features = self.multi_modal_projector(selected_video_feature)
        video_features = self.model.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    def clear_cache(self):
        self.video_features = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):
        num_frames = video.shape[0]
        all_features = []
        for chunk_st in range(0, num_frames, encode_chunk_size):
            chunk_ed = chunk_st + encode_chunk_size
            chunk_video = video[chunk_st:chunk_ed]
            pixel_values_videos = self.processor.video_processor(chunk_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
            video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
            all_features.append(video_features)
            logger.debug(f'Encoded frames {chunk_st}-{min(chunk_ed, num_frames)}, features shape: {video_features.shape}')
        self.video_features = torch.cat(all_features, dim=1)  # (1, total_frames*196, D)
        logger.debug(f'Total video features shape: {self.video_features.shape}')

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, **kwargs):
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []

        # Build full input embeddings: [init_prompt | video_features | prompt]
        init_embeds = self.get_input_embeddings()(self.init_prompt_ids)  # (1, N_init, D)

        prompt_ids = self.processor.tokenizer(input_text['prompt'], return_tensors="pt").input_ids.to(self.device)
        prompt_embeds = self.get_input_embeddings()(prompt_ids)  # (1, N_prompt, D)

        inputs_embeds = torch.cat([init_embeds, self.video_features, prompt_embeds], dim=1)  # (1, N_total, D)

        # Prefill
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states[0, -1, :])
        token = torch.argmax(logits, dim=-1).item()
        output_ids.append(token)

        # Autoregressive decoding
        for i in range(1, max_new_tokens):
            if token in stop_token_ids:
                break

            outputs = self.language_model(
                input_ids=torch.as_tensor([[token]], device=self.device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = self.lm_head(hidden_states[0, -1, :])
            token = torch.argmax(logits, dim=-1).item()
            output_ids.append(token)

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        return output


def load_model(
    model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
    n_local=None, retrieve_size=64, chunk_size=1, sample_fps=None
):
    n_frame_tokens = 196

    processor = LlavaOnevisionProcessor.from_pretrained(model_path)

    model = LlavaOneVision_Vanilla.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    model.init_vanilla(
        processor=processor,
        n_frame_tokens=n_frame_tokens,
    )

    logger.info(f'Vanilla model loaded (no ReKV)')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()
    return model, processor
