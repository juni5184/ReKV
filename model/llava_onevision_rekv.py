import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger
import math
import torch.nn as nn

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def apply_pooling(self, image_features):
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, seq_len, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / 2), math.ceil(width / 2)]
        image_features = nn.functional.interpolate(image_features, size=scaled_shape, mode="bilinear")

        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(batch_frames, -1, dim)
        return image_features

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

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: 0) Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        # NOTE: 1) Activate retrieval mode for kv_cache
        for layer_kv in self.kv_cache:  # activate retrieval mode
            if layer_kv is not None:
                layer_kv.set_retrieval()

        # NOTE: 2) Internal/External retrieval mode selection
        if retrieved_indices is not None:
            for layer_kv in self.kv_cache:
                if layer_kv is not None:
                    assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                    layer_kv.set_retrieved_block_indices(retrieved_indices)

        out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
        past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        # NOTE: 3) Off retrieval mode
        for layer_kv in self.kv_cache:  # reset to default
            if layer_kv is not None:
                layer_kv.reset_retrieval()

        # # 4) 본문 프롬프트 prefill
        input_ids = self.processor.tokenizer(input_text["prompt"]).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        past_key_values = out.past_key_values

        # 5) 토큰 생성 루프
        for i in range(max_new_tokens):
            hidden_states = out.last_hidden_state
            logits = self.lm_head(hidden_states)  # (B=1, T, V)

            # 마지막 토큰 로짓만
            if logits.dim() == 3:
                last_token_logits = logits[0, -1, :]
            elif logits.dim() == 2:
                last_token_logits = logits[-1, :]
            else:
                last_token_logits = logits  # (V,)

            # greedy (원하면 temperature/top-k/top-p 추가)
            token = int(torch.argmax(last_token_logits))
            output_ids.append(token)

            if stop_token_ids and token in stop_token_ids:
                break

            out = self.language_model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
