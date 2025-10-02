import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from logzero import logger
    
from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV


class Qwen2_5_VL_ReKV(Qwen2_5_VLForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt
    
    @classmethod # override
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args,
        processor=None, n_frame_tokens=None, init_prompt_ids=None,
        n_local=15000, topk=64, chunk_size=2,
        **kwargs
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        Abstract_ReKV.__init__(
            model,
            processor=processor,
            n_frame_tokens=n_frame_tokens,
            init_prompt_ids=init_prompt_ids,
            n_local=n_local,
            topk=topk,
            chunk_size=chunk_size,
        )
        return model

    # transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1159
    def _get_video_features(self, pixel_values_videos, video_grid_thw):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)  # (sum_tokens, C_v)
        split_sizes = (video_grid_thw.prod(-1) // (self.visual.spatial_merge_size ** 2)).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds
        
    def _encode_video_chunk(self, video_chunk):
        frames = video_chunk
        # 1) Tensor -> HWC, uint8
        if isinstance(frames, torch.Tensor):
            if frames.ndim != 4:
                raise ValueError(f"Expected 4D video tensor, got {frames.shape}")

            # (N,3,H,W) -> (N,H,W,3)
            if frames.shape[1] == 3 and frames.shape[-1] != 3:
                frames = frames.permute(0, 2, 3, 1)

            # float → uint8 [0,255]
            if frames.dtype != torch.uint8:
                if torch.is_floating_point(frames) and frames.max() <= 1.0:
                    frames = frames * 255.0
                frames = frames.round().clamp(0, 255).to(torch.uint8)
            frames_np = frames.cpu().numpy()  # (N,H,W,3), uint8
            frames_list = [f for f in frames_np]   # List[np.ndarray(H,W,3)]
        else:
            # If already List[np.ndarray(H,W,3)]
            frames_list = frames

        # 2) extreme aspect ratio guard(optional)
        # 200.0 is the default value in Qwen2.5VL
        safe = []
        for img in frames_list:
            h, w = int(img.shape[0]), int(img.shape[1])
            if h == 0 or w == 0:
                continue
            ar = max(h, w) / max(1, min(h, w))
            if ar < 200.0:
                safe.append(img)
        if not safe:
            raise ValueError("No valid frames after AR check (<200).")

        # 3) Qwen video processor → pixel tensor
        out = self.processor.video_processor(safe, return_tensors="pt")
        pixel_values_videos = out.pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)

        # (추가) grid_thw Extract — BatchFeature has both attributes and key access
        video_grid_thw = (
            getattr(out, "video_grid_thw", None)
            if hasattr(out, "video_grid_thw")
            else out.get("video_grid_thw", None)
        )
        if video_grid_thw is None:
            video_grid_thw = (
                getattr(out, "grid_thw", None)
                if hasattr(out, "grid_thw")
                else out.get("grid_thw", None)
            )
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is missing from processor output.")

        if not isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = torch.as_tensor(video_grid_thw)
        video_grid_thw = video_grid_thw.to(self.device)

        video_features = self._get_video_features(pixel_values_videos, video_grid_thw)  # (1, S, hidden)
        # tuple -> tensor
        if isinstance(video_features, tuple):
            video_features = video_features[0]
        # 2D -> (1, S, H)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(0)

        B, S, H = video_features.shape
        assert self.n_local >= S, f'n_local: {self.n_local}, video_features.shape: {video_features.shape}'
        
        attention_mask = video_features.new_ones((B, S), dtype=torch.long)

        # 2) _encode_video_chunk: block align padding + mask pass
        rem = S % self.n_frame_tokens 
        if rem != 0:
            pad_len = self.n_frame_tokens - rem
            pad_embed = video_features.new_zeros((B, pad_len, H))
            video_features = torch.cat([video_features, pad_embed], dim=1)
            pad_mask = attention_mask.new_zeros((B, pad_len))
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        output = self.language_model(
            inputs_embeds=video_features,          # already LLM hidden dimension
            attention_mask=attention_mask,         # padding is 0, real token is 1
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True
        )
        self.kv_cache = output.past_key_values
        
    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=8):  # video: (Nv, H, W, 3)
        super().encode_video(video, encode_chunk_size)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []

        # 0) Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        # 1) Activate retrieval mode for kv_cache
        for layer_kv in self.kv_cache:  # activate retrieval mode
            if layer_kv is not None:
                layer_kv.set_retrieval()

        # 2) Internal/External retrieval mode selection
        if retrieved_indices is not None:
            for layer_kv in self.kv_cache:
                if layer_kv is not None:
                    assert layer_kv.block_size == self.n_frame_tokens, \
                        f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                    layer_kv.set_retrieved_block_indices(retrieved_indices)

        out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
        past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        # 3) Off retrieval mode
        for layer_kv in self.kv_cache:  # reset to default
            if layer_kv is not None:
                layer_kv.reset_retrieval()

        # 4) Prompt prefill
        input_ids = self.processor.tokenizer(input_text["prompt"]).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        past_key_values = out.past_key_values

        # 5) Token generation loop
        for i in range(max_new_tokens):
            hidden_states = out.last_hidden_state
            logits = self.lm_head(hidden_states)  # (B=1, T, V)

            # Last token logits only
            if logits.dim() == 3:
                last_token_logits = logits[0, -1, :]
            elif logits.dim() == 2:
                last_token_logits = logits[-1, :]
            else:
                last_token_logits = logits  # (V,)

            # greedy (add temperature/top-k/top-p if desired)
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


def load_model(model_path='model_zoo/Qwen2.5-VL-3B-Instruct',
               n_init=None, n_local=15000, topk=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 128 # Qwen2.5VL uses variable value, set 128 for now
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

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
    model = Qwen2_5_VL_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, 
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