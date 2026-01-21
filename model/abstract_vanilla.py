import torch
from logzero import logger


class Abstract_Vanilla:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

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

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass
