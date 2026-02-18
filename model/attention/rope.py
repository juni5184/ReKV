import torch
from typing import Union, Tuple

class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self, 
        dim: int, 
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None
        
    def _update_cos_sin_tables(self, seq_len, device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, length, right, cos, sin): 
        cos = cos[:, :, right-length:right, :]
        sin = sin[:, :, right-length:right, :]
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(x.dtype)

    def apply_rotary_pos_emb_one_angle(self, x, index=15000): 
        assert x.dim() == 4
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(index, x.device)
        return self.apply_rotary_pos_emb(x, 1, index, self._cos_cached, self._sin_cached)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim=-2) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q.dim() == 4
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k.size(seq_dim), k.device)
        return (
            self.apply_rotary_pos_emb(q, q.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
            self.apply_rotary_pos_emb(k, k.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
        )
