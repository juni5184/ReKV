"""
Multimodal Rotary Position Embedding (M-RoPE) for Qwen2.5-VL.

M-RoPE extends standard RoPE to handle multimodal (vision + text) sequences by using
3D position indices: temporal, height, and width. For text tokens, all three dimensions
share the same position value. For vision tokens (images/videos), each dimension encodes
different spatial/temporal positions.

Reference: https://qwenlm.github.io/blog/qwen2-vl/
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Multimodal Rotary Position Embedding to query and key tensors.

    For Qwen2.5-VL, the head dimension is split into 3 sections for temporal, height,
    and width position embeddings. Each section uses its corresponding position index.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine embedding of shape (3, batch, seq_len, head_dim)
        sin: Sine embedding of shape (3, batch, seq_len, head_dim)
        mrope_section: List of section sizes for [temporal, height, width] rope.
            The sum should equal head_dim // 2.
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.

    Returns:
        Tuple of rotated (query, key) tensors.
    """
    # Double the section sizes since we're working with interleaved cos/sin
    mrope_section = mrope_section * 2

    # Split and recombine cos/sin using section-wise assignment
    # Each section i uses position dimension (i % 3)
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
        dim=-1
    ).unsqueeze(unsqueeze_dim)

    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MultiModalRotaryEmbedding(nn.Module):
    """Multimodal Rotary Position Embedding for Qwen2.5-VL.

    This module computes position embeddings for multimodal sequences where
    position IDs have 3 dimensions: (3, batch_size, seq_len) representing
    temporal, height, and width positions.

    For text tokens, all 3 dimensions typically have the same position value.
    For vision tokens, each dimension encodes the corresponding spatial/temporal position.

    Args:
        dim: Dimension of the rotary embedding (typically head_dim).
        base: Base for the inverse frequency computation. Default: 10000.
        mrope_section: List of section sizes [temporal, height, width] for splitting
            the head dimension. Sum should equal dim // 2.
        attention_scaling: Scaling factor applied to cos/sin. Default: 1.0.
    """

    def __init__(
        self,
        dim: int,
        base: Union[int, float] = 10000,
        mrope_section: Optional[List[int]] = None,
        attention_scaling: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.attention_scaling = attention_scaling

        # Default section: split across 3 dimensions, handling remainder
        if mrope_section is None:
            half_dim = dim // 2
            base_size = half_dim // 3
            remainder = half_dim % 3
            # Distribute remainder to later sections
            self.mrope_section = [
                base_size,
                base_size + (1 if remainder >= 2 else 0),
                base_size + (1 if remainder >= 1 else 0),
            ]
        else:
            self.mrope_section = list(mrope_section)

        # Validate section sizes
        expected_size = dim // 2
        actual_size = sum(self.mrope_section)
        assert actual_size == expected_size, (
            f"mrope_section sum ({actual_size}) must equal dim // 2 ({expected_size})"
        )

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Caching for efficiency
        self._seq_len_cached = -1
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _compute_cos_sin(
        self,
        position_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin embeddings for given position IDs.

        Args:
            position_ids: Position indices of shape (3, batch_size, seq_len)
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (cos, sin) tensors of shape (3, batch_size, seq_len, dim)
        """
        # Expand inv_freq for 3D position computation: (3, batch, dim//2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float()
        inv_freq_expanded = inv_freq_expanded.expand(3, position_ids.shape[1], -1, 1)

        # Expand position_ids: (3, batch, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # Compute frequencies via matrix multiplication
        # (3, batch, dim//2, 1) @ (3, batch, 1, seq_len) -> (3, batch, dim//2, seq_len)
        # Transpose to (3, batch, seq_len, dim//2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

        # Duplicate for interleaved format: (3, batch, seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings for the input tensor.

        Args:
            x: Input tensor (used only for device/dtype reference)
            position_ids: 3D position indices of shape (3, batch_size, seq_len)

        Returns:
            Tuple of (cos, sin) tensors of shape (3, batch_size, seq_len, dim)
        """
        return self._compute_cos_sin(position_ids, x.device, x.dtype)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)
            cos: Cosine embedding from forward()
            sin: Sine embedding from forward()
            unsqueeze_dim: Dimension for broadcasting

        Returns:
            Tuple of rotated (query, key) tensors
        """
        return apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, self.mrope_section, unsqueeze_dim
        )

    @torch.no_grad()
    def get_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached cos/sin for sequential 1D positions (text-only mode).

        For text-only sequences, all 3 position dimensions are identical.

        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (cos, sin) for positions [0, seq_len)
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Create 1D positions and replicate across 3 dimensions
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

            self._cos_cached, self._sin_cached = self._compute_cos_sin(
                position_ids, device, dtype
            )

        return self._cos_cached, self._sin_cached

    def apply_rotary_one_pos(
        self,
        x: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """Apply rotary embedding at a single position (for decoding).

        Args:
            x: Input tensor
            position: Single position index (same for all 3 dimensions)

        Returns:
            Rotated tensor
        """
        # Get cached cos/sin up to this position
        cos, sin = self.get_cos_sin_cache(position + 1, x.device, x.dtype)

        # Extract single position
        if cos.dim() == 4:
            cos = cos[:, :, position:position+1, :]
            sin = sin[:, :, position:position+1, :]

        # For single position, apply simplified rotation
        # Combine sections for 1D position (all dimensions same)
        mrope_section_doubled = self.mrope_section * 2
        cos_combined = torch.cat(
            [m[0] for m in cos.split(mrope_section_doubled, dim=-1)],
            dim=-1
        )
        sin_combined = torch.cat(
            [m[0] for m in sin.split(mrope_section_doubled, dim=-1)],
            dim=-1
        )

        return (x * cos_combined) + (rotate_half(x) * sin_combined)


def create_multimodal_rope_from_config(config) -> MultiModalRotaryEmbedding:
    """Create MultiModalRotaryEmbedding from a Qwen2.5-VL config.

    Args:
        config: Model configuration with rope_scaling and head_dim info

    Returns:
        Configured MultiModalRotaryEmbedding instance
    """
    head_dim = config.hidden_size // config.num_attention_heads

    # Get rope parameters from config
    base = getattr(config, "rope_theta", 10000)

    mrope_section = None
    attention_scaling = 1.0

    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if "mrope_section" in config.rope_scaling:
            mrope_section = config.rope_scaling["mrope_section"]
        # Some configs have attention_scaling factor
        if "factor" in config.rope_scaling:
            attention_scaling = 1.0 / config.rope_scaling["factor"]

    return MultiModalRotaryEmbedding(
        dim=head_dim,
        base=base,
        mrope_section=mrope_section,
        attention_scaling=attention_scaling,
    )
