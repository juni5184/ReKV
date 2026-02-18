import torch

from model.attention import (
    RotaryEmbeddingESM, 
    rekv_attention_forward,
)

def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache: bool = False,
        **kwargs,
    ):
        attn_out = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_values, 
            self.q_proj, self.k_proj, self.v_proj, self.o_proj, 
            self.head_dim, self.config.num_attention_heads, self.config.num_key_value_heads
        )
        
        return attn_out, None 

    return hf_forward


def patch_hf(
    model,
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    
    # This approach lacks scalability and will be refactored.
    from transformers import (  
        Qwen2ForCausalLM, Qwen2Model, 
        Qwen2_5_VLTextModel, Qwen3VLTextModel
    )
    from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

    # Language model forward 
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = [None for _ in range(len(self.layers))]
            
        hidden_states = inputs_embeds
        
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]                

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        
    # Extract HF RoPE config
    hf_rope = model.rotary_emb 
    base = hf_rope.config.rope_theta 
    distance_scale = 1.0 
    partial_rotary_factor = getattr(hf_rope.config, "partial_rotary_factor", 1.0)
    head_dim = getattr(hf_rope.config, "head_dim", hf_rope.config.hidden_size // hf_rope.config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    # LLava-OneVision
    if isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen2Model):
        rope = RotaryEmbeddingESM(
            dim=dim,
            base=base,
            distance_scale=distance_scale
        )
        forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))
        
    # Qwen2.5-VL
    # elif isinstance(model, Qwen2_5_VLTextModel) or isinstance(model, Qwen3VLTextModel):   
    #     rope = MultimodalRotaryEmbedding(
    #         dim=dim,
    #         base=base,
    #         distance_scale=distance_scale
    #     )
    #     forward = huggingface_forward(qwen_vl_rekv_attention_forward(**attn_kwargs))
        
    else:
        raise ValueError(f"DD NOT support {model.__class__.__name__}.")

    # Patch rotary embedding
    model.position_bias = rope

    # Patch model forward and attention forward
    Model = model.__class__
    Attention = model.layers[0].self_attn.__class__

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model._old_forward = model.forward
    model.forward = model_forward.__get__(model, Model)
    return model