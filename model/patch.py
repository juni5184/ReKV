import torch

from model.attention import RotaryEmbeddingESM, rekv_attention_forward

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
    model, # language model
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    from transformers.models.llama.modeling_llama import BaseModelOutputWithPast
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

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
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache and past_key_values is None:
            past_key_values = [None for _ in range(len(self.layers))]
            
        hidden_states = inputs_embeds
        
        if isinstance(model, Qwen2Model):
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=self.position_bias,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )

        elif isinstance(model, Qwen2_5_VLTextModel):
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=self.position_bias,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))

    if isinstance(model, Qwen2Model): # Llava-onevision
        Attention = model.layers[0].self_attn.__class__ # Qwen2Attention
        Model = model.__class__ # Qwen2Model (Qwen2ForCausalLM)
        hf_rope = model.rotary_emb # Qwen2RotaryEmbedding
        
        base = hf_rope.config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
        partial_rotary_factor = hf_rope.config.partial_rotary_factor if hasattr(hf_rope.config, "partial_rotary_factor") else 1.0
        dim = int((hf_rope.config.hidden_size // hf_rope.config.num_attention_heads) * partial_rotary_factor)

        rope = RotaryEmbeddingESM(
            dim=dim,
            base=base,
            distance_scale=distance_scale
        )
        
        model.position_bias = rope

    elif isinstance(model, Qwen2_5_VLTextModel): # Qwen2.5VL
        Attention = model.layers[0].self_attn.__class__ 
        Model = model.__class__
        hf_rope = model.rotary_emb
        
        base = hf_rope.config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
        partial_rotary_factor = hf_rope.config.partial_rotary_factor if hasattr(hf_rope.config, "partial_rotary_factor") else 1.0
        dim = int((hf_rope.config.hidden_size // hf_rope.config.num_attention_heads) * partial_rotary_factor)

        #TODO: Multi-modal rotary embedding
        # multi_modal_rope = MultiModalRotaryEmbedding(
        #     dim=dim,
        #     base=base,
        #     distance_scale=distance_scale
        # )
        rope = RotaryEmbeddingESM(
            dim=dim,
            base=base,
            distance_scale=distance_scale
        )
        
        model.position_bias = rope
    else:
        raise ValueError(f"Only supports Qwen2 models, not {model.__class__.__name__}.")


    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model._old_forward = model.forward
    model.forward = model_forward.__get__(model, Model)

    return model