import copy
import torch

from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention

def rekv_attention_forward(
    n_local, n_init, retrieve_size, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)

    def forward(
        self, 
        query : torch.Tensor, 
        key_value : torch.Tensor,
        position_embeddings,
        use_cache, past_key_values,
        project_q, project_k, project_v, attention_out, 
        dim_head, num_heads, num_heads_kv,
    ):
        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads_kv * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads_kv * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        
        if position_embeddings._cos_cached is not None and position_embeddings._cos_cached.device != h_q.device:
            position_embeddings = copy.deepcopy(position_embeddings)
            if position_embeddings.inv_freq.device != h_q.device:
                position_embeddings.inv_freq = position_embeddings.inv_freq.to(h_q.device)
            if position_embeddings._cos_cached is not None:
                position_embeddings._cos_cached = position_embeddings._cos_cached.to(h_q.device)
            if position_embeddings._sin_cached is not None:
                position_embeddings._sin_cached = position_embeddings._sin_cached.to(h_q.device)
        
        if past_key_values[self.layer_idx] is None:
            past_key_values[self.layer_idx] = ContextManager(
                position_embeddings,
                n_init, n_local, 
                block_size, max_cached_block, 
                retrieve_size, chunk_size, exc_block_size,
                fattn, async_global_stream, pin_memory,
            )

        past_key_value = past_key_values[self.layer_idx]

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:       
            # Retrieval mode 
            if type(past_key_value) is ContextManager:  
                past_k, past_v = past_key_value.get_retrieved_kv(global_q) # Internal retrieval
                updata_kv_cache = False # We do not update KV cache with the input KV (h_k, h_v) because we only use it for retrieval
            # Sliding-window attention (Generation)
            else:  
                # past_key_value (i-th layer): Tuple (past_k, past_v)
                past_k, past_v = past_key_value
                updata_kv_cache = True

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            """ 3. Update KV cache """
            if updata_kv_cache:
                if len_k <= n_local + n_init:
                    h_k_cache = h_k
                    h_v_cache = h_v
                    
                else: # len_k > n_local + n_init
                    # Keep the last n_local tokens + initial tokens
                    # KV cache size: n_init + n_local
                    split_idx = len_k - n_local
                    h_k_cache = torch.cat([h_k[:, :, :n_init, :], h_k[:, :, split_idx:, :]], dim=2)
                    h_v_cache = torch.cat([h_v[:, :, :n_init, :], h_v[:, :, split_idx:, :]], dim=2)
                  
                past_key_values[self.layer_idx] = (h_k_cache, h_v_cache)
            else:
                past_key_values[self.layer_idx] = (past_k, past_v)
                
            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + n_local < len_k:
                # Keep the last n_local tokens + current query tokens 
                # KV size: len_q + n_local
                split_idx = len_k - (len_q + n_local)
                h_k_ = h_k_[:, :, split_idx:, :]
                h_v_ = h_v_[:, :, split_idx:, :]
        
            # TODO: Reassign RoPE (treat as consecutive tokens)
            local_h_q, local_h_k = position_embeddings(h_q_, h_k_)
            local_h_v = h_v_

            """ 5. Get init QKV and apply RoPE to init Q 
            (Infinite-LM assigns the same position_ids to initial tokens) """
            if len_k > n_local:
                init_h_q = position_embeddings.apply_rotary_pos_emb_one_angle(
                    h_q, n_local
                )
                init_h_k = h_k
                init_h_v = h_v
                init_h_k = init_h_k[:, :, :n_init, :].contiguous()
                init_h_v = init_h_v[:, :, :n_init, :].contiguous()

            else:
                init_h_q = h_q
                init_h_k = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_k.device,
                    dtype=h_k.dtype
                )
                init_h_v = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_v.device,
                    dtype=h_v.dtype
                )

            """ 6. Sliding Window Attention """
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, n_local), complement_sliding_window=True)
            o, _ = attn.get_result() 
            
        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            
        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, num_heads * dim_head)
        o = attention_out(o)
        return o

    return forward
