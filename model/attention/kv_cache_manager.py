import torch
from typing import Optional, Tuple

from .dot_production_attention import get_multi_stage_dot_production_attention

# -------------------------------------------------------------------------------
# CudaCache: A fixed-size block of GPU memory designed for rapid KV-Cache storage
#            Used for local window caching to efficiently store and reuse memory.
# -------------------------------------------------------------------------------
class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        """
        Args:
            num_units (int): Number of independent cache blocks (usually set to batch_size or n_block).
            unit_size (int): Size of each block (e.g., block_size * hidden_dim * 2: 2 for K and V).
            dtype (torch.dtype): Data type for storage.
        """
        self.num_units = num_units  # Number of available blocks.
        self.unit_size = unit_size  # Size of each cache block.
        self.dtype = dtype
        # Pre-allocate a large tensor to act as contiguous memory for all blocks.
        self.data = torch.empty(
            (num_units, unit_size),
            device = "cuda",
            dtype=dtype
        )
        # Track which blocks are available for allocation.
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        """
        Allocate a cache block from the available pool.
        Returns:
            (torch.Tensor, int): The cache tensor (flat view) and its block index.
        """
        assert len(self.idle_set) > 0, "No idle cache blocks available"
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        """
        Mark a block as free/available again after use.
        Args:
            idx (int): Index of cache block to free.
        """
        assert idx not in self.idle_set, "Block already free"
        self.idle_set.add(idx)


# -------------------------------------------------------------------------------
# MemoryUnit: Represents a KV-Cache block that can move data between CPU and GPU.
#             Handles pinning memory, asynchronous transfers, and block reuse.
# -------------------------------------------------------------------------------
class MemoryUnit:
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        """
        Args:
            kv (Tuple[Tensor, Tensor]): Tuple of (K, V) tensors to manage.
            cache (CudaCache): Shared cache manager for block allocations.
            load_to_cache (bool): If True, immediately stage this data on GPU.
            pin_memory (bool): If True, pin the CPU-side memory for faster transfer.
        """
        self.cache = cache

        # Move input K/V tensors to CPU (non-blocking) if they are on GPU.
        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        # Optional: Pin the data to improve CPU-GPU transfer speeds.
        if pin_memory:
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        # Optionally preload this block onto GPU.
        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            # Reshape for usage: leading 2 for K/V.
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            # Book a CUDA event to track transfer completion.
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        # Store persistent pointers and memory state.
        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        """
        Load the KV block onto the GPU if not resident.
        Optionally copy out to an explicit target location.

        Args:
            target (Optional[Tuple[Tensor]]): Tuple of (K, V) GPU tensors to copy into.

        Returns:
            tuple: (allocated_new_buffer: bool, copy_event: Optional[torch.cuda.Event])
        """
        if self.gpu_data is not None:
            # Already on GPU: just optionally copy to target.
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None
            return False, target_event

        # Allocate GPU memory for new cache block
        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        if target is not None:
            # Copy CPU data to both cache and provided target
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)
        else:
            # Only copy from CPU to cache buffer
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        # Track completion of copy
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event if target is not None else None

    def get(self):
        """
        Retrieve the cached KV tensors on GPU, waiting for transfer if still pending.
        
        Returns:
            torch.Tensor: The (2, ...) GPU view of the KV data.
        """
        assert self.gpu_data is not None, "Data not staged on GPU"
        self.event.wait()
        return self.gpu_data

    def offload(self):
        """
        Offload (delete) the GPU cache for this block. Only affects GPU side—CPU copy is kept.
        """
        assert self.gpu_data is not None, "No GPU data to offload"
        self.event.wait()  # Ensure any on-going transfers finish.
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        """
        Estimate the amount of CPU memory used by the block (in bytes).
        """
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()


# -------------------------------------------------------------------------------
# VectorTensor: Dynamic GPU-based cache for storing representative feature vectors.
#               Used for block-level "summary" features (e.g., for similarity search).
# -------------------------------------------------------------------------------
class VectorTensor:
    def __init__(
        self, 
        hidden_size,
        element_dtype,
        device
    ):
        """
        Args:
            hidden_size (int): Dimensionality of each stored vector.
            element_dtype (torch.dtype): Data type for representation.
            device (torch.device/str): Target device (typically 'cuda').
        """
        init_cached_size = 16  # Start with room for 16 representations
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0        # Current number of valid vectors
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    def append_cache(self):
        """
        Double the size of the underlying GPU buffer to permit appending more vectors.
        (This is an O(N) operation on cache size.)
        """
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        """
        Append a batch of new vectors to the cache, expanding if necessary.

        Args:
            tensor (Tensor): Tensor of shape (batch, hidden_size), contiguous.
        """
        assert tensor.dtype == self.data.dtype, "Mismatched dtypes"
        assert tensor.size(1) == self.hidden_size, f'{tensor.size(1)} vs {self.hidden_size}'
        assert tensor.is_contiguous(), "Tensor must be contiguous"
        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)
        self.length += append_l

    def get_data(self):
        """
        Get all currently cached vectors as a tensor ([:self.length, :]).
        Returns:
            Tensor: All vectors as a contiguous tensor.
        """
        return self.data[:self.length, ...]

    def get_cosine_similarity(self, tensor: torch.Tensor):
        """
        Compute similarity between a query vector and all stored keys.

        Args:
            tensor (Tensor): Shape (hidden_size,)

        Returns:
            Tensor: Similarity logits (length,)
        """
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        # Convert to float32 for stable dot product calculation
        key = self.data[:self.length].float() # (T, D)
        query = tensor[None, :].float()  # (1, D)
        logits = torch.matmul(query, key.T)[0]  # (T,)
        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def __len__(self):
        return self.length


# ------------------------------------------------------------------------------
# GLOBAL_STREAM: Dedicated CUDA stream for managing global asynchronous KV ops.
# Used for non-blocking data staging (concurrent with main attention compute).
# ------------------------------------------------------------------------------
GLOBAL_STREAM = None


# ------------------------------------------------------------------------------
# ContextManager: The overarching memory and attention manager—coordinates local
#                 and global KV cache, block offloading/staging, and block retrieval.
# ------------------------------------------------------------------------------
class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, chunk_size, exc_block_size, 
                 fattn: bool = False,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
    ):
        """
        Main controller for multi-stage attention context memory. Handles block streaming,
        cache management, and chunking.

        Args:
            position_embedding: Position embedding (e.g., rotary) handler.
            n_init (int): Number of tokens in initial context window.
            n_local (int): Sliding window length (tokens always locally on GPU).
            block_size (int): How many tokens per global context block.
            max_cached_block (int): Max number of global context blocks to keep on GPU per batch.
            topk (int): Number of blocks to retrieve by similarity.
            chunk_size (int): Number of blocks in each retrieval chunk.
            exc_block_size (int): How many input tokens to process at a time.
            fattn (bool): Fast attention trigger.
            async_global_stream (bool): Enable async global KV staging.
            pin_memory (bool): Whether to pin CPU memory for fast transfer.
        """
        self.length = 0  # Total number of tokens processed/held
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local # Sanity: there should be enough local capacity.
        self.topk = topk
        self.chunk_size = chunk_size
        # Get attention operator factory
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        
        # Acquire/capture a global CUDA stream if doing async global work
        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        # Set up state for context block retrieval (retrieved indices, etc.)
        self.reset_retrieval()

    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks = None):
        """
        Remove least-recently-used blocks (by access time or usage count).
        Args:
            u (int): Batch index/unit.
            num_remove (Optional[int]): Number of blocks to remove. If None, remove to enforce max_cached_block.
            ignore_blocks (Optional[Set[int]]): Blocks to protect (do not remove).
        """
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        # Sort by last-access metric (LRU).
        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    def _from_group_kv(self, tensor):
        """
        For head-grouped queries, expand grouped keys/values into standard head format.

        Args:
            tensor (Tensor): (batch_size, n_head_kv, length, dim_head)
        Returns:
            Tensor: (batch_size, n_head, length, dim_head)
        """
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv

        # Reshape and expand, then flatten heads
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))
        tensor = tensor.expand((self.num_units, self.unit_size_kv, num_group, length, dim_head)).reshape((self.num_units, self.num_heads, length, dim_head))
        return tensor
    
    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        """
        Initialize context manager state using tensor metadata only.
        Checks and sets expected sizes, dtypes, and device mapping.
        """
        assert local_q.dim() == 4
        
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        # Validate all provided Q/K/V tensors are correctly shaped and contiguous on GPU
        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        # Key dimension bookkeeping
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head

        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        # List of blocks for each unit (batch) representing global context memory.
        self.global_blocks = [[] for _ in range(self.num_units)] # Shape: [batch_size][memory_unit]
        # LRU tracking: Dicts per unit for block-id:score (access recency, count, etc.)
        self.cached_blocks = [{} for _ in range(self.num_units)] # Shape: [batch_size]{block_id:score}
        self.num_global_block = 0

        # For similarity retrieval: block_k stores agg. feature vectors for each block.
        self.block_k = [
            VectorTensor(
                dim_head * self.unit_size, global_k.dtype, global_k.device
            ) for _ in range(self.num_units)
        ] # Per batch, store each block's (num_heads * dim_head) mean for similarity

        # Sliding window local KV-cache (always stays on GPU).
        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device) 
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        # Staging area for global tokens awaiting processing into memory blocks (unblocked context).
        # Tuple of (K, V) with zero length.
        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        # 'init_k'/'init_v' stores the initial context tokens before sliding-window fills and offloading starts.
        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # Allocate a buffer for concatenating K/V chunks during compute.
        # Shape: (2, batch_size, n_head_kv, buffer_len, dim_head)
        # buffer_len is upper bound for all immediately-needed tokens.
        buffer_len = self.topk * self.block_size + self.n_init
        self.global_buffer = torch.zeros(
            (2, self.num_units, self.unit_size_kv, buffer_len, dim_head),
            dtype = global_k.dtype, device=global_k.device
        )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        # Underlying memory manager for keeping a fixed number of global blocks staged.
        self.cuda_cache = CudaCache(
            num_units=self.max_cached_block * self.num_units,
            unit_size=self.block_size * self.unit_size_kv * dim_head * 2, # "*2" for K, V per block
            dtype=local_k.dtype
        ) 

        self.initialized = True

    def set_retrieval(self):
        """
        Mark that block retrieval should occur on next compute pass.
        """
        self.to_retrieve = True

    def reset_retrieval(self):
        """
        Reset retrieval state (clear similarity matrix and block indices).
        """
        self.similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def set_retrieved_block_indices(self, retrieved_block_indices):
        """
        Set which blocks to retrieve for the next pass.
        Args:
            retrieved_block_indices (list or tensor): batch_size x n_frames
        """
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices

    def get_retrieved_kv(self, query=None):
        """
        Retrieve (and copy to GPU buffer) the context KV blocks indexed by self.retrieved_block_indices.
        Optionally, if a query is passed, calculate similarity and set indices.

        Args:
            query (Tensor, optional): (batch_size, num_heads, length, dim_head): Used for similarity search.

        Returns:
            global_h_k, global_h_v: Blocked/collated K and V matrices on the global buffer (ready for attention).
        """
        if query is not None:  # Do adaptive retrieval based on query-key similarity
            block_topk = self._calc_block_topk(query)
            self.set_retrieved_block_indices(block_topk)

        assert len(self.retrieved_block_indices) == self.num_units

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        with torch.cuda.stream(GLOBAL_STREAM):
            if self.init_exc:  # After sliding window filled, blocks are staged/offloaded as MemoryUnits
                # Ensure there is room for newly loaded blocks: remove LRU if needed (except needed ones)
                for u in range(self.num_units):
                    num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                    for b_idx in self.retrieved_block_indices[u]:
                        if b_idx not in self.cached_blocks[u]:
                            num_remove += 1
                    self._remove_lru_blocks(u, num_remove, ignore_blocks=self.retrieved_block_indices[u])

                # Mark access count (for LRU), update load_count
                self.load_count += 1
                for u in range(self.num_units):
                    for b_idx in self.retrieved_block_indices[u]:
                        self.cached_blocks[u][b_idx] = self.load_count
                
                # Initial window (init KV) was already loaded at buffer[:n_init]; now load N retrieved blocks.
                init_st = 0
                init_ed = init_st + self.init_k.size(-2)
                ed = init_ed
                assert self.global_buffer_init_st == init_st or self.global_buffer_init_ed == init_ed

                # Loop over batches, and per-batch, over block indices to load requested blocks into buffer.
                for u in range(self.num_units):
                    assert self.retrieved_block_indices[u][-1] < self.num_global_block, f'{self.retrieved_block_indices[u][-1]}, {self.num_global_block}'
                    for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                        st = init_ed + cnt * self.block_size
                        ed = st + self.block_size
                        # The .load method of MemoryUnit will ensure that if block is offloaded, it's re-staged on GPU.
                        self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))

            else:  # All context is still contained in self.global_remainder (before any offload occurs).
                # Copy initial tokens directly from continuous buffer.
                init_st = 0
                init_ed = init_st + self.n_init
                global_h_k[:, :, init_st:init_ed] = self.global_remainder[0][:, :, init_st:init_ed]
                global_h_v[:, :, init_st:init_ed] = self.global_remainder[1][:, :, init_st:init_ed]
                ed = init_ed

                # Copy block by block, using block index, from the remainder tensor.
                for u in range(self.num_units):
                    for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                        remainder_st = init_ed + b_idx * self.block_size
                        remainder_ed = remainder_st + self.block_size
                        if remainder_st >= self.global_remainder[0].size(2):
                            break
                        st = init_ed + cnt * self.block_size
                        ed = st + self.block_size
                        global_h_k[u, :, st:ed] = self.global_remainder[0][u, :, remainder_st:remainder_ed]
                        global_h_v[u, :, st:ed] = self.global_remainder[1][u, :, remainder_st:remainder_ed]

            # Truncate output to the actual range written.
            global_h_k = global_h_k[:, :, :ed, :]
            global_h_v = global_h_v[:, :, :ed, :]
            # At this point, global_h_k/global_h_v contain [init] + [requested-retrieved-blocks] for each batch.

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        assert global_h_k.size(-2) <= self.n_init + self.n_local
        return global_h_k, global_h_v 

    def _calc_block_topk(
        self, global_h_q
    ):
        """
        Compute indices of top-k blocks for retrieval, using query-key similarity.

        Args:
            global_h_q (Tensor): Shape (batch_size, num_heads, length, dim_head)

        Returns:
            List[List[int]]: Per-batch, which block-ids to fetch.
        """
        # Average pool query across length axis, merge (num_heads, dim_head) for vector sim.
        global_h_q = global_h_q.mean(dim=2, keepdim=False)  # (batch_size, num_heads, dim_head)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(self.num_units, self.dim_head * self.unit_size)  # (batch_size, dim_head * num_heads)
        logits = None

        # Scenario 1: Not enough global blocks yet to require top-k selection.
        if self.num_global_block <= self.topk:
            if not self.init_exc:
                # Local window not full: operate on still-accumulating global_remainder only.
                assert self.global_remainder[0].size(-2) > self.n_init, f'{self.global_remainder[0].shape}'
                # Only consider tokens after the initial window as candidates.
                global_k = self.global_remainder[0][:, :, self.n_init:, :]  # (batch_size, n_head_kv, L, dim_head)
                global_k = self._from_group_kv(global_k)  # (batch_size, num_heads, L, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f'{global_k.shape}'
                block_num = global_k.size(-2) // self.block_size  # number of candidate "frames"
                if block_num <= self.topk:
                    # Not enough blocks to need retrieval: just take all indices.
                    ret = [list(range(block_num)) for _ in range(self.num_units)]
                else:
                    # Segment by block, mean-pool, then similarity
                    global_k = global_k.transpose(1, 2)  # (batch_size, length, num_heads, dim_head)
                    global_k = global_k.reshape(self.num_units, block_num, self.block_size, self.unit_size * self.dim_head)  # (batch, block, block_size, hidden)
                    global_k = global_k.mean(dim=-2, keepdim=False)  # (batch_size, block_num, hidden)
                    logits = torch.matmul(global_k, global_h_q[:, :, None]).squeeze(dim=-1)  # (batch_size, block_num)
            else:
                # Sliding window filled but total blocks < topk: retrieve all available.
                ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        else:
            # Scenario 2: Enough blocks staged, use feature-level similarity from stored representations.
            logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])  # (batch_size, block_num)

        # If using logits, divide up into chunks and pick top-k from chunks.
        if logits is not None:
            self.similarity = logits

            # Form chunked groups: e.g., each chunk is chunk_size sequential blocks
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            # For all full-size chunks, mean-pool over chunk; drop extra to a separate group at end (if exists)
            chunked_logits = logits[:, :logits.shape[1]-remainder_size].reshape(self.num_units, -1, self.chunk_size).mean(dim=-1)  # (batch_size, #chunks)
            
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
            
            # Pick top-(topk // chunk_size) chunks for every batch, then map chunk index to block indices
            ret = chunked_logits.topk(self.topk//self.chunk_size, dim=1).indices
            ret = ret.sort(dim=1)[0][:, :, None]  # sort chunk indices (batch_size, n_chunk, 1)
            ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]  # expand to (batch_size, n_chunk, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # flatten to (batch_size, topk)
            ret = ret.cpu().tolist()

            # Filter out any overflow indices due to last chunk possibly being short.
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return ret

    def get_global_hidden_and_mask(self, exc_length):
        """
        Prepare the global KV-buffer up to exc_length tokens, including ensuring the init KV-cache is filled.

        Args:
            exc_length (int): Number of additional tokens to consider for staging.

        Returns:
            (global_h_k, global_h_v): Prepared keys and values for global context.
        """
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        # Indices and lengths for fresh context tokens to append.
        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # Check if we're still filling the initial window; if so, keep loading until it's full.
        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2),
                global_remainder_len - self.n_local
            )
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True  # Initial window now filled; future appends trigger offloading

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        # (Re)load the actual "init" window from init_k/init_v into buffer if necessary
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        global_h_k = global_h_k[:, :, :init_ed, :]
        global_h_v = global_h_v[:, :, :init_ed, :]

        return global_h_k, global_h_v

    def _append(
        self,
        local_q, local_k, local_v, global_q,
    ):
        """
        Calculate local+global attention results for a chunk.

        Args:
            local_q (Tensor): (batch_size, num_heads, length, dim_head)
            local_k (Tensor): (batch_size, num_heads, length, dim_head)
            local_v (Tensor): (batch_size, num_heads, length, dim_head)
            global_q (Tensor): (batch_size, num_heads, length, dim_head)

        Returns:
            chunk_o (Tensor): (batch_size, num_heads, length, dim_head)
        """
        # Positionally encode Q/K via RoPE or similar
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # Attend over block (chunk) using local context (sliding window)
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=False, sliding_window=self.n_local
        )

        # Load/init the global context cache for this chunk
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(exc_length=global_q.size(-2))

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # Attend over (buffered) global KV with complement mode to ensure full context coverage
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True,  # the final append operation
            get_score=False, 
            sliding_window=None,
            complement_sliding_window=True,
        )

        o, _ = attn.get_result()

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _append_global(
        self
    ):
        """
        Offload (segment and shift to CPU memory) a block of the global KV context,
        once local window is full and further tokens would exceed GPU cache quota.

        This is called after every token/block append, and is only active once block regime is triggered.
        """
        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # Blocks only offloaded once the sliding window is filled (init_exc==True)
        if self.init_exc:
            assert global_remainder_len % self.block_size == 0, f'global_remainder_len: {global_remainder_len}, block_size: {self.block_size}'
            while global_remainder_len > 0:
                global_remainder_len -= self.block_size

                # For each batch, wrap this block in a MemoryUnit and offload to CPU memory,
                # making it available for later staging when needed.
                for u in range(self.num_units):
                    self.global_blocks[u].append((
                        MemoryUnit(
                            kv=(
                                self.global_remainder[0][u, :, global_remainder_st:global_remainder_st + self.block_size, :],
                                self.global_remainder[1][u, :, global_remainder_st:global_remainder_st + self.block_size, :]
                            ),
                            cache=self.cuda_cache,
                            load_to_cache=False,
                            pin_memory=self.pin_memory
                        )
                    ))

                # For block retrieval, take the mean-pooled K embedding of each new block as its representative signature.
                global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
                global_block_k = self._from_group_kv(global_block_k)  # (batch_size, num_heads, block_size, dim_head)

                # Within-head average pooling and reshape for vector sim
                global_block_k = global_block_k.mean(dim=-2, keepdim=False)  # (batch_size, num_heads, dim_head)
                global_block_k = global_block_k.reshape(self.num_units, -1)  # (batch_size, num_heads * dim_head)
                global_block_k = global_block_k[:, None, :]  # Add chunk dimension
                for u in range(self.num_units):
                    self.block_k[u].append(global_block_k[u])
                
                self.num_global_block += 1
                global_remainder_st += self.block_size

        self._global_remainder_st = global_remainder_st
        self._global_remainder_ed = global_remainder_ed

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        """
        The main entry point for appending new chunks of tokens to the ContextManager.

        Args:
            local_q, local_k, local_v: Local (not-yet-global) Q/K/V blocks [shape: (batch, head, length, dim)].
            global_q, global_k, global_v: New global context Q/K/V to concatenate.

        Returns:
            Tensor: Output chunk after attention (batch_size, num_heads, total_len, dim_head)
        """
        # First-time: initialize all buffers/state per batch, head, etc.
        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2)
        
        # Wait for any global async streams to finish before proceeding.
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # Update local KV-cache by concatenating in the new local block.
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # Append new global K/V to staged global_remainder, setting start/end pointers
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)
            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

        # Positional encode all new (possibly global) queries using the rotary embedding.
        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        o_list = []
        # Chunk up the new input (e.g., exc_block_size tokens at a time):
        for st in range(0, input_length, self.exc_block_size):
            ed = min(st + self.exc_block_size, input_length)

            # Compute range within local_k/v: slide window to the current block position
            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length

            # Perform append and compute attention for this block
            chunk_o = self._append(
                local_q=local_q[:, :, st:ed, :],
                local_k=self.local_k[:, :, kv_st: kv_ed, :],
                local_v=self.local_v[:, :, kv_st: kv_ed, :],
                global_q=global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            # After appending, stage new global blocks (if window is full).
            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        self.length += input_length

        # Truncate local_k/v cache to sliding window
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]

        # Update the global_remainder to only reference tokens not yet processed.
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        assert not self.init_exc or self._global_remainder_st == self._global_remainder_ed, f'self.init_exc: {self.init_exc}, global_remainder_st: {self._global_remainder_st}, global_remainder_ed: {self._global_remainder_ed}'
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :]
            )

        ret = torch.cat(o_list, dim=-2)
        
        return ret
    
    def size(self, *args, **kwargs):
        """
        Return the current logical length (total tokens indexed/cached).
        """
        return self.length

    def calculate_cpu_memory(self):
        """
        Sum the memory (in bytes) consumed by all blocks (MemoryUnits) currently offloaded onto CPU.
        """
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory
