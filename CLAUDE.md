# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReKV is the official PyTorch implementation of "Streaming Video Question-Answering with In-context Video KV-Cache Retrieval" (ICLR 2025). It provides a training-free approach that integrates with existing Video Large Language Models (Video-LLMs) to enable efficient streaming video QA via sliding-window attention and KV-Cache retrieval from RAM/disk.

Based on [InfLLM](https://github.com/thunlp/InfLLM), [StreamingLLM](https://github.com/mit-han-lab/streaming-llm), and [Flash-VStream](https://github.com/IVGSZ/Flash-VStream).

## Setup

```bash
bash prepare.sh                              # Creates conda env "rekv" with Python 3.11
sudo sysctl -w vm.max_map_count=262144       # Required for KV-Cache offloading
```

Requires a pinned transformers commit (`66bc4def`) installed from source. Models go in `model_zoo/`, datasets in `data/`.

## Running Evaluation

```bash
python -m video_qa.run_eval \
    --num_chunks 4 \
    --model llava_ov_7b \
    --dataset qaego4d \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64
```

- `--num_chunks`: Number of parallel processes (typically = number of GPUs)
- `--model`: `llava_ov_0.5b`, `llava_ov_7b`
- `--dataset`: `mlvu`, `qaego4d`, `videomme`
- `--only_eval`: Skip inference, only run evaluation on existing results
- `--sample`: Use sample/subset annotation files for quick testing
- `--debug true`: Enables debugpy on port 2345

Individual VQA solvers can be run directly:
```bash
python video_qa/rekv_offline_vqa.py --model llava_ov_7b --anno_path data/qaego4d/test_mc.json ...
```

## Architecture

### Three-Phase Pipeline

1. **Encode video** → `encode_video()` processes frames chunk-by-chunk. The first chunk automatically calls `encode_init_prompt()` to produce the initial KV-Cache. Subsequent frames are encoded via the patched language model, with KV-Caches managed by `ContextManager` (sliding window on GPU, overflow offloaded to CPU RAM).
2. **Retrieval** → The question text is encoded against the context memory's representative keys (`VectorTensor`) to find the top-k most relevant KV-Cache blocks. These blocks are loaded from CPU back to GPU.
3. **Question answering** → The full prompt (question + answer format) is encoded with the retrieved KV-Cache, then autoregressive decoding generates the answer using standard sliding-window attention.

### Model Integration Pattern

`LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV)` in `model/llava_onevision_rekv.py` uses dual inheritance. HuggingFace's `__init__` is called at construction; ReKV-specific initialization happens later via `init_rekv()`.

The `load_model()` factory function handles: loading the HF model, calling `init_rekv()` to set ReKV parameters, and calling `patch_hf()` to monkey-patch the language model's attention layers.

### Attention Monkey-Patching (`model/patch.py`)

`patch_hf()` replaces HuggingFace attention forward methods with ReKV's custom attention. It also replaces the model's forward to use custom RoPE (`RotaryEmbeddingESM`) and custom layer iteration. Currently only targets `Qwen2ForCausalLM` / `Qwen2Model` (LLaVA-OneVision). Qwen2.5-VL / Qwen3-VL support is stubbed out but commented.

### KV-Cache Management (`model/attention/kv_cache_manager.py`)

- `ContextManager`: Orchestrates the KV-Cache lifecycle — GPU local window (`CudaCache`), CPU offloading (`MemoryUnit`), block-based indexing, representative key storage (`VectorTensor`), and retrieval mode
- `CudaCache`: Fixed-size GPU memory allocator for the sliding window
- `MemoryUnit`: Manages CPU↔GPU transfers with `pin_memory` support and async CUDA events
- `VectorTensor`: Dynamically growing GPU cache storing average-pooled representative keys per block, used for cosine-similarity retrieval
- Block size is 196 (one frame's tokens after vision pooling)

### Custom Attention (`model/attention/rekv_attention.py`)

`rekv_attention_forward()` returns a closure with two code paths:
- **Video encoding**: Delegates to `ContextManager.append()` which manages sliding-window + offloading
- **QA / retrieval**: Falls back to standard sliding-window attention (InfLLM-style) with init tokens, retrieval from CPU cache, and RoPE

Two backends in `dot_production_attention/`: pure PyTorch (`torch_impl.py`) and Triton/Flash-Attention (`triton_impl.py`), selected via the `fattn` flag.

### RoPE (`model/attention/rope.py`)

`RotaryEmbeddingESM`: Standard rotary position embeddings with caching. Provides `forward(q, k)` for consecutive-position RoPE and `apply_rotary_pos_emb_one_angle(x, index)` for fixed-position RoPE (used for init tokens and global queries during video encoding).

### Video QA Pipeline (`video_qa/`)

- `BaseVQA`: Abstract class handling video loading (via decord), prompt formatting, chunked multi-GPU evaluation, and result CSV export
- `ReKVOfflineVQA`: Processes entire video before answering questions (primary solver)
- `ReKVStreamVQA`: Interleaves video encoding with QA (exists but not wired into `run_eval.py`)
- `run_eval.py`: Orchestrator that spawns parallel processes per GPU, merges CSV results, then runs dataset-specific evaluation scripts from `video_qa/eval/`

### Key Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_local` | Sliding window size (tokens) | 15000 |
| `n_init` | Initial prompt token count | Auto from prompt |
| `retrieve_size` | Number of KV-Cache blocks to retrieve | 64 |
| `block_size` | Tokens per block (= tokens per frame) | 196 |
| `chunk_size` | Retrieval chunk size (groups of blocks) | 1 |
| `sample_fps` | Video sampling rate | 0.5 |
| `encode_chunk_size` | Frames per encoding batch | 64 |
| `max_cached_block` | Max blocks cached on GPU | 128 |
| `exc_block_size` | Execution block size for chunked attention | 196 |
