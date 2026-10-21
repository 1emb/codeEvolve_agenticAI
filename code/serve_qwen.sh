#!/bin/bash
# Serve Qwen3-Coder-30B-A3B-Instruct (bf16) on one H100 with vLLM, OpenAI-compatible API.
#   bash serve_qwen.sh            # GPU 0, port 8000
#   CUDA_VISIBLE_DEVICES=1 PORT=8001 bash serve_qwen.sh
# CodeEvolve then needs: export API_BASE=http://localhost:8000/v1 API_KEY=EMPTY
export HF_HUB_CACHE=/home/sheshnagnobackup/mnguyen/huggingface_models
export MAX_JOBS=16   # cap FlashInfer/ninja JIT parallelism; 128-way nvcc got SIGKILLed
export VLLM_USE_DEEP_GEMM=0   # deep_gemm not installed; vllm 0.20 warmup probes it anyway and crashes (FP8-only, irrelevant for bf16)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate egrat     # vllm 0.20, torch cu130
exec vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --port "${PORT:-8000}" \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --moe-backend triton   # FlashInfer CUTLASS MoE needs a nvcc JIT build that fails here
