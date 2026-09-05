# uv run vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
#   --tensor-parallel-size 2 \
#   --dtype float16 \
#   --gpu-memory-utilization 0.92 \
#   --max-model-len 4096 \
#   --enable-prefix-caching \
#   --enable-chunked-prefill \
#   --max-num-batched-tokens 8192 \
#   --max-num-seqs 64


#T4 GPU
VLLM_ATTENTION_BACKEND=xformers uv run vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64