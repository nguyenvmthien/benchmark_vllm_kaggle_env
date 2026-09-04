# vLLM performance benchmark

A reproducible concurrency sweep for an OpenAI-compatible vLLM endpoint. It records:

- TTFT, inter-token/decode latency (ITL/TPOT), and end-to-end latency at mean/p50/p95/p99
- aggregate output TPS, request throughput (RPS), success/error rate
- NVIDIA GPU utilization and VRAM usage during every load level
- a configurable saturation point based on TPS growth, TTFT SLO, and error rate
- machine-readable JSON and a four-panel PNG dashboard with notices

## 1. Preflight

The checker does not import CUDA libraries at startup and gives stable PASS, WARN, and FAIL output.
Exit status is non-zero only for hard failures, so it can be used in CI.

    uv run python src/scripts/check_runable.py --model mistralai/Mistral-7B-Instruct-v0.3

You can also provide only a model family. The checker searches Hugging Face, filters against the
vLLM registry installed on this machine, estimates quantized/non-quantized weight memory, ranks
models against currently free VRAM, and reports the likely vLLM runner:

    uv run python src/scripts/check_runable.py Qwen
    uv run python src/scripts/check_runable.py --family Qwen --recommend-limit 10

Use `--runner generate|pooling|transcription` only when you need to override vLLM's automatic
runner selection. Family discovery requires Hub access; exact cached model checks support `--offline`.

Also verify a running OpenAI-compatible endpoint:

    uv run python src/scripts/check_runable.py --check-server --json

Model size is automatically read from Hugging Face safetensors metadata. Use --model-size-b only as an override for private or non-safetensors repositories. The memory estimate covers static model weights plus 15% loading overhead. KV cache depends on
context length, batch shape, architecture, and cache dtype, so it is explicitly not presented as
an exact estimate.

## 2. Serve

Edit src/scripts/serve.sh for the target hardware, then:

    bash src/scripts/serve.sh

## 3. Benchmark

    bash src/scripts/benchmark.sh \
      --model mistralai/Mistral-7B-Instruct-v0.3 \
      --concurrency 1,4,8,16,32,64 \
      --requests-per-worker 4 \
      --max-tokens 128

Each level runs concurrency times requests-per-worker requests through a semaphore, producing a
closed-loop load. Increase requests-per-worker (for example, 10-20) for publication-quality
percentiles. The defaults classify saturation when any condition is met:

- throughput growth from the previous level is below 10%
- TTFT p95 exceeds 2 seconds
- request error rate exceeds 1%

Tune these using --min-tps-growth, --max-ttft-p95, and --max-error-rate.
The recommended operating concurrency is the last healthy level before the trigger. If no trigger
is found, expand --concurrency.

Outputs are written to benchmark_output/benchmark_<UTC timestamp>.json and .png. Generate a
dashboard again with:

    uv run python -m src.analyse.visual benchmark_output/benchmark_<timestamp>.json

## Metric notes

- Output TPS = successful completion tokens / wall-clock duration of the level.
- TTFT = request start to first non-empty streamed content.
- ITL / decode latency = time from first token to stream completion / (output_tokens - 1).
- Token counts use the OpenAI stream final usage.completion_tokens. Older servers fall back to
  SSE content-event counts and emit a warning because one event is not guaranteed to equal one token.
- GPU values are sampled using nvidia-smi; missing telemetry is visible as a warning, never silently
  treated as zero utilization.

## Development checks

    python -m unittest discover -s tests -v
    python -m compileall -q src main.py
