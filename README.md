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

Use `--runner generate|pooling|draft` only when you need to override vLLM's automatic
runner selection. Family discovery requires Hub access; exact cached model checks support `--offline`.

The generated serve command uses a balanced inference profile by default: prefix caching, chunked
prefill, reproducible vLLM generation defaults, automatic KV-cache dtype, 64 concurrent sequences,
and an 8192-token scheduler budget. Select another goal with `--profile safe|latency|throughput`,
or override `--max-num-seqs` and `--max-num-batched-tokens` directly. Disable either optimization
with `--no-enable-prefix-caching` or `--no-enable-chunked-prefill`.

### Copy-paste examples

Discover the best Qwen checkpoint for the GPUs that are currently free, using the balanced profile:

```bash
uv run python src/scripts/check_runable.py Qwen
```

Inspect more Hub candidates and print the ten best recommendations:

```bash
uv run python src/scripts/check_runable.py \
  --family Qwen \
  --family-candidates 80 \
  --recommend-limit 10
```

Check an exact instruction model with conservative scheduler limits:

```bash
uv run python src/scripts/check_runable.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --profile safe \
  --max-model-len 4096
```

Check an AWQ checkpoint for two GPUs. Quantization is normally detected from the checkpoint, but it
can be made explicit for private or incomplete model metadata:

```bash
uv run python src/scripts/check_runable.py \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --profile balanced
```

Tune for low latency or maximum throughput:

```bash
# Lower scheduler queue/batch limits for more predictable latency.
uv run python src/scripts/check_runable.py Qwen --profile latency

# Larger scheduling limits for an offline or high-throughput service.
uv run python src/scripts/check_runable.py Qwen --profile throughput
```

Override scheduler limits directly when the built-in profile is not enough:

```bash
uv run python src/scripts/check_runable.py Qwen \
  --profile balanced \
  --max-num-seqs 96 \
  --max-num-batched-tokens 12288 \
  --max-model-len 8192 \
  --served-model-name qwen
```

Check an embedding/reranker model by selecting the pooling runner:

```bash
uv run python src/scripts/check_runable.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --runner pooling \
  --profile throughput
```

Disable optimizations when diagnosing model compatibility or scheduler issues:

```bash
uv run python src/scripts/check_runable.py Qwen \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill
```

Check a model already available locally without accessing Hugging Face:

```bash
uv run python src/scripts/check_runable.py \
  --model ./models/Qwen2.5-7B-Instruct-AWQ \
  --offline \
  --model-size-b 7.6 \
  --quantization awq
```

Produce machine-readable output for CI or another script:

```bash
uv run python src/scripts/check_runable.py Qwen --json > preflight.json
jq '.ready, .selected_model, .inference_recommendation, .recommended_serve_command' preflight.json
```

Verify a running OpenAI-compatible endpoint:

```bash
uv run python src/scripts/check_runable.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --check-server \
  --api-url http://localhost:8000/v1/chat/completions \
  --json
```

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
percentiles.

Run simultaneous burst levels (every request in a level is submitted together):

    bash src/scripts/benchmark.sh --mode burst --burst-sizes 16,32,64,128

Run open-loop traffic at fixed offered rates; submission does not wait for earlier responses:

    bash src/scripts/benchmark.sh --mode request-rate \
      --request-rates 1,2,4,8 --rate-duration 60

The request-rate duration is the submission window. Each level completes after all submitted
streams finish, so growing latency and total duration expose overload. Burst mode disables the
client's default 100-connection limit so large bursts reach the server together.

The defaults classify saturation when any condition is met:

- throughput growth from the previous level is below 10%
- TTFT p95 exceeds 2 seconds
- request error rate exceeds 1%

Tune these using --min-tps-growth, --max-ttft-p95, and --max-error-rate.
The recommended operating concurrency is the last healthy level before the trigger. If no trigger
is found, expand --concurrency.

The benchmark command is a single pipeline: after all load levels finish, it writes
benchmark_output/benchmark_<UTC timestamp>.json and immediately renders the matching .png
dashboard. A plotting failure fails the command instead of silently leaving a partial result.

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
