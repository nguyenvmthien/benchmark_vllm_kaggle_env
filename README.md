# InferCap

InferCap is an open-source toolkit for checking inference feasibility, selecting a serving configuration, verifying a deployment, and measuring its capacity. The current implementation supports **NVIDIA GPUs and vLLM**.

Current capabilities include preflight checks, Hugging Face model-family discovery, static model-weight memory estimates, recommended `vllm serve` commands, endpoint verification, three benchmark modes, GPU and vLLM telemetry, saturation analysis, and JSON/PNG output artifacts.

## Installation

Python 3.12 or newer is required. Install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The environment includes vLLM and requires a platform on which the configured vLLM/PyTorch stack can run.

## Quick start

Check an exact model against the local environment and detected GPUs:

```bash
uv run infercap check --model mistralai/Mistral-7B-Instruct-v0.3
```

If the checks pass, the output includes a recommended `vllm serve` command. Start it, then benchmark the OpenAI-compatible endpoint:

```bash
uv run infercap benchmark \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --concurrency 1,4,8,16,32,64
```

Use `infercap check` for preflight checks and `infercap benchmark` for benchmarking. Run `uv sync` after updating to install the package. `uv run python -m infercap` invokes the same CLI; internal Python imports now use `infercap.*`.

## Preflight and exact model checks

Preflight checks Python, PyTorch, vLLM, system memory, NVIDIA GPUs, model architecture, and estimated model-weight memory. It reports `PASS`, `WARN`, and `FAIL` results and exits non-zero for hard failures. Model size comes from Hugging Face safetensors metadata when available. Cached or local models can use offline mode and an explicit parameter count:

```bash
uv run infercap check \
  --model ./models/example-model \
  --offline --model-size-b 7 --quantization awq
```

The memory calculation estimates weights plus 15% loading overhead using current free GPU memory and the selected tensor-parallel size. It excludes KV cache and other runtime allocations.

**Feasibility is a static estimate, not a runtime guarantee.** Passing preflight does not prove that a model will load successfully or meet a performance target. Use `--json` for machine-readable output.

## Family discovery

A family name triggers a Hugging Face Hub search. InferCap filters candidates using the installed vLLM architecture registry, estimates weight memory, ranks candidates, and selects a likely fit:

```bash
uv run infercap check Qwen --recommend-limit 5
```

Family discovery requires Hub access. Exact cached model checks can use `--offline`.

## Serving recommendation

Successful preflight output includes a generated `vllm serve` command. The default `balanced` profile enables prefix caching and chunked prefill and supplies scheduler, model-length, dtype, KV-cache dtype, and generation-config flags. Other profiles are `safe`, `latency`, and `throughput`; individual settings can be overridden through the preflight CLI.

Recommendations are configuration starting points. They do not start the server or validate runtime stability.

## Endpoint verification

Add `--check-server` to query `/v1/models` and confirm that the requested model ID is present:

```bash
uv run infercap check \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --check-server --api-url http://localhost:8000/v1/chat/completions
```

## Benchmark modes

The benchmark sends streamed chat-completion requests to an OpenAI-compatible endpoint:

- `concurrency` runs a closed-loop sweep; each level sends `concurrency × requests-per-worker` requests.
- `burst` submits all requests in a level together.
- `request-rate` submits open-loop traffic at fixed offered rates for a configured duration.

Select a mode with `--mode`; relevant controls include `--concurrency`, `--burst-sizes`, `--request-rates`, `--rate-duration`, `--requests-per-worker`, and `--max-tokens`. Use `--prompt-profile shared-prefix` to exercise prefix-cache reuse.

## Telemetry

Each load level records request success and errors, output TPS, RPS, end-to-end latency, time to first token (TTFT), and inter-token/decode latency with mean, p50, p95, and p99 summaries.

InferCap samples NVIDIA utilization and VRAM through `nvidia-smi`. It samples compatible vLLM Prometheus metrics from the API origin's `/metrics` endpoint by default, including KV-cache use, prefix-cache activity, preemptions, and running or waiting requests. Unavailable telemetry is reported as unavailable or as a warning, not as measured zero usage.

## Saturation analysis

Saturation is reported at the last healthy tested level before a configured threshold is crossed. Defaults are throughput growth below 10%, TTFT p95 above 2 seconds, or request error rate above 1%. Adjust them with `--min-tps-growth`, `--max-ttft-p95`, and `--max-error-rate`.

## Output artifacts

By default, each benchmark writes:

- `benchmark_output/benchmark_<UTC timestamp>.json`, containing configuration, per-level results, telemetry summaries, and saturation analysis;
- `benchmark_output/benchmark_<UTC timestamp>.png`, a four-panel dashboard rendered from the report.

Change the destination with `--output-dir`. Preflight JSON is printed to standard output with `--json` and can be redirected by the caller.

## Limitations

- Hardware detection and telemetry are NVIDIA-specific and depend on `nvidia-smi`.
- Compatibility checks and serving recommendations are vLLM-specific.
- Model discovery and automatic parameter counts depend on Hugging Face metadata and network access unless cached or overridden.
- Weight-memory estimates exclude KV cache and other runtime allocations.
- Endpoint verification checks reachability and the served model ID; it is not a full inference validation.
- Benchmarks target the streamed OpenAI-compatible chat-completions API and run from one client process.

InferCap does not currently support other hardware backends or inference runtimes.

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src scripts tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [docs/architecture.md](docs/architecture.md) for the current and target architecture.
