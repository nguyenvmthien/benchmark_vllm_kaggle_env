# InferCap architecture

This document distinguishes the code that exists today from the intended v0.1 architecture. Names in the target section are proposals, not implemented modules or public APIs.

## Current architecture

InferCap currently has two public console scripts and a compatibility entry point:

- `benchmark-vllm-check` calls `src.scripts.check_runable:main` for preflight, discovery, recommendation, and endpoint checks.
- `benchmark-vllm` calls `src.benchmark.main:main` for load generation, measurement, analysis, and reports.
- `main.py` also invokes the benchmark entry point.

The current implementation is organized as follows:

- `src/scripts/check_runable.py` owns preflight CLI parsing, environment and model checks, Hub discovery, static weight-memory estimation, vLLM configuration recommendations, endpoint verification, and presentation.
- `src/benchmark/main.py` owns benchmark CLI parsing and orchestration for concurrency, burst, and request-rate modes.
- `src/benchmark/client.py` sends and measures streamed OpenAI-compatible requests.
- `src/benchmark/gpu.py` samples NVIDIA GPU utilization and memory.
- `src/benchmark/kv_cache.py` samples and normalizes supported vLLM Prometheus metrics.
- `src/analyse/metrics.py` computes distributions and selects a saturation level.
- `src/analyse/visual.py` renders a PNG dashboard from a JSON report.
- `src/config/prompts.py` and `src/config/settings.py` provide benchmark inputs and configuration values.

The preflight module combines core checks with CLI output, while benchmark orchestration coordinates load generation, telemetry, analysis, persistence, and plotting.

## Current execution flows

### Preflight and recommendation

1. Parse an exact model/path or a family query and serving-profile options.
2. Check Python, vLLM, PyTorch, available system memory, and GPUs reported by `nvidia-smi`.
3. For a family query, search Hugging Face, inspect model configurations, filter against the installed vLLM registry, rank candidates, and select one.
4. Resolve or accept a model parameter count and estimate weight memory plus loading overhead against currently free VRAM.
5. Validate model architectures against vLLM's supported architecture registry.
6. Optionally query `/v1/models` to verify endpoint reachability and the served model ID.
7. Print human-readable or JSON results and, when hard checks pass, a recommended `vllm serve` command.

This is static characterization. It does not load the model and is not a runtime guarantee.

### Benchmark and reporting

1. Parse the endpoint, model, load mode, thresholds, sampling intervals, and output location.
2. Derive the vLLM metrics URL from the API origin unless one is supplied.
3. Warm up the endpoint.
4. Run each configured concurrency, burst-size, or offered request-rate level.
5. Concurrently sample NVIDIA GPU and vLLM Prometheus telemetry.
6. Aggregate request latency, token throughput, errors, and telemetry for each level.
7. Select the last healthy level before throughput growth, TTFT, or error-rate criteria trigger saturation.
8. Write a timestamped JSON report and render the corresponding PNG dashboard.

## NVIDIA-specific responsibilities

- Preflight invokes `nvidia-smi` to discover GPU identity, compute capability, and total/free memory.
- Benchmark telemetry invokes `nvidia-smi` repeatedly to sample utilization and used/total VRAM.

GPU capacity calculations assume similar devices participating through vLLM tensor parallelism. Missing NVIDIA tooling is a hard preflight failure and produces unavailable benchmark GPU telemetry.

## vLLM-specific responsibilities

- Require the installed `vllm` package during preflight.
- Compare Transformers architectures with `vllm.ModelRegistry`.
- Generate `vllm serve` flags and scheduler profiles.
- Use vLLM-oriented runner and quantization choices.
- Sample known vLLM V0/V1 Prometheus metric names for KV cache, prefix caching, preemptions, and scheduler queues.
- Expect vLLM's OpenAI-compatible streaming and usage behavior in benchmark measurements.

## Known architectural limitations

- Hardware discovery, feasibility, and telemetry do not share a hardware-provider interface.
- Preflight domain logic, external-system access, CLI parsing, and presentation are concentrated in one module.
- Runtime compatibility and serve-command generation are not behind a runtime adapter.
- The weight estimate does not model KV cache or all runtime allocations.
- Family discovery depends directly on Hugging Face Hub metadata and the installed vLLM registry.
- Endpoint verification confirms only reachability and model listing.
- Benchmark execution, telemetry coordination, report writing, and visualization are tightly orchestrated by one module.
- The versioned JSON report has no separately defined schema or compatibility policy.
- Load generation runs from one client process, which can itself become a bottleneck.

## v0.1 target architecture

The v0.1 direction is to establish clearer boundaries while preserving current behavior and commands. Proposed responsibilities are:

- a small CLI/presentation layer for argument parsing, exit codes, and human or JSON output;
- core data models for detected hardware, feasibility results, serving recommendations, benchmark configurations, and reports;
- an NVIDIA hardware adapter for discovery, capacity inputs, and telemetry;
- a vLLM runtime adapter for architecture compatibility, serving flags, endpoint semantics, and runtime metrics;
- services that orchestrate preflight, discovery, recommendation, verification, and benchmarking without owning presentation;
- analysis and artifact writers that consume explicit report data rather than CLI state.

These are target boundaries only. They should be introduced incrementally through issue-backed changes, with tests that demonstrate preserved behavior. The target should keep hardware-specific and runtime-specific logic isolated without adding abstractions for unimplemented backends.

## Non-goals

- Implementing additional hardware backends or inference runtimes.
- Renaming existing console scripts or entry points without a compatibility plan.
- Replacing runtime validation with a promise based on static feasibility.
- Building a distributed load generator.
- Redesigning the report schema without a separately scoped change.
- Broad production-code rewrites solely to match the target boundaries.
