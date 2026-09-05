# InferCap architecture

This document distinguishes the current package structure from the longer-term architecture direction. Names in the target section are proposals, not implemented modules or public APIs.

## Current architecture

InferCap installs the `infercap` package from `src/infercap/`. The console
command and `python -m infercap` both dispatch through `infercap.cli:main`:

- `infercap check` uses `infercap.preflight.cli` for arguments and presentation.
- `infercap benchmark` uses `infercap.benchmark.cli` for arguments and execution setup.
- Running `infercap` without a subcommand displays help.

```text
src/infercap/
├── __init__.py
├── __main__.py
├── cli.py
├── preflight/
│   ├── cli.py          # Arguments, JSON/text output, exit status
│   └── checks.py       # Environment/model checks and serving recommendations
├── benchmark/
│   ├── cli.py          # Arguments and benchmark configuration
│   ├── runner.py       # Load execution and report orchestration
│   ├── client.py       # OpenAI-compatible streaming requests
│   ├── nvidia.py       # NVIDIA telemetry through nvidia-smi
│   └── vllm_metrics.py # vLLM Prometheus metrics
├── analysis/
│   ├── metrics.py      # Distributions and saturation estimates
│   └── plotting.py     # PNG dashboards from JSON reports
└── config/
    ├── prompts.py      # Benchmark prompt inputs
    └── settings.py     # Legacy configuration constants
```

Each subpackage has an explicit `__init__.py`. Operational helpers live under
`scripts/`, outside the installed Python package: `benchmark.sh`, `serve_vllm.sh`,
and `check_vllm_environment.py`. The serving helper is a hardware-specific example;
it is not a general deployment command.

`preflight.checks.run_checks` and `benchmark.runner.run` are callable without CLI
parsing. Preflight still accepts and updates an `argparse.Namespace`; replacing
that with a dedicated configuration model is a separate change. Benchmark
execution still coordinates report writing and plotting in its runner.

## Package migration

Run `uv sync` after updating. Public `infercap check` and `infercap benchmark`
commands retain their options and exit codes. Internal imports now use
`infercap.*`; the former `src.*` imports are not compatibility aliases. Use
`python -m infercap` in place of the removed root `main.py` and forwarding modules.
The source-layout package must be installed before running tests. CI builds and
installs a wheel and verifies the CLI outside the checkout.

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
- Preflight checks, external-system access, discovery, and serving recommendations still share one core module; CLI parsing and presentation are separate.
- Runtime compatibility and serve-command generation are not behind a runtime adapter.
- The weight estimate does not model KV cache or all runtime allocations.
- Family discovery depends directly on Hugging Face Hub metadata and the installed vLLM registry.
- Endpoint verification confirms only reachability and model listing.
- Benchmark execution, telemetry coordination, report writing, and visualization are tightly orchestrated by one module.
- The versioned JSON report is covered by fixture contract tests, but has no standalone machine-readable schema.
- Load generation runs from one client process, which can itself become a bottleneck.

## Further architecture work

Further work should establish clearer boundaries while preserving current behavior and commands. Proposed responsibilities are:

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
