# InferCap Agent Instructions

## Project

InferCap is an open-source, hardware-aware inference capacity, compatibility, serving, and benchmarking toolkit.

The project aims to answer:

1. What hardware is available?
2. What models or inference workloads can run on it?
3. Which runtime and configuration should be used?
4. Can the configuration run successfully?
5. How well does it perform?

## Current scope

The current implementation primarily supports:

* NVIDIA GPUs
* vLLM
* model feasibility / preflight checks
* model discovery
* serving configuration generation
* runtime verification
* inference benchmarking
* GPU and vLLM telemetry
* saturation analysis

## Future direction

The architecture should allow future support for:

* CPU
* AMD GPU
* NPU
* TPU
* DPU / SmartNIC
* additional inference accelerators
* additional inference runtimes
* model types beyond LLMs

Do not implement future backends or hardware support unless explicitly requested.

## Engineering principles

* Preserve existing behavior during refactors.
* Prefer incremental, reviewable changes over rewrites.
* Avoid premature abstractions.
* Keep hardware-specific logic isolated.
* Keep runtime-specific logic isolated.
* Separate CLI / presentation logic from core logic.
* Do not mix unrelated refactors into one change.
* Add or update tests when behavior changes.
* Keep public interfaces small and explicit.
* Do not claim unsupported functionality in documentation.
* Treat feasibility estimates as estimates, not runtime guarantees.

## Python environment

Use `uv` for dependency and environment management.

Common commands:

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

Use the existing project tooling when formatting, linting, or testing is already configured.

Do not introduce new development dependencies without a clear reason.

## Refactoring workflow

For architecture or structural changes:

1. inspect the existing implementation,
2. identify current responsibilities and coupling,
3. propose the smallest useful change,
4. list affected files,
5. preserve existing behavior,
6. implement incrementally,
7. run relevant tests,
8. document externally visible changes.

## Contributions

Significant features and refactors should correspond to a GitHub issue.

Pull requests should:

* solve one coherent problem,
* remain reasonably small,
* include tests when applicable,
* avoid unrelated cleanup,
* document user-visible behavior changes.

Contributor-facing tasks should have clear scope and acceptance criteria.
