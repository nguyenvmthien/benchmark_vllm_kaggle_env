# Standardize the InferCap package and CLI boundaries

## Problem

The installed package is named `src`, module names mix spelling and responsibilities,
and forwarding entry points obscure the public CLI. Preflight and benchmark modules
also combine argument parsing with execution logic.

## Scope and plan

1. Move Python modules into the explicit `src/infercap/` package and update packaging.
2. Rename analysis/plotting modules and split preflight and benchmark CLI code from
   their execution modules without changing algorithms, options, or report fields.
3. Replace redundant forwarding files with `infercap.__main__`; move operational
   helper scripts to the repository-level `scripts/` directory.
4. Update imports, tests, CI, and documentation, then validate installed artifacts.

Affected files: `pyproject.toml`, Python modules under `src/`, helper scripts,
`tests/`, the CI workflow, `README.md`, `CONTRIBUTING.md`, and architecture docs.

## Acceptance criteria

- `infercap check` and `infercap benchmark` retain arguments, defaults, and exit codes.
- `python -m infercap` uses the same dispatcher as the console command.
- Preflight and benchmark report characterization tests pass unchanged in assertions.
- A built wheel imports and exposes the CLI outside the repository checkout.
- CLI help does not load vLLM, PyTorch, or require a GPU.
- No imports use the former `src` package; operational scripts are outside the wheel.
- No new hardware/runtime support, dependency, or report schema is introduced.

## Compatibility

The public console commands remain unchanged. Internal `src.*` imports and direct
execution of the removed `main.py`/benchmark forwarding modules are replaced by
`infercap.*` imports and `python -m infercap`. Run `uv sync` after updating.

## Issue tracking

This document is an issue-ready local scope record. A GitHub issue has not been
linked: the available GitHub CLI is not authenticated.
