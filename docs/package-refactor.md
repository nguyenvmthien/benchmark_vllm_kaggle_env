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

Tracked under [Packaging #8](https://github.com/nguyenvmthien/infercap/issues/8)
and the package/CLI portion of [Architecture #1](https://github.com/nguyenvmthien/infercap/issues/1).
This refactor does not complete the remaining hardware/runtime extraction work in #1.

## Validation results

- Package layout, CLI/core separation, script relocation, and documentation completed.
- Source distribution and wheel built successfully with `uv build`.
- All 45 tests passed against the installed wheel on Python 3.12, 3.13, and 3.14.
- Console and module entry points passed outside the repository checkout.
- Wheel contains only the `infercap` package and distribution metadata.
- Core preflight and benchmark function/class ASTs match the pre-refactor revision.
- Blocking flake8 checks, compileall, shell syntax, actionlint, lock consistency,
  and `git diff --check` passed.

GPU runtime execution was not part of this refactor validation. Preflight core
still uses a namespace configuration and contains hardware/runtime integration;
benchmark reporting is still orchestrated by the runner. These remain scoped
follow-up items under the architecture issue.
