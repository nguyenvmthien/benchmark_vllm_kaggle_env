# Contributing to InferCap

Thanks for helping improve InferCap. Contributions should stay focused on inference capacity, compatibility, serving, verification, benchmarking, and analysis within the project's documented scope.

## Development setup

InferCap uses [uv](https://docs.astral.sh/uv/) for dependency and environment management:

```bash
git clone <your-fork-url>
cd infercap
uv sync
```

Run the project checks before submitting a pull request:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src scripts tests
git diff --check
```

## Issues and scope

Open or identify an issue before starting a significant feature or refactor. Architecture changes should be discussed in an issue before implementation so responsibilities, compatibility, and migration can be agreed upon.

Issues labeled `good first issue` should be narrowly scoped, have clear acceptance criteria, and require limited project context. Issues labeled `help wanted` are ready for community input but may require more investigation or design discussion. Comment on an issue before beginning substantial work to avoid duplicated effort.

## Pull requests

A pull request should solve one coherent problem and link its issue when applicable. Keep changes reviewable and preserve existing behavior unless the issue explicitly calls for a behavior change.

- Explain what changed and why.
- Add or update tests for behavior changes and bug fixes.
- Document externally visible behavior and API changes.
- Do not mix unrelated refactors or cleanup into a feature pull request.
- Do not claim support for hardware or runtimes that the implementation does not provide.
- Keep hardware-specific and runtime-specific responsibilities isolated.
- Run the test, compile, and whitespace checks above.

If a change affects architecture, public commands, entry points, report formats, or compatibility expectations, call that out explicitly in the pull request.
