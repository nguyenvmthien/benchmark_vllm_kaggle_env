# Contributing to InferCap

Thanks for helping improve InferCap. Contributions should stay focused on inference capacity, compatibility, serving, verification, benchmarking, and analysis within the project's documented scope.

## Development setup

InferCap uses [uv](https://docs.astral.sh/uv/) for dependency and environment management:

```bash
git clone <your-fork-url>
cd infercap
uv sync
```

The Python package lives in `src/infercap/`. Install it with `uv sync` before running tests; tests import the installed `infercap` package.

Run the project checks before submitting a pull request:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src scripts tests
uvx flake8 src scripts tests --count --select=E9,F63,F7,F82 --show-source --statistics
git diff --check
```

The repository does not currently configure a type checker. The CI workflow also runs a non-blocking style report; keep new code within its reported line-length and complexity guidance where practical.

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

## Releasing to PyPI

Releases are published by GitHub Actions through PyPI Trusted Publishing. The one-time PyPI setup must trust this repository and workflow file: `nguyenvmthien/infercap` and `.github/workflows/publish.yml`.

Before each release, update the package version in `pyproject.toml` to a version that has not been uploaded before, then refresh the lockfile and run the checks:

```bash
uv lock
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src scripts tests
rm -rf dist
uv build
uvx twine check dist/*
```

Commit and push the version change. After CI passes, create a GitHub Release using the matching `v<version>` tag (for example, version `0.2.1` uses tag `v0.2.1`). Publishing starts when the release is marked published. PyPI versions are immutable, so never reuse an uploaded version.
