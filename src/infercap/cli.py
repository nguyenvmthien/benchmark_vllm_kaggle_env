"""Command-line interface for InferCap."""
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="infercap",
        description="Check inference feasibility and benchmark model endpoints.",
    )
    commands = parser.add_subparsers(dest="command", title="commands", required=True)
    commands.add_parser("check", help="Check model feasibility and serving configuration")
    commands.add_parser("benchmark", help="Benchmark an OpenAI-compatible endpoint")
    if not argv:
        parser.print_help()
        return 0

    # Parse only the command; its existing parser owns all remaining options.
    args = parser.parse_args(argv[:1])
    if args.command == "check":
        from infercap.preflight.cli import main as run_command
    else:
        from infercap.benchmark.cli import main as run_command
    return run_command(argv[1:], prog=f"{parser.prog} {args.command}") or 0


if __name__ == "__main__":
    raise SystemExit(main())
