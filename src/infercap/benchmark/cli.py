"""Concurrency, burst, and request-rate benchmarks for an OpenAI-compatible server."""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from infercap.benchmark.runner import BenchmarkConfig, run

def parse_levels(value: str) -> list[int]:
    levels = [int(item) for item in value.split(",")]
    if not levels or any(level < 1 for level in levels) or levels != sorted(set(levels)):
        raise argparse.ArgumentTypeError("levels must be unique positive integers in ascending order")
    return levels

def parse_rates(value: str) -> list[float]:
    rates = [float(item) for item in value.split(",")]
    if not rates or any(rate <= 0 for rate in rates) or rates != sorted(set(rates)):
        raise argparse.ArgumentTypeError("rates must be unique positive numbers in ascending order")
    return rates

def parse_args(argv: list[str] | None = None, *, prog: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--concurrency", type=parse_levels, default=parse_levels("1,4,8,16,32,64"))
    parser.add_argument("--mode", choices=("concurrency", "burst", "request-rate"),
                        default="concurrency")
    parser.add_argument("--burst-sizes", type=parse_levels, default=parse_levels("16,32,64,128"))
    parser.add_argument("--request-rates", type=parse_rates, default=parse_rates("1,2,4,8"),
                        help="Offered requests per second for open-loop mode")
    parser.add_argument("--rate-duration", type=float, default=30.0,
                        help="Seconds to submit traffic at each request-rate level")
    parser.add_argument("--metrics-url", help="vLLM Prometheus URL (default: API origin + /metrics)")
    parser.add_argument("--kv-sample-interval", type=float, default=0.5)
    parser.add_argument("--prompt-profile", choices=("random", "shared-prefix"), default="random")
    parser.add_argument("--shared-prefix-words", type=int, default=1024,
                        help="Approximate shared-prefix length for KV/prefix-cache testing")
    parser.add_argument("--requests-per-worker", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--min-tps-growth", type=float, default=0.10)
    parser.add_argument("--max-ttft-p95", type=float, default=2.0)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"))
    args = parser.parse_args(argv)
    if args.requests_per_worker < 1 or args.max_tokens < 1 or args.warmup_requests < 0:
        parser.error("request counts and --max-tokens must be positive")
    if (args.timeout <= 0 or args.gpu_sample_interval <= 0 or args.kv_sample_interval <= 0
            or args.rate_duration <= 0 or args.shared_prefix_words <= 0):
        parser.error("timeout and GPU sample interval must be positive")
    if not 0 <= args.max_error_rate <= 1 or args.min_tps_growth < 0:
        parser.error("thresholds must be non-negative; error rate must be <= 1")
    return args

def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    args = parse_args(argv, prog=prog)
    api = urlsplit(args.api_url)
    metrics_url = args.metrics_url or urlunsplit((api.scheme, api.netloc, "/metrics", "", ""))
    config = BenchmarkConfig(args.api_url, args.model, args.concurrency,
        args.requests_per_worker, args.max_tokens, args.timeout, args.warmup_requests,
        args.gpu_sample_interval, args.min_tps_growth, args.max_ttft_p95,
        args.max_error_rate, args.seed, args.mode, args.burst_sizes,
        args.request_rates, args.rate_duration, metrics_url, args.kv_sample_interval,
        args.prompt_profile, args.shared_prefix_words)
    asyncio.run(run(config, args.output_dir))

if __name__ == "__main__":
    main()
