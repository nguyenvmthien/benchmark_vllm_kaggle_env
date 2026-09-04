"""Concurrency, burst, and request-rate benchmarks for an OpenAI-compatible server."""
from __future__ import annotations
import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import aiohttp
from src.analyse.metrics import distribution, find_saturation
from src.analyse.visual import plot_report
from src.benchmark.client import fetch_request
from src.benchmark.gpu import GPUMonitor
from src.config.prompts import ALL_PROMPTS

@dataclass(frozen=True)
class BenchmarkConfig:
    api_url: str
    model: str
    concurrency_levels: list[int]
    requests_per_worker: int
    max_tokens: int
    timeout: float
    warmup_requests: int
    gpu_sample_interval: float
    min_tps_growth: float
    max_ttft_p95: float
    max_error_rate: float
    seed: int
    mode: str = "concurrency"
    burst_sizes: list[int] | None = None
    request_rates: list[float] | None = None
    rate_duration: float = 30.0

def summarize(results: list[dict], elapsed: float, load_value: int | float, gpu: dict,
              *, mode: str = "concurrency") -> dict:
    valid = [item for item in results if item["success"]]
    tokens = sum(item["output_tokens"] for item in valid)
    return {
        "concurrency": load_value, "load_value": load_value, "mode": mode,
        "requests": len(results), "successful_requests": len(valid),
        "failed_requests": len(results) - len(valid),
        "error_rate": (len(results) - len(valid)) / len(results) if results else 1.0,
        "duration_seconds": elapsed, "output_tokens": tokens,
        "throughput_tps": tokens / elapsed if elapsed else 0.0,
        "request_throughput_rps": len(valid) / elapsed if elapsed else 0.0,
        "ttft": distribution(x["ttft"] for x in valid),
        "itl": distribution(x["itl"] for x in valid),
        "latency": distribution(x["latency"] for x in valid),
        "gpu": gpu,
        "warnings": build_warnings(results, gpu),
        "errors": [x["error"] for x in results if x["error"]][:10],
    }

def build_warnings(results: list[dict], gpu: dict) -> list[str]:
    warnings: list[str] = []
    successful = sum(x["success"] for x in results)
    if 0 < successful < 20:
        warnings.append("Fewer than 20 successful samples; tail percentiles are unstable.")
    if any(x["token_count_source"] != "usage" for x in results if x["success"]):
        warnings.append("Server did not return stream usage; token counts are SSE-event estimates.")
    if not gpu["available"]:
        warnings.append("GPU telemetry unavailable (nvidia-smi missing or returned no samples).")
    elif gpu["max_vram_utilization_pct"] >= 95:
        warnings.append("VRAM usage reached at least 95%; OOM risk is high.")
    if results and sum(not x["success"] for x in results) / len(results) > 0.01:
        warnings.append("Error rate exceeded 1%.")
    return warnings

async def run_level(config: BenchmarkConfig, level: int, rng: random.Random) -> dict:
    request_count = level * config.requests_per_worker
    prompts = [rng.choice(ALL_PROMPTS) for _ in range(request_count)]
    semaphore = asyncio.Semaphore(level)
    stop = asyncio.Event()
    monitor = GPUMonitor(config.gpu_sample_interval)
    monitor_task = asyncio.create_task(monitor.run(stop))

    async with aiohttp.ClientSession() as session:
        async def bounded(prompt: str) -> dict:
            async with semaphore:
                return await fetch_request(session, config.api_url, config.model, prompt,
                                           config.max_tokens, timeout=config.timeout)
        started = time.perf_counter()
        results = await asyncio.gather(*(bounded(prompt) for prompt in prompts))
        elapsed = time.perf_counter() - started
    stop.set()
    await monitor_task
    summary = summarize(results, elapsed, level, monitor.summary(), mode="concurrency")
    print_level(summary)
    return summary

async def run_scheduled_level(config: BenchmarkConfig, value: int | float,
                              rng: random.Random) -> dict:
    """Run a simultaneous burst or an open-loop request-rate level."""
    if config.mode == "burst":
        request_count = int(value)
        delays = [0.0] * request_count
    else:
        request_count = max(1, int(float(value) * config.rate_duration))
        delays = [index / float(value) for index in range(request_count)]
    prompts = [rng.choice(ALL_PROMPTS) for _ in range(request_count)]
    stop = asyncio.Event()
    monitor = GPUMonitor(config.gpu_sample_interval)
    monitor_task = asyncio.create_task(monitor.run(stop))

    # Avoid aiohttp's default 100-connection queue masking a large server-side burst.
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        started = time.perf_counter()

        async def scheduled(prompt: str, delay: float) -> dict:
            remaining = started + delay - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)
            return await fetch_request(session, config.api_url, config.model, prompt,
                                       config.max_tokens, timeout=config.timeout)

        results = await asyncio.gather(*(
            scheduled(prompt, delay) for prompt, delay in zip(prompts, delays)
        ))
        elapsed = time.perf_counter() - started
    stop.set()
    await monitor_task
    summary = summarize(results, elapsed, value, monitor.summary(), mode=config.mode)
    if config.mode == "request-rate":
        summary["offered_requests_per_second"] = float(value)
        summary["submission_duration_seconds"] = config.rate_duration
    print_level(summary)
    return summary

def print_level(item: dict) -> None:
    label = {"burst": "burst_size", "request-rate": "offered_rps"}.get(
        item.get("mode"), "concurrency")
    print(f"{label}={item['load_value']:<4} success={item['successful_requests']}/{item['requests']} "
          f"TPS={item['throughput_tps']:.2f} RPS={item['request_throughput_rps']:.2f}")
    print(f"  TTFT p50/p95/p99: {item['ttft']['p50']:.3f}/{item['ttft']['p95']:.3f}/{item['ttft']['p99']:.3f}s")
    print(f"  ITL  p50/p95/p99: {item['itl']['p50']:.4f}/{item['itl']['p95']:.4f}/{item['itl']['p99']:.4f}s")
    print(f"  E2E  p50/p95/p99: {item['latency']['p50']:.3f}/{item['latency']['p95']:.3f}/{item['latency']['p99']:.3f}s")
    if item["gpu"]["available"]:
        print(f"  GPU avg/max={item['gpu']['avg_utilization_pct']:.1f}/{item['gpu']['max_utilization_pct']:.0f}% "
              f"VRAM max={item['gpu']['max_vram_used_mb']} MiB")
    for warning in item["warnings"]:
        print(f"  WARNING: {warning}")

async def warm_up(config: BenchmarkConfig) -> None:
    if not config.warmup_requests:
        return
    print(f"Warming up with {config.warmup_requests} request(s)...")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(fetch_request(
            session, config.api_url, config.model, ALL_PROMPTS[index % len(ALL_PROMPTS)],
            min(config.max_tokens, 16), timeout=config.timeout)
            for index in range(config.warmup_requests)))
    failures = [x for x in results if not x["success"]]
    if failures:
        raise RuntimeError(f"Warm-up failed: {failures[0]['error']}")

async def run(config: BenchmarkConfig, output_dir: Path) -> Path:
    await warm_up(config)
    rng = random.Random(config.seed)
    if config.mode == "concurrency":
        levels = [await run_level(config, level, rng) for level in config.concurrency_levels]
    else:
        values = config.burst_sizes if config.mode == "burst" else config.request_rates
        levels = [await run_scheduled_level(config, value, rng) for value in (values or [])]
    saturation = find_saturation(levels, min_tps_growth=config.min_tps_growth,
                                 max_ttft_p95=config.max_ttft_p95,
                                 max_error_rate=config.max_error_rate)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "config": asdict(config), "levels": levels, "saturation": saturation}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    recommendation = {"burst": "burst size", "request-rate": "request rate"}.get(
        config.mode, "concurrency")
    print(f"\nSaturation: {saturation['reason']} Recommended {recommendation}: {saturation['level']}")
    print(f"JSON report: {path}")
    plot_report(path)
    return path

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    if args.requests_per_worker < 1 or args.max_tokens < 1 or args.warmup_requests < 0:
        parser.error("request counts and --max-tokens must be positive")
    if args.timeout <= 0 or args.gpu_sample_interval <= 0 or args.rate_duration <= 0:
        parser.error("timeout and GPU sample interval must be positive")
    if not 0 <= args.max_error_rate <= 1 or args.min_tps_growth < 0:
        parser.error("thresholds must be non-negative; error rate must be <= 1")
    return args

def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(args.api_url, args.model, args.concurrency,
        args.requests_per_worker, args.max_tokens, args.timeout, args.warmup_requests,
        args.gpu_sample_interval, args.min_tps_growth, args.max_ttft_p95,
        args.max_error_rate, args.seed, args.mode, args.burst_sizes,
        args.request_rates, args.rate_duration)
    asyncio.run(run(config, args.output_dir))

if __name__ == "__main__":
    main()
