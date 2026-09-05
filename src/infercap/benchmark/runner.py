"""Concurrency, burst, and request-rate benchmarks for an OpenAI-compatible server."""
from __future__ import annotations
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import aiohttp
from infercap.analysis.metrics import distribution, find_saturation
from infercap.analysis.plotting import plot_report
from infercap.benchmark.client import fetch_request
from infercap.benchmark.nvidia import GPUMonitor
from infercap.benchmark.vllm_metrics import KVCacheMonitor
from infercap.config.prompts import ALL_PROMPTS

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
    metrics_url: str | None = None
    kv_sample_interval: float = 0.5
    prompt_profile: str = "random"
    shared_prefix_words: int = 1024

def summarize(results: list[dict], elapsed: float, load_value: int | float, gpu: dict,
              kv_cache: dict, *, mode: str = "concurrency") -> dict:
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
        "gpu": gpu, "kv_cache": kv_cache,
        "warnings": build_warnings(results, gpu, kv_cache),
        "errors": [x["error"] for x in results if x["error"]][:10],
    }

def build_warnings(results: list[dict], gpu: dict, kv_cache: dict) -> list[str]:
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
    if not kv_cache["available"]:
        warnings.append(f"vLLM KV-cache metrics unavailable: {kv_cache.get('error', 'unknown error')}.")
    elif kv_cache.get("preemptions"):
        warnings.append(f"KV-cache pressure caused {kv_cache['preemptions']:.0f} preemption(s).")
    if kv_cache.get("prefix_caching_enabled", "").lower() == "false":
        warnings.append("vLLM reports automatic prefix caching is disabled.")
    if results and sum(not x["success"] for x in results) / len(results) > 0.01:
        warnings.append("Error rate exceeded 1%.")
    return warnings

def make_prompts(config: BenchmarkConfig, count: int, rng: random.Random) -> list[str]:
    if config.prompt_profile == "random":
        return [rng.choice(ALL_PROMPTS) for _ in range(count)]
    prefix_seed = (
        "You are reviewing a production inference system. Preserve all relevant context, "
        "constraints, terminology, and operational assumptions before answering. "
    )
    words = prefix_seed.split()
    shared_prefix = " ".join(words[index % len(words)] for index in range(config.shared_prefix_words))
    return [f"{shared_prefix}\n\nRequest {index}: {rng.choice(ALL_PROMPTS)}" for index in range(count)]

async def start_monitors(config: BenchmarkConfig) -> tuple[asyncio.Event, GPUMonitor, KVCacheMonitor,
                                                        list[asyncio.Task]]:
    stop = asyncio.Event()
    gpu = GPUMonitor(config.gpu_sample_interval)
    kv = KVCacheMonitor(config.metrics_url or "", config.kv_sample_interval)
    tasks = [asyncio.create_task(gpu.run(stop)), asyncio.create_task(kv.run(stop))]
    await kv.ready.wait()
    return stop, gpu, kv, tasks

async def stop_monitors(stop: asyncio.Event, tasks: list[asyncio.Task]) -> None:
    stop.set()
    await asyncio.gather(*tasks)

def add_prefix_comparison(summary: dict, results: list[dict], initial_count: int) -> None:
    initial = [x for x in results[:initial_count] if x["success"]]
    repeated = [x for x in results[initial_count:] if x["success"]]
    if not initial or not repeated:
        return
    initial_ttft = distribution(x["ttft"] for x in initial)
    repeated_ttft = distribution(x["ttft"] for x in repeated)
    baseline = initial_ttft["mean"]
    summary["shared_prefix_comparison"] = {
        "initial_requests": len(initial), "repeated_requests": len(repeated),
        "initial_ttft": initial_ttft, "repeated_ttft": repeated_ttft,
        "mean_ttft_reduction_pct": (
            100 * (baseline - repeated_ttft["mean"]) / baseline if baseline else None),
    }

async def run_level(config: BenchmarkConfig, level: int, rng: random.Random) -> dict:
    request_count = level * config.requests_per_worker
    prompts = make_prompts(config, request_count, rng)
    semaphore = asyncio.Semaphore(level)
    stop, monitor, kv_monitor, monitor_tasks = await start_monitors(config)

    async with aiohttp.ClientSession() as session:
        async def bounded(prompt: str) -> dict:
            async with semaphore:
                return await fetch_request(session, config.api_url, config.model, prompt,
                                           config.max_tokens, timeout=config.timeout)
        started = time.perf_counter()
        results = await asyncio.gather(*(bounded(prompt) for prompt in prompts))
        elapsed = time.perf_counter() - started
    await stop_monitors(stop, monitor_tasks)
    summary = summarize(results, elapsed, level, monitor.summary(), kv_monitor.summary(),
                        mode="concurrency")
    if config.prompt_profile == "shared-prefix":
        add_prefix_comparison(summary, results, level)
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
    prompts = make_prompts(config, request_count, rng)
    stop, monitor, kv_monitor, monitor_tasks = await start_monitors(config)

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
    await stop_monitors(stop, monitor_tasks)
    summary = summarize(results, elapsed, value, monitor.summary(), kv_monitor.summary(),
                        mode=config.mode)
    if config.prompt_profile == "shared-prefix":
        initial_count = int(value) if config.mode == "burst" else 1
        add_prefix_comparison(summary, results, initial_count)
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
    kv = item["kv_cache"]
    if kv["available"]:
        hit_rate = kv.get("prefix_cache_hit_rate_pct")
        print(f"  KV cache avg/max={kv.get('avg_usage_pct')!s}/{kv.get('max_usage_pct')!s}% "
              f"hit-rate={f'{hit_rate:.1f}%' if hit_rate is not None else 'n/a'} "
              f"preemptions={kv.get('preemptions')} waiting-max={kv.get('max_waiting_requests')}")
    comparison = item.get("shared_prefix_comparison")
    if comparison:
        reduction = comparison["mean_ttft_reduction_pct"]
        print(f"  Shared prefix initial/repeated TTFT mean="
              f"{comparison['initial_ttft']['mean']:.3f}/{comparison['repeated_ttft']['mean']:.3f}s "
              f"reduction={f'{reduction:.1f}%' if reduction is not None else 'n/a'}")
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
