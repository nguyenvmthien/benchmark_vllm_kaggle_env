"""Sample vLLM Prometheus metrics with compatibility across V0 and V1 names."""
from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass, field
import aiohttp

_SAMPLE = re.compile(r"^([^#\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$")
_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
_ALIASES = {
    "usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "hits": ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"),
    "queries": ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total"),
    "hit_rate": ("vllm:gpu_prefix_cache_hit_rate",),
    "preemptions": ("vllm:num_preemptions", "vllm:num_preemptions_total"),
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
}

def parse_prometheus(text: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line.strip())
        if match:
            try:
                metrics.setdefault(match.group(1), []).append(float(match.group(2)))
            except ValueError:
                pass
    return metrics

def _value(metrics: dict[str, list[float]], key: str, *, maximum: bool = False) -> float | None:
    for name in _ALIASES[key]:
        values = metrics.get(name)
        if values:
            return max(values) if maximum else sum(values)
    return None

@dataclass
class KVCacheMonitor:
    metrics_url: str
    interval: float = 0.5
    timeout: float = 2.0
    samples: list[dict[str, float | None]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    cache_config: dict[str, str] = field(default_factory=dict)

    async def sample(self, session: aiohttp.ClientSession) -> None:
        try:
            async with session.get(self.metrics_url, timeout=self.timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                body = await response.text()
                metrics = parse_prometheus(body)
                if not self.cache_config:
                    for line in body.splitlines():
                        if line.startswith("vllm:cache_config_info{"):
                            self.cache_config = dict(_LABEL.findall(line))
                            break
            self.samples.append({
                "usage": _value(metrics, "usage", maximum=True),
                "hits": _value(metrics, "hits"),
                "queries": _value(metrics, "queries"),
                "hit_rate": _value(metrics, "hit_rate", maximum=True),
                "preemptions": _value(metrics, "preemptions"),
                "running": _value(metrics, "running"),
                "waiting": _value(metrics, "waiting"),
            })
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            if not self.errors:
                self.errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            self.ready.set()

    async def run(self, stop: asyncio.Event) -> None:
        async with aiohttp.ClientSession() as session:
            while not stop.is_set():
                await self.sample(session)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
            await self.sample(session)

    def summary(self) -> dict:
        usable = [sample for sample in self.samples if any(v is not None for v in sample.values())]
        if not usable:
            return {"available": False, "samples": 0, "metrics_url": self.metrics_url,
                    "error": self.errors[0] if self.errors else "No supported vLLM metrics found"}
        usage = [float(x["usage"]) for x in usable if x["usage"] is not None]
        running = [float(x["running"]) for x in usable if x["running"] is not None]
        waiting = [float(x["waiting"]) for x in usable if x["waiting"] is not None]

        def delta(key: str) -> float | None:
            values = [float(x[key]) for x in usable if x[key] is not None]
            return max(0.0, values[-1] - values[0]) if len(values) >= 2 else None

        hits, queries, preemptions = delta("hits"), delta("queries"), delta("preemptions")
        legacy_rates = [float(x["hit_rate"]) for x in usable if x["hit_rate"] is not None]
        hit_rate = hits / queries if hits is not None and queries else (
            legacy_rates[-1] if legacy_rates else None)
        return {
            "available": True, "samples": len(usable), "metrics_url": self.metrics_url,
            "cache_config": self.cache_config,
            "prefix_caching_enabled": self.cache_config.get("enable_prefix_caching"),
            "kv_cache_dtype": self.cache_config.get("cache_dtype"),
            "block_size": self.cache_config.get("block_size"),
            "configured_gpu_memory_utilization": self.cache_config.get("gpu_memory_utilization"),
            "avg_usage_pct": 100 * sum(usage) / len(usage) if usage else None,
            "max_usage_pct": 100 * max(usage) if usage else None,
            "prefix_cache_hits": hits, "prefix_cache_queries": queries,
            "prefix_cache_hit_rate_pct": 100 * hit_rate if hit_rate is not None else None,
            "preemptions": preemptions,
            "max_running_requests": max(running) if running else None,
            "max_waiting_requests": max(waiting) if waiting else None,
        }
