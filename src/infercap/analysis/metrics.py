"""Pure functions used to aggregate benchmark measurements."""
from __future__ import annotations
import math
from statistics import mean
from typing import Iterable, Sequence

def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def distribution(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values if value is not None]
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {"mean": mean(samples), "p50": percentile(samples, 50),
            "p95": percentile(samples, 95), "p99": percentile(samples, 99)}

def calculate_percentiles(data: Iterable[float], metric_name: str | None = None) -> dict[str, float]:
    """Backward-compatible alias."""
    return distribution(data)

def find_saturation(levels: Sequence[dict], *, min_tps_growth: float = 0.10,
                    max_ttft_p95: float = 2.0, max_error_rate: float = 0.01) -> dict:
    """Locate the last healthy level before throughput flattens or an SLO fails."""
    if not levels:
        return {"found": False, "level": None, "reason": "No successful benchmark levels."}
    for index, current in enumerate(levels):
        reasons: list[str] = []
        if current.get("error_rate", 0.0) > max_error_rate:
            reasons.append(f"error rate {current['error_rate']:.1%} > {max_error_rate:.1%}")
        if current.get("ttft", {}).get("p95", 0.0) > max_ttft_p95:
            reasons.append(f"TTFT p95 {current['ttft']['p95']:.3f}s > {max_ttft_p95:.3f}s")
        if index:
            previous_tps = levels[index - 1].get("throughput_tps", 0.0)
            if previous_tps <= 0:
                continue
            growth = (current.get("throughput_tps", 0.0) - previous_tps) / previous_tps
            current["tps_growth"] = growth
            if growth < min_tps_growth:
                reasons.append(f"TPS growth {growth:.1%} < {min_tps_growth:.1%}")
        if reasons:
            return {"found": True, "level": levels[max(0, index - 1)]["concurrency"],
                    "trigger_level": current["concurrency"], "reason": "; ".join(reasons)}
    return {"found": False, "level": levels[-1]["concurrency"],
            "reason": "No saturation detected in tested concurrency range."}
