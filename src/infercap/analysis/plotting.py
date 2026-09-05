"""Render benchmark JSON as a compact dashboard."""
from __future__ import annotations
import json
from pathlib import Path

def plot_report(report_path: str | Path, output_path: str | Path | None = None) -> Path:
    import matplotlib.pyplot as plt
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    levels = report["levels"]
    mode = report.get("config", {}).get("mode", "concurrency")
    x_values = [x.get("load_value", x["concurrency"]) for x in levels]
    x_label = {"burst": "burst size", "request-rate": "offered requests/s"}.get(
        mode, "concurrency")
    output_path = Path(output_path) if output_path else report_path.with_suffix(".png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(x_values, [x["throughput_tps"] for x in levels], "o-", label="Output TPS")
    ax.set_title("Total throughput"); ax.set_ylabel("token/s"); ax.grid(alpha=.3)
    ax = axes[0, 1]
    for key in ("p50", "p95", "p99"):
        ax.plot(x_values, [x["ttft"][key] for x in levels], "o-", label=key)
    ax.axhline(report["config"]["max_ttft_p95"], color="red", ls=":", label="p95 SLO")
    ax.set_title("Time to first token"); ax.set_ylabel("seconds"); ax.legend(); ax.grid(alpha=.3)
    ax = axes[1, 0]
    for key in ("p50", "p95", "p99"):
        ax.plot(x_values, [x["itl"][key] * 1000 for x in levels], "o-", label=key)
    ax.set_title("Inter-token/decode latency"); ax.set_ylabel("milliseconds"); ax.legend(); ax.grid(alpha=.3)
    ax = axes[1, 1]
    has_gpu = any(x["gpu"]["available"] for x in levels)
    has_kv = any(x.get("kv_cache", {}).get("available") for x in levels)
    if has_gpu:
        ax.plot(x_values, [x["gpu"]["avg_utilization_pct"] for x in levels], "o-", label="GPU avg %")
        ax.plot(x_values, [x["gpu"]["max_vram_utilization_pct"] for x in levels], "s--", label="VRAM max %")
    if has_kv:
        ax.plot(x_values, [x.get("kv_cache", {}).get("max_usage_pct") or 0 for x in levels],
                "d-", label="KV cache max %")
    if has_gpu or has_kv:
        ax.set_ylim(0, 105); ax.legend()
    else:
        ax.text(.5, .5, "GPU and KV telemetry unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("GPU / VRAM / KV-cache utilization"); ax.set_ylabel("percent"); ax.grid(alpha=.3)
    for ax in axes.flat:
        ax.set_xlabel(x_label)
    saturation = report["saturation"]
    warning_count = sum(len(item.get("warnings", [])) for item in levels)
    color = "#ffd6d6" if saturation["found"] else "#fff0bd"
    fig.text(.5, .01, f"NOTICE: {saturation['reason']} Recommended {x_label}: {saturation['level']} | {warning_count} warning(s) in JSON",
             ha="center", bbox={"facecolor": color, "edgecolor": "#aa5555", "pad": 6})
    fig.suptitle(f"vLLM {mode} benchmark — {report['config']['model']}")
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"PNG dashboard: {output_path}")
    return output_path
