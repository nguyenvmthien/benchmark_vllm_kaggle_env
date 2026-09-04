"""Asynchronous NVIDIA GPU telemetry collection."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field

@dataclass
class GPUMonitor:
    interval: float = 0.5
    samples: list[dict] = field(default_factory=list)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                process = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL)
                stdout, _ = await process.communicate()
                if process.returncode == 0:
                    timestamp = asyncio.get_running_loop().time()
                    for line in stdout.decode().strip().splitlines():
                        index, util, used, total = (int(value.strip()) for value in line.split(","))
                        self.samples.append({"timestamp": timestamp, "gpu": index,
                                             "utilization_pct": util, "vram_used_mb": used,
                                             "vram_total_mb": total})
            except (FileNotFoundError, ValueError):
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False, "samples": 0, "avg_utilization_pct": 0.0,
                    "max_utilization_pct": 0.0, "max_vram_used_mb": 0,
                    "max_vram_utilization_pct": 0.0}
        return {"available": True, "samples": len(self.samples),
                "avg_utilization_pct": sum(x["utilization_pct"] for x in self.samples) / len(self.samples),
                "max_utilization_pct": max(x["utilization_pct"] for x in self.samples),
                "max_vram_used_mb": max(x["vram_used_mb"] for x in self.samples),
                "max_vram_utilization_pct": max(
                    100 * x["vram_used_mb"] / x["vram_total_mb"] for x in self.samples if x["vram_total_mb"])}
