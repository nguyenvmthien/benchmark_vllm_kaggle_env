import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.analyse.visual import plot_report


FIXTURE = Path(__file__).parent / "fixtures" / "benchmark_report_v1.json"

TOP_LEVEL_FIELDS = {"schema_version", "created_at", "config", "levels", "saturation"}
CONFIG_FIELDS = {
    "api_url", "model", "concurrency_levels", "requests_per_worker", "max_tokens",
    "timeout", "warmup_requests", "gpu_sample_interval", "min_tps_growth",
    "max_ttft_p95", "max_error_rate", "seed", "mode", "burst_sizes",
    "request_rates", "rate_duration", "metrics_url", "kv_sample_interval",
    "prompt_profile", "shared_prefix_words",
}
LEVEL_FIELDS = {
    "concurrency", "load_value", "mode", "requests", "successful_requests",
    "failed_requests", "error_rate", "duration_seconds", "output_tokens",
    "throughput_tps", "request_throughput_rps", "ttft", "itl", "latency",
    "gpu", "kv_cache", "warnings", "errors",
}
DISTRIBUTION_FIELDS = {"mean", "p50", "p95", "p99"}
GPU_FIELDS = {
    "available", "samples", "avg_utilization_pct", "max_utilization_pct",
    "max_vram_used_mb", "max_vram_utilization_pct",
}
KV_CACHE_FIELDS = {
    "available", "samples", "metrics_url", "cache_config",
    "prefix_caching_enabled", "kv_cache_dtype", "block_size",
    "configured_gpu_memory_utilization", "avg_usage_pct", "max_usage_pct",
    "prefix_cache_hits", "prefix_cache_queries", "prefix_cache_hit_rate_pct",
    "preemptions", "max_running_requests", "max_waiting_requests",
}


class BenchmarkReportSchemaV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_required_top_level_contract_and_version(self):
        self.assertEqual(self.report["schema_version"], 1)
        self.assertTrue(TOP_LEVEL_FIELDS <= self.report.keys())
        self.assertTrue(CONFIG_FIELDS <= self.report["config"].keys())
        self.assertTrue(self.report["levels"])
        datetime.fromisoformat(self.report["created_at"])

    def test_required_per_load_level_contract(self):
        for level in self.report["levels"]:
            with self.subTest(load=level["load_value"]):
                self.assertTrue(LEVEL_FIELDS <= level.keys())
                for metric in ("ttft", "itl", "latency"):
                    self.assertEqual(set(level[metric]), DISTRIBUTION_FIELDS)
                self.assertEqual(
                    level["requests"],
                    level["successful_requests"] + level["failed_requests"],
                )
                self.assertIsInstance(level["warnings"], list)
                self.assertIsInstance(level["errors"], list)

    def test_current_gpu_and_kv_cache_telemetry_contract(self):
        for level in self.report["levels"]:
            with self.subTest(load=level["load_value"]):
                self.assertTrue(level["gpu"]["available"])
                self.assertEqual(set(level["gpu"]), GPU_FIELDS)
                self.assertTrue(level["kv_cache"]["available"])
                self.assertEqual(set(level["kv_cache"]), KV_CACHE_FIELDS)
                self.assertIsInstance(level["kv_cache"]["cache_config"], dict)

    def test_saturation_recommendation_contract(self):
        saturation = self.report["saturation"]
        self.assertEqual(
            set(saturation), {"found", "level", "trigger_level", "reason"}
        )
        self.assertTrue(saturation["found"])
        loads = [level["load_value"] for level in self.report["levels"]]
        self.assertIn(saturation["level"], loads)
        self.assertIn(saturation["trigger_level"], loads)
        self.assertIsInstance(saturation["reason"], str)
        self.assertTrue(saturation["reason"])

    def test_plot_report_consumes_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.png"
            result = plot_report(FIXTURE, output)
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_fixture_represents_static_viewer_field_assumptions(self):
        # app.js selects a best level, resolves the recommendation by load, and
        # renders throughput, latency, request health, and available GPU telemetry.
        levels = self.report["levels"]
        best = max(levels, key=lambda level: level["throughput_tps"])
        recommended = next(
            level for level in levels
            if level["load_value"] == self.report["saturation"]["level"]
        )
        self.assertGreater(best["request_throughput_rps"], 0)
        self.assertGreaterEqual(recommended["ttft"]["p95"], 0)
        self.assertGreaterEqual(recommended["latency"]["p95"], 0)
        self.assertTrue(any(level["gpu"]["available"] for level in levels))
        self.assertEqual(
            sum(level["failed_requests"] for level in levels), 1
        )


if __name__ == "__main__":
    unittest.main()
