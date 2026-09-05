import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from infercap.benchmark.runner import BenchmarkConfig, run
from infercap.benchmark.cli import parse_rates


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_rates_accepts_fractional_open_loop_rates(self):
        self.assertEqual(parse_rates("0.5,2,10"), [0.5, 2.0, 10.0])

    async def test_request_rate_mode_uses_open_loop_runner(self):
        config = BenchmarkConfig(
            api_url="http://localhost/v1/chat/completions", model="test-model",
            concurrency_levels=[1], requests_per_worker=1, max_tokens=1, timeout=1,
            warmup_requests=0, gpu_sample_interval=1, min_tps_growth=0.1,
            max_ttft_p95=2, max_error_rate=0.01, seed=42,
            mode="request-rate", request_rates=[2.0], rate_duration=1,
        )
        level = {"concurrency": 2.0, "load_value": 2.0, "mode": "request-rate",
                 "throughput_tps": 1, "error_rate": 0, "ttft": {"p95": 0.1}}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("infercap.benchmark.runner.warm_up", new=AsyncMock()),
            patch("infercap.benchmark.runner.run_scheduled_level",
                  new=AsyncMock(return_value=level)) as scheduled,
            patch("infercap.benchmark.runner.plot_report"),
        ):
            await run(config, Path(directory))
            scheduled.assert_awaited_once()
            self.assertEqual(scheduled.await_args.args[1], 2.0)

    async def test_run_writes_report_then_renders_dashboard(self):
        config = BenchmarkConfig(
            api_url="http://localhost/v1/chat/completions",
            model="test-model",
            concurrency_levels=[1],
            requests_per_worker=1,
            max_tokens=1,
            timeout=1,
            warmup_requests=0,
            gpu_sample_interval=1,
            min_tps_growth=0.1,
            max_ttft_p95=2,
            max_error_rate=0.01,
            seed=42,
        )
        level = {
            "concurrency": 1,
            "throughput_tps": 1,
            "error_rate": 0,
            "ttft": {"p95": 0.1},
        }

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("infercap.benchmark.runner.warm_up", new=AsyncMock()),
            patch(
                "infercap.benchmark.runner.run_level",
                new=AsyncMock(return_value=level),
            ),
            patch("infercap.benchmark.runner.plot_report") as plot_report,
        ):
            report_path = await run(config, Path(directory))
            self.assertTrue(report_path.exists())
            plot_report.assert_called_once_with(report_path)


if __name__ == "__main__":
    unittest.main()
