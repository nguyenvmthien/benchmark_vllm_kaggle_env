import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.benchmark.main import BenchmarkConfig, run


class PipelineTests(unittest.IsolatedAsyncioTestCase):
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
            patch("src.benchmark.main.warm_up", new=AsyncMock()),
            patch(
                "src.benchmark.main.run_level",
                new=AsyncMock(return_value=level),
            ),
            patch("src.benchmark.main.plot_report") as plot_report,
        ):
            report_path = await run(config, Path(directory))
            self.assertTrue(report_path.exists())
            plot_report.assert_called_once_with(report_path)


if __name__ == "__main__":
    unittest.main()
