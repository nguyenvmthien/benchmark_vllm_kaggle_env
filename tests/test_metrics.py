import unittest
from infercap.analysis.metrics import distribution, find_saturation, percentile

class MetricsTests(unittest.TestCase):
    def test_percentiles_interpolate(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(distribution([1, 2, 3])["p95"], 2.9)

    def test_empty_distribution_is_stable(self):
        self.assertEqual(distribution([]), {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0})

    def test_saturation_on_flat_throughput(self):
        levels = [
            {"concurrency": 1, "throughput_tps": 100, "error_rate": 0, "ttft": {"p95": .1}},
            {"concurrency": 2, "throughput_tps": 105, "error_rate": 0, "ttft": {"p95": .2}},
        ]
        result = find_saturation(levels, min_tps_growth=.1)
        self.assertTrue(result["found"])
        self.assertEqual(result["level"], 1)

    def test_saturation_on_slo(self):
        levels = [{"concurrency": 4, "throughput_tps": 100, "error_rate": 0,
                   "ttft": {"p95": 2.5}}]
        self.assertTrue(find_saturation(levels, max_ttft_p95=2)["found"])

if __name__ == "__main__":
    unittest.main()
