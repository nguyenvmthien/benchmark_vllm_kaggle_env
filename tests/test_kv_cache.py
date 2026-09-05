import unittest

from infercap.benchmark.vllm_metrics import KVCacheMonitor, parse_prometheus


class KVCacheMetricsTests(unittest.TestCase):
    def test_parser_supports_labels_colons_and_scientific_values(self):
        metrics = parse_prometheus(
            '# HELP ignored help\n'
            'vllm:kv_cache_usage_perc{model_name="m"} 7.5e-1\n'
            'vllm:num_requests_waiting{model_name="m"} 3\n'
            'vllm:cache_config_info{block_size="16",enable_prefix_caching="True"} 1\n'
        )
        self.assertEqual(metrics["vllm:kv_cache_usage_perc"], [0.75])
        self.assertEqual(metrics["vllm:num_requests_waiting"], [3.0])

    def test_summary_uses_counter_deltas_and_peak_gauges(self):
        monitor = KVCacheMonitor("http://localhost/metrics")
        monitor.samples = [
            {"usage": .2, "hits": 100, "queries": 200, "hit_rate": None,
             "preemptions": 4, "running": 1, "waiting": 0},
            {"usage": .8, "hits": 160, "queries": 300, "hit_rate": None,
             "preemptions": 6, "running": 8, "waiting": 5},
        ]
        result = monitor.summary()
        self.assertEqual(result["max_usage_pct"], 80)
        self.assertEqual(result["prefix_cache_hits"], 60)
        self.assertEqual(result["prefix_cache_queries"], 100)
        self.assertEqual(result["prefix_cache_hit_rate_pct"], 60)
        self.assertEqual(result["preemptions"], 2)
        self.assertEqual(result["max_waiting_requests"], 5)

    def test_unavailable_is_not_reported_as_zero(self):
        result = KVCacheMonitor("http://localhost/metrics").summary()
        self.assertFalse(result["available"])
        self.assertNotIn("max_usage_pct", result)


if __name__ == "__main__":
    unittest.main()
