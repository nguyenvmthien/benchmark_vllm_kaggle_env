import argparse
import io
import json
import sys
import types
import unittest
import urllib.error
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.scripts.check_runable import (
    Check,
    InferenceRecommendation,
    ModelCandidate,
    check_model_compatibility,
    check_server,
    detect_quantization,
    discover_family,
    main,
    run_checks,
)


def preflight_args(**overrides):
    values = {
        "model": "org/model",
        "family": None,
        "family_candidates": 40,
        "recommend_limit": 5,
        "model_size_b": 7.0,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
        "runner": "auto",
        "profile": "balanced",
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "kv_cache_dtype": "auto",
        "generation_config": "vllm",
        "quantization": None,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "served_model_name": None,
        "min_ram_gib": 8,
        "offline": False,
        "check_server": False,
        "api_url": "http://localhost:8000/v1/chat/completions",
        "timeout": 3,
        "as_json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PreflightReadinessTests(unittest.TestCase):
    def run_with_host(self, args, *, free_mib=24 * 1024, compatibility="PASS"):
        gpu = [{
            "index": 0,
            "name": "Mock GPU",
            "memory_total_mib": 24 * 1024,
            "memory_free_mib": free_mib,
            "compute_capability": "8.0",
        }]
        with (
            patch("src.scripts.check_runable.package_version", return_value="1.0"),
            patch("src.scripts.check_runable.system_memory_gib", return_value=32.0),
            patch("src.scripts.check_runable.query_gpus", return_value=(gpu, None)),
            patch(
                "src.scripts.check_runable.check_model_compatibility",
                return_value=Check("model_compatibility", compatibility, "compatibility result"),
            ),
        ):
            return run_checks(args)

    def test_warning_only_preflight_has_no_failures(self):
        args = preflight_args(model_size_b=None, offline=True)
        checks, _, _ = self.run_with_host(args)
        statuses = {item.name: item.status for item in checks}
        self.assertEqual(statuses["weight_memory"], "WARN")
        self.assertNotIn("FAIL", statuses.values())

    def test_any_failed_check_makes_main_not_ready_and_exit_one(self):
        args = preflight_args(as_json=True)
        checks = [Check("python", "PASS", "3.12"), Check("gpu", "FAIL", "missing")]
        with (
            patch("src.scripts.check_runable.parse_args", return_value=args),
            patch("src.scripts.check_runable.run_checks", return_value=(checks, [], [])),
            redirect_stdout(io.StringIO()) as output,
        ):
            status = main()
        report = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertFalse(report["ready"])
        self.assertIsNone(report["recommended_serve_command"])

    def test_warning_only_main_is_ready_and_exit_zero(self):
        args = preflight_args(as_json=True)
        checks = [Check("weight_memory", "WARN", "unknown")]
        with (
            patch("src.scripts.check_runable.parse_args", return_value=args),
            patch("src.scripts.check_runable.run_checks", return_value=(checks, [], [])),
            redirect_stdout(io.StringIO()) as output,
        ):
            status = main()
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(report["ready"])
        self.assertTrue(report["recommended_serve_command"].startswith("vllm serve org/model"))

    def test_weight_memory_uses_static_formula_and_minimum_free_gpu(self):
        args = preflight_args(model_size_b=7.0, tensor_parallel_size=2)
        gpus = [
            {"index": 0, "name": "A", "memory_total_mib": 16384,
             "memory_free_mib": 12288, "compute_capability": "8.0"},
            {"index": 1, "name": "B", "memory_total_mib": 16384,
             "memory_free_mib": 10240, "compute_capability": "8.0"},
        ]
        with (
            patch("src.scripts.check_runable.package_version", return_value="1.0"),
            patch("src.scripts.check_runable.system_memory_gib", return_value=32.0),
            patch("src.scripts.check_runable.query_gpus", return_value=(gpus, None)),
            patch("src.scripts.check_runable.check_model_compatibility",
                  return_value=Check("model_compatibility", "PASS", "supported")),
        ):
            checks, _, _ = run_checks(args)
        weight = next(item for item in checks if item.name == "weight_memory")
        self.assertEqual(weight.status, "PASS")
        self.assertEqual(
            weight.message,
            "7.000B x 2 bytes x 1.15 / TP=2 = 7.5 GiB/GPU; free=10.0 GiB/GPU",
        )

    def test_explicit_quantization_controls_weight_bytes(self):
        args = preflight_args(model="org/plain-model", model_size_b=20.0, quantization="awq")
        checks, _, _ = self.run_with_host(args, free_mib=12 * 1024)
        weight = next(item for item in checks if item.name == "weight_memory")
        self.assertEqual(weight.status, "PASS")
        self.assertIn("20.000B x 0.5 bytes x 1.15", weight.message)


class ModelResolutionAndRankingTests(unittest.TestCase):
    @patch("src.scripts.check_runable.discover_family")
    def test_bare_model_name_is_resolved_as_family(self, discover):
        candidate = ModelCandidate("Qwen/Qwen-Instruct-AWQ", 7.0, "text-generation",
                                   ["Supported"], "awq", 100, 4.0, True, 1.0)
        discover.return_value = ([candidate], None)
        args = preflight_args(model="Qwen", model_size_b=7.0)
        with (
            patch("src.scripts.check_runable.package_version", return_value="1.0"),
            patch("src.scripts.check_runable.system_memory_gib", return_value=32.0),
            patch("src.scripts.check_runable.query_gpus", return_value=([{
                "index": 0, "name": "GPU", "memory_total_mib": 16384,
                "memory_free_mib": 16384, "compute_capability": "8.0"}], None)),
            patch("src.scripts.check_runable.check_model_compatibility",
                  return_value=Check("model_compatibility", "PASS", "supported")),
        ):
            checks, _, recommendations = run_checks(args)
        discover.assert_called_once_with("Qwen", "float16", 14.4, 40)
        self.assertEqual(args.model, candidate.model_id)
        self.assertEqual(args.quantization, "awq")
        self.assertEqual(recommendations, [candidate])
        self.assertEqual(next(x for x in checks if x.name == "family_discovery").status, "PASS")

    @patch("src.scripts.check_runable.discover_family")
    def test_repository_id_is_treated_as_exact_model(self, discover):
        args = preflight_args(model="Qwen/Qwen-Instruct")
        with (
            patch("src.scripts.check_runable.package_version", return_value="1.0"),
            patch("src.scripts.check_runable.system_memory_gib", return_value=32.0),
            patch("src.scripts.check_runable.query_gpus", return_value=([], "no gpu")),
            patch("src.scripts.check_runable.check_model_compatibility",
                  return_value=Check("model_compatibility", "PASS", "supported")),
        ):
            checks, _, recommendations = run_checks(args)
        discover.assert_not_called()
        self.assertEqual(args.model, "Qwen/Qwen-Instruct")
        self.assertEqual(recommendations, [])
        self.assertNotIn("family_discovery", [item.name for item in checks])

    def test_discovery_filters_and_ranks_current_candidates(self):
        def info(model_id, size_b, *, arch="Supported", downloads=1, tags=None, gated=False):
            return SimpleNamespace(
                id=model_id,
                safetensors=SimpleNamespace(total=size_b * 1e9) if size_b else None,
                config={"architectures": [arch]} if arch else {},
                downloads=downloads,
                tags=tags or [],
                gated=gated,
                pipeline_tag="text-generation",
            )

        candidates = [
            info("Qwen/Qwen-7B-Instruct-AWQ", 7, downloads=1000),
            info("community/Qwen-8B-Chat", 8, downloads=100000),
            info("Qwen/Qwen-30B-Instruct", 30, downloads=1000000),
            info("Qwen/Qwen-unsupported", 1, arch="Unsupported"),
            info("Qwen/tiny-random-Qwen", 1),
            info("Qwen/Qwen-gated", 1, gated=True),
        ]
        api = MagicMock()
        api.list_models.return_value = candidates
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = MagicMock(return_value=api)
        vllm = types.ModuleType("vllm")
        vllm.ModelRegistry = SimpleNamespace(get_supported_archs=lambda: {"Supported"})
        with patch.dict(sys.modules, {"huggingface_hub": hub, "vllm": vllm}):
            results, error = discover_family("Qwen", "float16", 10.0, 40)
        self.assertIsNone(error)
        self.assertEqual([item.model_id for item in results], [
            "Qwen/Qwen-7B-Instruct-AWQ",
            "community/Qwen-8B-Chat",
            "Qwen/Qwen-30B-Instruct",
        ])
        self.assertTrue(results[0].fits)
        self.assertEqual(results[0].quantization, "awq")
        self.assertFalse(results[-1].fits)
        api.list_models.assert_called_once_with(
            search="Qwen", sort="downloads", limit=40, expand=["safetensors", "config"])


class CompatibilityAndQuantizationTests(unittest.TestCase):
    def compatibility(self, config, supported):
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = SimpleNamespace(from_pretrained=MagicMock(return_value=config))
        vllm = types.ModuleType("vllm")
        vllm.ModelRegistry = SimpleNamespace(get_supported_archs=lambda: set(supported))
        with patch.dict(sys.modules, {"transformers": transformers, "vllm": vllm}):
            result = check_model_compatibility("org/model", True)
        transformers.AutoConfig.from_pretrained.assert_called_once_with(
            "org/model", local_files_only=True)
        return result

    def test_supported_multimodal_architecture_is_pass(self):
        config = SimpleNamespace(
            architectures=["SupportedModel"], vision_config={}, audio_config={},
            max_position_embeddings=8192)
        result = self.compatibility(config, {"SupportedModel"})
        self.assertEqual(result.status, "PASS")
        self.assertEqual(
            result.message,
            "architecture=['SupportedModel']; runner=generate; "
            "modalities=text+image/video+audio; model_context=8192",
        )

    def test_unsupported_architecture_is_fail(self):
        result = self.compatibility(SimpleNamespace(architectures=["UnknownModel"]), {"Other"})
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.message, "unsupported architecture(s): ['UnknownModel']")

    def test_config_load_error_is_fail(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = SimpleNamespace(
            from_pretrained=MagicMock(side_effect=OSError("not cached")))
        vllm = types.ModuleType("vllm")
        vllm.ModelRegistry = SimpleNamespace(get_supported_archs=lambda: set())
        with patch.dict(sys.modules, {"transformers": transformers, "vllm": vllm}):
            result = check_model_compatibility("org/model", True)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.message, "validation failed: OSError: not cached")

    def test_current_quantization_normalization_and_ignored_int_names(self):
        self.assertEqual(detect_quantization("org/model", tags=["bnb-4bit"]), "bitsandbytes")
        self.assertIsNone(detect_quantization("org/model-int4"))
        self.assertIsNone(detect_quantization("org/model-int8"))
        self.assertEqual(
            detect_quantization("org/model-AWQ", config={
                "quantization_config": {"quant_method": "GPTQ"}}),
            "gptq",
        )


class JsonAndEndpointTests(unittest.TestCase):
    def test_json_output_shape_is_stable(self):
        args = preflight_args(as_json=True)
        checks = [Check("gpu", "PASS", "one GPU")]
        gpus = [{"index": 0}]
        candidates = [ModelCandidate("org/model", 7, "text-generation", ["Arch"], None, 1)]
        recommendation = InferenceRecommendation("balanced", {"dtype": "float16"}, ["reason"])
        with (
            patch("src.scripts.check_runable.parse_args", return_value=args),
            patch("src.scripts.check_runable.run_checks",
                  return_value=(checks, gpus, candidates)),
            patch("src.scripts.check_runable.inference_recommendation",
                  return_value=recommendation),
            patch("src.scripts.check_runable.serve_command", return_value="vllm serve org/model"),
            redirect_stdout(io.StringIO()) as output,
        ):
            status = main()
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(list(report), [
            "ready", "checks", "gpus", "selected_model", "recommendations",
            "inference_recommendation", "recommended_serve_command",
        ])
        self.assertEqual(set(report["checks"][0]), {
            "name", "status", "message", "recommendation"})
        self.assertEqual(set(report["recommendations"][0]), {
            "model_id", "parameters_b", "pipeline_tag", "architectures",
            "quantization", "downloads", "estimated_weight_gib", "fits", "score",
        })
        self.assertEqual(set(report["inference_recommendation"]), {
            "profile", "flags", "reasons"})

    def response(self, payload):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps(payload).encode()
        return response

    def test_chat_completions_url_is_replaced_with_models_and_matching_model_passes(self):
        response = self.response({"data": [{"id": "org/model"}]})
        with patch("src.scripts.check_runable.urllib.request.urlopen",
                   return_value=response) as urlopen:
            result = check_server(
                "http://localhost:8000/v1/chat/completions", "org/model", 3)
        urlopen.assert_called_once_with("http://localhost:8000/v1/models", timeout=3)
        self.assertEqual(result.status, "PASS")

    def test_base_url_appends_v1_models_and_missing_model_warns(self):
        with patch("src.scripts.check_runable.urllib.request.urlopen",
                   return_value=self.response({"data": [{"id": "other"}]})) as urlopen:
            result = check_server("http://localhost:8000", "org/model", 2)
        urlopen.assert_called_once_with("http://localhost:8000/v1/models", timeout=2)
        self.assertEqual(result.status, "WARN")
        self.assertIn("served=['other']", result.message)

    def test_endpoint_connection_error_is_fail(self):
        with patch("src.scripts.check_runable.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("refused")):
            result = check_server("http://localhost:8000/v1/models", "org/model", 1)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("refused", result.message)


if __name__ == "__main__":
    unittest.main()
