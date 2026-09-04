import argparse
import unittest
from src.scripts.check_runable import (
    detect_quantization, infer_runner, inference_recommendation,
    resolve_model_size_b, serve_command,
)

class ModelSizeTests(unittest.TestCase):
    def test_explicit_override_does_not_need_hub(self):
        self.assertEqual(resolve_model_size_b("any/model", 7.25, False), (7.25, "CLI override"))

    def test_offline_without_override_is_explained(self):
        self.assertEqual(resolve_model_size_b("any/model", None, True), (None, "offline mode"))

    def test_quantization_is_detected_from_id_or_config(self):
        self.assertEqual(detect_quantization("org/Qwen-AWQ"), "awq")
        self.assertEqual(detect_quantization("org/model", config={
            "quantization_config": {"quant_method": "GPTQ"}}), "gptq")

    def test_runner_covers_generation_pooling_and_audio(self):
        self.assertEqual(infer_runner("text-generation", ["Qwen3ForCausalLM"]), "generate")
        self.assertEqual(infer_runner("feature-extraction", ["Qwen2Model"]), "pooling")
        self.assertEqual(infer_runner(None, ["WhisperForConditionalGeneration"]), "generate")

    def test_balanced_profile_adds_inference_flags(self):
        args = argparse.Namespace(
            model="Qwen/Qwen2.5-7B-Instruct-AWQ", profile="balanced",
            max_num_seqs=None, max_num_batched_tokens=None, tensor_parallel_size=2,
            dtype="float16", gpu_memory_utilization=.9, max_model_len=4096,
            generation_config="vllm", kv_cache_dtype="auto",
            enable_prefix_caching=True, enable_chunked_prefill=True,
            quantization=None, runner="auto", served_model_name=None)
        rec = inference_recommendation(args)
        self.assertEqual(rec.flags["max-num-seqs"], 64)
        self.assertEqual(rec.flags["quantization"], "awq")
        command = serve_command(args, rec)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertIn("--generation-config vllm", command)

if __name__ == "__main__":
    unittest.main()
