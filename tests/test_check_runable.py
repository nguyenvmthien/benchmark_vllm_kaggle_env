import unittest
from src.scripts.check_runable import (
    detect_quantization, infer_runner, resolve_model_size_b,
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
        self.assertEqual(infer_runner(None, ["WhisperForConditionalGeneration"]), "transcription")

if __name__ == "__main__":
    unittest.main()
