import unittest
from src.scripts.check_runable import resolve_model_size_b

class ModelSizeTests(unittest.TestCase):
    def test_explicit_override_does_not_need_hub(self):
        self.assertEqual(resolve_model_size_b("any/model", 7.25, False), (7.25, "CLI override"))

    def test_offline_without_override_is_explained(self):
        self.assertEqual(resolve_model_size_b("any/model", None, True), (None, "offline mode"))

if __name__ == "__main__":
    unittest.main()
