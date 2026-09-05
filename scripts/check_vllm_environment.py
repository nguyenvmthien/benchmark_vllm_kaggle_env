from vllm import LLM, SamplingParams
import vllm
import torch

print("vLLM:", vllm.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())