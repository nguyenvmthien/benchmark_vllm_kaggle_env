#!/usr/bin/env python3
"""Preflight checks for serving a model with vLLM.

This script never imports vLLM or torch just to print their versions, so it is
safe on partially configured hosts. Exit code: 0=ready/warnings, 1=hard failure.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class Check:
    name: str
    status: str
    message: str
    recommendation: str | None = None

def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def system_memory_gib() -> float:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0])
        return fields["MemAvailable"] / 1024**2
    except (OSError, KeyError, ValueError):
        return 0.0

def query_gpus() -> tuple[list[dict], str | None]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        gpus = []
        for line in result.stdout.strip().splitlines():
            index, name, total, free, capability = (part.strip() for part in line.split(",", 4))
            gpus.append({"index": int(index), "name": name, "memory_total_mib": int(total),
                         "memory_free_mib": int(free), "compute_capability": capability})
        return gpus, None
    except FileNotFoundError:
        return [], "nvidia-smi is not installed or not on PATH"
    except (subprocess.SubprocessError, ValueError) as exc:
        return [], f"nvidia-smi failed: {exc}"

def check_server(url: str, model: str, timeout: float) -> Check:
    models_url = url.rstrip("/")
    if models_url.endswith("/chat/completions"):
        models_url = models_url[:-len("/chat/completions")] + "/models"
    elif not models_url.endswith("/models"):
        models_url += "/v1/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as response:
            payload = json.load(response)
        ids = [item.get("id") for item in payload.get("data", [])]
        if model in ids:
            return Check("api_server", "PASS", f"reachable; model {model!r} is served")
        return Check("api_server", "WARN", f"reachable, but model not found; served={ids}",
                     "Use the served model id or start the requested model.")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return Check("api_server", "FAIL", f"unreachable: {exc}",
                     "Start vLLM first, then repeat this check.")

def resolve_model_size_b(model: str, override: float | None, offline: bool) -> tuple[float | None, str]:
    if override is not None:
        return override, "CLI override"
    if offline:
        return None, "offline mode"
    try:
        from huggingface_hub import model_info
        info = model_info(model, files_metadata=False)
        total = getattr(getattr(info, "safetensors", None), "total", None)
        if total:
            return float(total) / 1e9, "Hugging Face safetensors metadata"
        return None, "Hub metadata has no safetensors parameter count"
    except Exception as exc:
        return None, f"Hub lookup failed: {type(exc).__name__}: {exc}"

def check_model_compatibility(model: str, offline: bool) -> Check:
    try:
        from transformers import AutoConfig
        from vllm import ModelRegistry
        config = AutoConfig.from_pretrained(model, local_files_only=offline)
        architectures = list(getattr(config, "architectures", None) or [])
        supported = sorted(set(architectures) & ModelRegistry.get_supported_archs())
        if supported:
            return Check("model_compatibility", "PASS", f"supported architecture(s): {supported}")
        return Check("model_compatibility", "FAIL", f"unsupported architecture(s): {architectures}",
                     "Choose a vLLM-supported model architecture or upgrade vLLM.")
    except Exception as exc:
        return Check("model_compatibility", "FAIL", f"validation failed: {type(exc).__name__}: {exc}",
                     "Cache the model config locally or run without --offline.")

def dtype_bytes(dtype: str) -> float:
    return {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1,
            "float8": 1, "int4": 0.5}.get(dtype, 2)

def run_checks(args: argparse.Namespace) -> tuple[list[Check], list[dict]]:
    checks: list[Check] = []
    python_ok = sys.version_info >= (3, 10)
    checks.append(Check("python", "PASS" if python_ok else "FAIL", platform.python_version(),
                        None if python_ok else "Install Python 3.10 or newer."))
    vllm_version, torch_version = package_version("vllm"), package_version("torch")
    checks.append(Check("vllm", "PASS" if vllm_version else "FAIL", vllm_version or "not installed",
                        None if vllm_version else "Install the vllm package."))
    checks.append(Check("torch", "PASS" if torch_version else "FAIL", torch_version or "not installed",
                        None if torch_version else "Install a CUDA-compatible PyTorch build."))
    ram = system_memory_gib()
    checks.append(Check("system_memory", "PASS" if ram >= args.min_ram_gib else "WARN",
                        f"{ram:.1f} GiB available (minimum target {args.min_ram_gib:.1f} GiB)",
                        None if ram >= args.min_ram_gib else "Free RAM or use a smaller/quantized model."))

    gpus, gpu_error = query_gpus()
    if gpu_error:
        checks.append(Check("gpu", "FAIL", gpu_error, "Install NVIDIA drivers and verify nvidia-smi."))
    elif len(gpus) < args.tensor_parallel_size:
        checks.append(Check("gpu", "FAIL", f"{len(gpus)} GPU(s), TP={args.tensor_parallel_size}",
                            "Reduce tensor parallel size or provide more GPUs."))
    else:
        detail = "; ".join(f"GPU {g['index']} {g['name']} free={g['memory_free_mib']} MiB"
                           for g in gpus)
        checks.append(Check("gpu", "PASS", detail))

    model_size_b, size_source = resolve_model_size_b(args.model, args.model_size_b, args.offline)
    if model_size_b:
        checks.append(Check("model_size", "PASS", f"{model_size_b:.3f}B parameters ({size_source})"))
        weight_gib = model_size_b * 1e9 * dtype_bytes(args.dtype) / 1024**3
        per_gpu = weight_gib * 1.15 / args.tensor_parallel_size
        usable = min((g["memory_free_mib"] / 1024 for g in gpus),
                     default=0)
        status = "PASS" if usable >= per_gpu else "FAIL"
        checks.append(Check("weight_memory", status,
            f"estimated {per_gpu:.1f} GiB/GPU including 15% overhead; budget {usable:.1f} GiB/GPU",
            None if status == "PASS" else "Use quantization, more GPUs, or a smaller model. "
            "KV cache memory is additional and workload-dependent."))
    else:
        checks.append(Check("weight_memory", "WARN", f"not detected ({size_source})",
                            "Check Hub access or override with --model-size-b."))

    if args.validate_model:
        checks.append(check_model_compatibility(args.model, args.offline))
    if args.check_server:
        checks.append(check_server(args.api_url, args.model, args.timeout))
    return checks, gpus

def serve_command(args: argparse.Namespace) -> str:
    parts = ["vllm", "serve", args.model, "--tensor-parallel-size", str(args.tensor_parallel_size),
             "--dtype", args.dtype, "--gpu-memory-utilization", str(args.gpu_memory_utilization),
             "--max-model-len", str(args.max_model_len)]
    return " ".join(parts)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model-size-b", type=float,
                        help="Optional parameter-count override in billions; normally auto-detected.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "int8", "float8", "int4"],
                        default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--min-ram-gib", type=float, default=8)
    parser.add_argument("--validate-model", action="store_true",
                        help="Validate the Hugging Face architecture against vLLM ModelRegistry.")
    parser.add_argument("--offline", action="store_true",
                        help="Do not query Hugging Face Hub; model size may require an override.")
    parser.add_argument("--check-server", action="store_true")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.model_size_b is not None and args.model_size_b <= 0:
        parser.error("--model-size-b must be positive")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args

def main() -> int:
    args = parse_args()
    checks, gpus = run_checks(args)
    failed = any(item.status == "FAIL" for item in checks)
    report = {"ready": not failed, "checks": [asdict(item) for item in checks],
              "gpus": gpus, "recommended_serve_command": serve_command(args)}
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        for item in checks:
            print(f"[{item.status:4}] {item.name}: {item.message}")
            if item.recommendation:
                print(f"       -> {item.recommendation}")
        print(f"\nSuggested command:\n{report['recommended_serve_command']}")
        print(f"\nResult: {'READY' if report['ready'] else 'NOT READY'}")
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
