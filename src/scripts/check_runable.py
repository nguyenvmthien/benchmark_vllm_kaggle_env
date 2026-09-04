#!/usr/bin/env python3
"""Preflight checks for serving a model with vLLM.

This script never imports vLLM or torch just to print their versions, so it is
safe on partially configured hosts. Exit code: 0=ready/warnings, 1=hard failure.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass
class Check:
    name: str
    status: str
    message: str
    recommendation: str | None = None

@dataclass
class ModelCandidate:
    model_id: str
    parameters_b: float | None
    pipeline_tag: str | None
    architectures: list[str]
    quantization: str | None
    downloads: int
    estimated_weight_gib: float | None = None
    fits: bool | None = None
    score: float = 0.0

QUANTIZATION_BYTES = {"awq": .5, "gptq": .5, "gguf": .5, "int4": .5,
                      "bnb-4bit": .5, "bitsandbytes": .5, "fp8": 1, "int8": 1}

def detect_quantization(model_id: str, tags: list[str] | None = None,
                        config: dict[str, Any] | None = None) -> str | None:
    haystack = " ".join([model_id, *(tags or [])]).lower()
    qconfig = (config or {}).get("quantization_config") or {}
    method = qconfig.get("quant_method") if isinstance(qconfig, dict) else None
    return str(method).lower() if method else next(
        (name for name in QUANTIZATION_BYTES if name in haystack), None)

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
            runner = infer_runner(None, architectures)
            modalities = ["text"]
            if any(hasattr(config, key) for key in ("vision_config", "image_token_id")):
                modalities.append("image/video")
            if hasattr(config, "audio_config"):
                modalities.append("audio")
            context = next((getattr(config, key) for key in
                            ("max_position_embeddings", "model_max_length", "n_positions")
                            if isinstance(getattr(config, key, None), int)), None)
            details = f"architecture={supported}; runner={runner}; modalities={'+'.join(modalities)}"
            if context:
                details += f"; model_context={context}"
            return Check("model_compatibility", "PASS", details)
        return Check("model_compatibility", "FAIL", f"unsupported architecture(s): {architectures}",
                     "Choose a vLLM-supported model architecture or upgrade vLLM.")
    except Exception as exc:
        return Check("model_compatibility", "FAIL", f"validation failed: {type(exc).__name__}: {exc}",
                     "Cache the model config locally or run without --offline.")

def dtype_bytes(dtype: str) -> float:
    return {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1,
            "float8": 1, "int4": 0.5}.get(dtype, 2)

def discover_family(family: str, dtype: str, capacity_gib: float,
                    limit: int) -> tuple[list[ModelCandidate], str | None]:
    """Query the Hub and rank vLLM-compatible checkpoints for this hardware."""
    try:
        from huggingface_hub import HfApi
        from vllm import ModelRegistry
        supported = ModelRegistry.get_supported_archs()
        results = []
        for info in HfApi().list_models(search=family, sort="downloads",
                                        limit=limit, expand=["safetensors", "config"]):
            model_id = str(getattr(info, "id", ""))
            lowered = model_id.casefold()
            excluded = ("internal-testing", "tiny-random", "test-model", "tokenizer-only")
            if (family.casefold() not in lowered or getattr(info, "gated", False) or
                    any(marker in lowered for marker in excluded)):
                continue
            config = getattr(info, "config", None) or {}
            archs = list(config.get("architectures") or [])
            if archs and not set(archs) & supported:
                continue
            total = getattr(getattr(info, "safetensors", None), "total", None)
            size_b = float(total) / 1e9 if total else None
            quant = detect_quantization(model_id, list(getattr(info, "tags", None) or []), config)
            weight = (size_b * 1e9 * QUANTIZATION_BYTES.get(quant, dtype_bytes(dtype))
                      / 1024**3 * 1.15) if size_b else None
            fits = weight <= capacity_gib if weight is not None and capacity_gib > 0 else None
            downloads = int(getattr(info, "downloads", 0) or 0)
            item = ModelCandidate(model_id, size_b, getattr(info, "pipeline_tag", None),
                                  archs, quant, downloads, weight, fits)
            instruct = any(x in model_id.casefold() for x in ("instruct", "chat"))
            official = model_id.casefold().startswith(f"{family.casefold()}/")
            fit_score = (10000 + min(size_b or 0, 100) * 10) if fits else (
                -10000 - (size_b * 1000) if size_b is not None else -20000)
            item.score = (fit_score + (1500 if instruct else 0) +
                          (1000 if official else 0) + math.log10(downloads + 1))
            results.append(item)
        return sorted(results, key=lambda x: x.score, reverse=True), None
    except Exception as exc:
        return [], f"Hub family search failed: {type(exc).__name__}: {exc}"

def infer_runner(pipeline_tag: str | None, architectures: list[str]) -> str:
    tag = (pipeline_tag or "").lower()
    arch_text = " ".join(architectures).lower()
    if (tag in {"feature-extraction", "sentence-similarity", "text-classification",
                "token-classification", "zero-shot-classification"} or
            any(x in arch_text for x in ("embedding", "classification", "rewardmodel", "bertmodel"))):
        return "pooling"
    if tag in {"automatic-speech-recognition", "audio-text-to-text"} or any(
            "whisper" in x.lower() for x in architectures):
        return "transcription"
    return "generate"

def run_checks(args: argparse.Namespace) -> tuple[list[Check], list[dict], list[ModelCandidate]]:
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
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = max(1, len(gpus))
        checks.append(Check("tensor_parallel", "PASS",
                            f"auto-selected TP={args.tensor_parallel_size} from {len(gpus)} detected GPU(s)"))
    if gpu_error:
        checks.append(Check("gpu", "FAIL", gpu_error, "Install NVIDIA drivers and verify nvidia-smi."))
    elif len(gpus) < args.tensor_parallel_size:
        checks.append(Check("gpu", "FAIL", f"{len(gpus)} GPU(s), TP={args.tensor_parallel_size}",
                            "Reduce tensor parallel size or provide more GPUs."))
    else:
        detail = "; ".join(f"GPU {g['index']} {g['name']} free={g['memory_free_mib']} MiB"
                           for g in gpus)
        checks.append(Check("gpu", "PASS", detail))

    recommendations: list[ModelCandidate] = []
    family = args.family or (args.model if "/" not in args.model and not Path(args.model).exists() else None)
    if family:
        if args.offline:
            checks.append(Check("family_discovery", "FAIL", "family search needs Hub access",
                                "Remove --offline or provide an exact cached model ID."))
        else:
            capacity = min((g["memory_free_mib"] / 1024 for g in gpus), default=0)
            capacity *= args.tensor_parallel_size * args.gpu_memory_utilization
            recommendations, error = discover_family(family, args.dtype, capacity,
                                                       args.family_candidates)
            if recommendations:
                fitting = [x for x in recommendations if x.fits]
                args.model = (fitting or recommendations)[0].model_id
                checks.append(Check("family_discovery", "PASS" if fitting else "WARN",
                                    f"selected {args.model} from {len(recommendations)} compatible candidates",
                                    None if fitting else "No candidate fits measured VRAM; consider quantization."))
            else:
                checks.append(Check("family_discovery", "FAIL", error or "no compatible model found",
                                    "Try a more specific family or exact repository ID."))

    model_size_b, size_source = resolve_model_size_b(args.model, args.model_size_b, args.offline)
    if model_size_b:
        checks.append(Check("model_size", "PASS", f"{model_size_b:.3f}B parameters ({size_source})"))
        weight_gib = model_size_b * 1e9 * dtype_bytes(args.dtype) / 1024**3
        per_gpu = weight_gib * 1.15 / args.tensor_parallel_size
        usable = min((g["memory_free_mib"] / 1024 for g in gpus),
                     default=0)
        status = "PASS" if usable >= per_gpu else "FAIL"
        required_gpus = math.ceil(weight_gib * 1.15 / usable) if usable > 0 else None
        formula = (f"{model_size_b:.3f}B x {dtype_bytes(args.dtype):g} bytes x 1.15 "
                   f"/ TP={args.tensor_parallel_size} = {per_gpu:.1f} GiB/GPU; "
                   f"free={usable:.1f} GiB/GPU")
        recommendation = None if status == "PASS" else (
            f"Weights alone need about {required_gpus} similar GPUs. "
            "Choose a quantized checkpoint or a smaller model; KV cache is additional.")
        checks.append(Check("weight_memory", status, formula, recommendation))
    else:
        checks.append(Check("weight_memory", "WARN", f"not detected ({size_source})",
                            "Check Hub access or override with --model-size-b."))

    checks.append(check_model_compatibility(args.model, args.offline))
    if args.check_server:
        checks.append(check_server(args.api_url, args.model, args.timeout))
    return checks, gpus, recommendations[:args.recommend_limit]

def serve_command(args: argparse.Namespace) -> str:
    parts = ["vllm", "serve", args.model, "--tensor-parallel-size", str(args.tensor_parallel_size),
             "--dtype", args.dtype, "--gpu-memory-utilization", str(args.gpu_memory_utilization),
             "--max-model-len", str(args.max_model_len)]
    if args.runner != "auto":
        parts.extend(["--runner", args.runner])
    return " ".join(parts)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Exact model ID/path or family, e.g. Qwen")
    parser.add_argument("--model", default=None, help="Exact model ID (legacy form)")
    parser.add_argument("--family", help="Discover and recommend a model family from the Hub")
    parser.add_argument("--family-candidates", type=int, default=40)
    parser.add_argument("--recommend-limit", type=int, default=5)
    parser.add_argument("--model-size-b", type=float,
                        help="Optional parameter-count override in billions; normally auto-detected.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "int8", "float8", "int4"],
                        default="float16")
    parser.add_argument("--tensor-parallel-size", type=int,
                        help="Override TP; default uses all detected GPUs.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--runner", choices=["auto", "generate", "pooling", "transcription"],
                        default="auto", help="Override the vLLM runner when needed")
    parser.add_argument("--min-ram-gib", type=float, default=8)
    parser.add_argument("--offline", action="store_true",
                        help="Do not query Hugging Face Hub; model size may require an override.")
    parser.add_argument("--check-server", action="store_true")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.query and args.model:
        parser.error("use either positional query or --model, not both")
    args.model = args.query or args.model or "mistralai/Mistral-7B-Instruct-v0.3"
    if args.family_candidates < 1 or args.recommend_limit < 1:
        parser.error("family/recommend limits must be >= 1")
    if args.model_size_b is not None and args.model_size_b <= 0:
        parser.error("--model-size-b must be positive")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args

def main() -> int:
    args = parse_args()
    checks, gpus, recommendations = run_checks(args)
    failed = any(item.status == "FAIL" for item in checks)
    report = {"ready": not failed, "checks": [asdict(item) for item in checks],
              "gpus": gpus,
              "selected_model": args.model,
              "recommendations": [asdict(x) for x in recommendations],
              "recommended_serve_command": None if failed else serve_command(args)}
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        for item in checks:
            print(f"[{item.status:4}] {item.name}: {item.message}")
            if item.recommendation:
                print(f"       -> {item.recommendation}")
        if recommendations:
            print("\nRanked model recommendations:")
            for i, item in enumerate(recommendations, 1):
                size = f"{item.parameters_b:.2f}B" if item.parameters_b else "size unknown"
                memory = f"~{item.estimated_weight_gib:.1f} GiB" if item.estimated_weight_gib else "memory unknown"
                fit = "FIT" if item.fits else ("NO FIT" if item.fits is False else "UNKNOWN")
                runner = infer_runner(item.pipeline_tag, item.architectures)
                quant = f", {item.quantization}" if item.quantization else ""
                print(f"  {i}. {item.model_id} [{fit}] {size}, {memory}, runner={runner}{quant}")
        if report["ready"]:
            print(f"\nRecommended command:\n{report['recommended_serve_command']}")
        else:
            print("\nNo runnable vLLM configuration fits the detected hardware.")
        print(f"\nResult: {'READY' if report['ready'] else 'NOT READY'}")
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
