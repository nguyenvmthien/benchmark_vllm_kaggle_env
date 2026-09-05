"""Preflight checks for serving a model with vLLM.

This script never imports vLLM or torch just to print their versions, so it is
safe on partially configured hosts. Exit code: 0=ready/warnings, 1=hard failure.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict

from infercap.preflight.checks import (
    PROFILE_DEFAULTS, infer_runner, inference_recommendation, run_checks, serve_command,
)

def parse_args(argv: list[str] | None = None, *, prog: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("query", nargs="?", help="Exact model ID/path or family, e.g. Qwen")
    parser.add_argument("--model", default=None, help="Exact model ID (legacy form)")
    parser.add_argument("--family", help="Discover and recommend a model family from the Hub")
    parser.add_argument("--family-candidates", type=int, default=40)
    parser.add_argument("--recommend-limit", type=int, default=5)
    parser.add_argument("--model-size-b", type=float,
                        help="Optional parameter-count override in billions; normally auto-detected.")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"],
                        default="float16", help="Weight/activation dtype; use --quantization for INT4/INT8")
    parser.add_argument("--tensor-parallel-size", type=int,
                        help="Override TP; default uses all detected GPUs.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--runner", choices=["auto", "draft", "generate", "pooling"],
                        default="auto", help="Override the vLLM runner when needed")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="balanced",
                        help="Scheduler tuning goal (default: balanced)")
    parser.add_argument("--max-num-seqs", type=int,
                        help="Override the profile's concurrent sequence limit")
    parser.add_argument("--max-num-batched-tokens", type=int,
                        help="Override the profile's token budget per scheduler iteration")
    parser.add_argument("--kv-cache-dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"])
    parser.add_argument("--generation-config", default="vllm",
                        help="Use vllm for reproducible defaults, or auto/model path")
    parser.add_argument("--quantization", choices=["awq", "gptq", "bitsandbytes"],
                        help="Explicit weight quantization override")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--served-model-name", help="Optional stable model name exposed by the API")
    parser.add_argument("--min-ram-gib", type=float, default=8)
    parser.add_argument("--offline", action="store_true",
                        help="Do not query Hugging Face Hub; model size may require an override.")
    parser.add_argument("--check-server", action="store_true")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.query and args.model:
        parser.error("use either positional query or --model, not both")
    args.model = args.query or args.model or "mistralai/Mistral-7B-Instruct-v0.3"
    if args.family_candidates < 1 or args.recommend_limit < 1:
        parser.error("family/recommend limits must be >= 1")
    if args.max_num_seqs is not None and args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be >= 1")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        parser.error("--max-num-batched-tokens must be >= 1")
    if args.model_size_b is not None and args.model_size_b <= 0:
        parser.error("--model-size-b must be positive")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args

def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = parse_args(argv, prog=prog)
    checks, gpus, recommendations = run_checks(args)
    failed = any(item.status == "FAIL" for item in checks)
    inference = inference_recommendation(args)
    report = {"ready": not failed, "checks": [asdict(item) for item in checks],
              "gpus": gpus,
              "selected_model": args.model,
              "recommendations": [asdict(x) for x in recommendations],
              "inference_recommendation": asdict(inference),
              "recommended_serve_command": None if failed else serve_command(args, inference)}
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
        print(f"\nInference profile: {inference.profile}")
        for reason in inference.reasons:
            print(f"  - {reason}")
        if report["ready"]:
            print(f"\nRecommended command:\n{report['recommended_serve_command']}")
        else:
            print("\nNo runnable vLLM configuration fits the detected hardware.")
        print(f"\nResult: {'READY' if report['ready'] else 'NOT READY'}")
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
