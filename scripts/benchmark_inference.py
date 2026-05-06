"""Benchmark PlasmidLM inference throughput.

Usage:
    python scripts/benchmark_inference.py                          # baseline only
    python scripts/benchmark_inference.py --compiled-generate      # CUDA graph + static KV cache
    python scripts/benchmark_inference.py --half                   # fp16
    python scripts/benchmark_inference.py --all                    # run both baseline and compiled
    python scripts/benchmark_inference.py --all --half             # run both in fp16
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure local plasmid_llm package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def benchmark(model, input_ids, max_new_tokens, gen_fn, warmup=1, runs=3, label=""):
    """Run generation and report timing stats."""
    for _ in range(warmup):
        _ = gen_fn(input_ids, max_new_tokens=min(100, max_new_tokens), temperature=0.8, top_k=50)
    torch.cuda.synchronize()

    times = []
    total_tokens = 0
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = gen_fn(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tokens = out.shape[1] - input_ids.shape[1]
        times.append(t1 - t0)
        total_tokens += gen_tokens
        print(f"  Run {i+1}: {t1-t0:.3f}s, {gen_tokens} tokens, {gen_tokens/(t1-t0):.1f} tok/s")

    avg_time = sum(times) / len(times)
    avg_tokens = total_tokens / len(times)
    print(f"\n[{label}] Average: {avg_time:.3f}s, {avg_tokens:.0f} tokens, {avg_tokens/avg_time:.1f} tok/s\n")
    return avg_time, avg_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="<org>/PlasmidLM", help="HF model ID or local path")
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--compiled-generate", action="store_true", help="Use generate_compiled (CUDA graph + static KV cache)")
    parser.add_argument("--half", action="store_true", help="Use fp16 precision")
    parser.add_argument("--all", action="store_true", help="Run both baseline and compiled-generate")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    def load_model():
        from plasmid_llm.models.hf_plasmid_lm import PlasmidLMForCausalLM, PlasmidLMConfig
        from plasmid_llm.models.hf_plasmid_lm.tokenization_plasmid_lm import PlasmidLMTokenizer
        config = PlasmidLMConfig.from_pretrained(args.model)
        tok = PlasmidLMTokenizer.from_pretrained(args.model)
        mdl = PlasmidLMForCausalLM.from_pretrained(args.model, config=config)
        mdl = mdl.cuda().eval()
        if args.half:
            mdl = mdl.half()
        return tok, mdl

    print(f"Loading model: {args.model}")
    tokenizer, model = load_model()

    prompt = "<bos><Kanamycin_resistance>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    precision = "fp16" if args.half else "fp32"
    print(f"Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Precision: {precision}")
    print(f"torch version: {torch.__version__}")
    print()

    results = {}

    run_baseline = args.all or not args.compiled_generate
    run_compiled_gen = args.all or args.compiled_generate

    # --- Baseline ---
    if run_baseline:
        print("=" * 60)
        print(f"BASELINE (generate_simple, {precision})")
        print("=" * 60)
        with torch.inference_mode():
            t, tok = benchmark(model, input_ids, args.max_new_tokens,
                               model.generate_simple, warmup=args.warmup,
                               runs=args.runs, label=f"baseline_{precision}")
        results["baseline"] = (t, tok)

    # --- generate_compiled (CUDA graph + static KV cache) ---
    if run_compiled_gen:
        if run_baseline:
            del model
            torch.cuda.empty_cache()
            _, model = load_model()

        print("=" * 60)
        print(f"COMPILED GENERATE (CUDA graph + static KV cache, {precision})")
        print("=" * 60)
        print("Warmup (includes graph capture)...")
        with torch.inference_mode():
            _ = model.generate_compiled(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
        torch.cuda.synchronize()
        print("Graph captured.\n")

        with torch.inference_mode():
            t, tok = benchmark(model, input_ids, args.max_new_tokens,
                               model.generate_compiled, warmup=args.warmup,
                               runs=args.runs, label=f"compiled_{precision}")
        results["compiled"] = (t, tok)

    # --- Summary ---
    if len(results) > 1:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, (t, tok) in results.items():
            print(f"  {name:25s}: {t:.3f}s, {tok/t:.1f} tok/s")
        if "baseline" in results:
            base_t = results["baseline"][0]
            for name, (t, tok) in results.items():
                if name != "baseline":
                    print(f"  {name} speedup vs baseline: {base_t/t:.2f}x")


if __name__ == "__main__":
    main()
