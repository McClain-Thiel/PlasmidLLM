#!/usr/bin/env python3
"""Generate plasmid sequences from a PlasmidLM checkpoint.

Uses the model's optimized generate_simple() with KV cache.
Saves FASTA + parquet with per-sequence metadata.

Usage:
    python eval/scripts/generate.py \
        --model McClain/PlasmidLM-kmer6-GRPO-plannotate \
        --n 100 --temperature 0.85 --top-p 0.95 \
        --output-dir eval/runs/firstlight_20260408
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plasmid_llm.provenance import file_sha256  # noqa: E402


def clean_dna(raw: str) -> str:
    seq = re.sub(r"<[^>]+>", "", raw.upper())
    return re.sub(r"[^ATGCN]", "", seq)


def completion_record(tokenizer, output_ids: list[int], prompt_length: int,
                      max_new_tokens: int) -> dict:
    ids = output_ids[prompt_length:]
    has_eos = tokenizer.eos_token_id in ids
    if has_eos:
        ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
    dna = clean_dna(tokenizer.decode(ids, skip_special_tokens=False))
    return dict(sequence=dna, length=len(dna),
                gc=(dna.count("G") + dna.count("C")) / max(len(dna), 1),
                has_eos=has_eos, generated_tokens=len(ids),
                truncated=not has_eos and len(ids) == max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="McClain/PlasmidLM-kmer6-GRPO-plannotate")
    parser.add_argument("--revision", help="Pinned Hugging Face model/tokenizer commit SHA")
    parser.add_argument("--prompts", default="data/training_pairs_v4.parquet",
                        help="Parquet with 'prompt' column. Sampled without replacement "
                             "unless --prompts-verbatim is set.")
    parser.add_argument("--prompts-verbatim", action="store_true",
                        help="Use every row in --prompts in order (no sampling). "
                             "For paired experiments where two runs must share prompts.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="PlasmidLLM")
    parser.add_argument("--run-name", help="Defaults to the unique output directory name")
    args = parser.parse_args()
    if min(args.n, args.batch_size, args.max_new_tokens) < 1:
        parser.error("Sample count, batch size and generation cap must be positive")
    if not Path(args.model).is_dir() and not re.fullmatch(r"[a-fA-F0-9]{40}", args.revision or ""):
        parser.error("Remote checkpoints require --revision with a full commit SHA")
    if args.repetition_penalty != 1.0:
        parser.error("generate_simple does not implement repetition penalties")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    # Save run config
    config = vars(args).copy()
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["prompts_sha256"] = file_sha256(Path(args.prompts))
    config["source_sha256"] = {
        str(p.relative_to(ROOT)): file_sha256(p) for p in (
            Path(__file__).resolve(),
            ROOT / "plasmid_llm/models/hf_plasmid_lm/modeling_plasmid_lm.py",
            ROOT / "plasmid_llm/models/hf_plasmid_lm/configuration_plasmid_lm.py",
        )
    }
    if Path(args.model).is_dir():
        config["checkpoint_sha256"] = {
            p.name: file_sha256(p) for p in Path(args.model).iterdir()
            if p.suffix in {".json", ".safetensors", ".py"}
        }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    import wandb
    from transformers import AutoTokenizer
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMForCausalLM

    run = wandb.init(project=args.wandb_project, name=args.run_name or out.name,
                     config=config, tags=["plasmidrl", "evaluation", f"seed-{args.seed}"])

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, padding_side="left",
    )
    # Custom PlasmidKmerTokenizer doesn't expose special tokens properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<EOS>"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<BOS>"
    # Use the same reviewed sampler for every checkpoint, including released bases.
    model = PlasmidLMForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {n_params:.1f}M params on {args.device}")

    prompt_df = pd.read_parquet(args.prompts)
    if args.prompts_verbatim:
        sampled = prompt_df["prompt"].tolist()
        print(f"Using all {len(sampled)} prompts from {args.prompts} verbatim (no sampling)")
    else:
        unique_prompts = prompt_df["prompt"].unique()
        sampled = np.random.choice(
            unique_prompts, size=min(args.n, len(unique_prompts)), replace=False,
        )
        print(f"Sampled {len(sampled)} prompts from {len(unique_prompts)} unique")
    if not len(sampled):
        raise ValueError("Evaluation panel is empty")

    # Generate
    all_results = []
    t0 = time.time()

    for batch_start in range(0, len(sampled), args.batch_size):
        batch_prompts = list(sampled[batch_start:batch_start + args.batch_size])
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)

        with torch.inference_mode():
            outputs = model.generate_simple(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(outputs.shape[0]):
            all_results.append({
                "prompt": batch_prompts[i],
                **completion_record(tokenizer, outputs[i].tolist(),
                                    inputs["input_ids"].shape[1], args.max_new_tokens),
                "seed": args.seed,
            })

        done = len(all_results)
        elapsed = time.time() - t0
        batch_lens = [r["length"] for r in all_results[-len(batch_prompts):]]
        print(f"  [{done}/{len(sampled)}] {done/elapsed*60:.1f} seq/min, "
              f"batch avg = {np.mean(batch_lens):.0f} bp")
        run.log({"generated_sequences": done, "sequences_per_minute": done / elapsed * 60})

    total_time = time.time() - t0

    # Save
    df = pd.DataFrame(all_results)
    if args.prompts_verbatim:
        for column in ("sample_id", "cluster_id", "panel_main", "minority_panels"):
            if column in prompt_df:
                df[column] = prompt_df[column].tolist()
        for column in ("plasmid_id", "sequence_id"):
            if column in prompt_df:
                df[f"source_{column}"] = prompt_df[column].tolist()
    df.index.name = "gen_id"
    df.to_parquet(out / "generations.parquet")

    with open(out / "generations.fasta", "w") as f:
        for i, row in df.iterrows():
            f.write(f">gen_{i}\n{row['sequence']}\n")

    summary = {
        "n_generated": len(df),
        "mean_length": float(df["length"].mean()),
        "median_length": float(df["length"].median()),
        "mean_gc": float(df["gc"].mean()),
        "eos_rate": float(df["has_eos"].mean()),
        "truncation_rate": float(df["truncated"].mean()),
        "mean_generated_tokens": float(df["generated_tokens"].mean()),
        "pct_lt_1kb": float((df["length"] < 1000).mean()),
        "total_time_s": round(total_time, 1),
        "seqs_per_min": round(len(df) / (total_time / 60), 1),
    }
    with open(out / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    run.log(summary)
    artifact = wandb.Artifact(f"{run.id}-generations", type="evaluation")
    for name in ("config.json", "generations.parquet", "generations.fasta", "generation_summary.json"):
        artifact.add_file(str(out / name))
    run.log_artifact(artifact).wait()
    run.finish()

    print(f"\n{'='*60}")
    print(f"Generated {len(df)} sequences in {total_time:.1f}s")
    print(f"  Mean length: {summary['mean_length']:.0f} bp")
    print(f"  Mean GC: {summary['mean_gc']:.3f}")
    print(f"  EOS rate: {summary['eos_rate']:.1%}")
    print(f"  <1kb: {summary['pct_lt_1kb']:.1%}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
