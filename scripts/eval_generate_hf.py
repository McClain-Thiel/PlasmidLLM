"""Generate sequences from an HF checkpoint and annotate with pLannotate.

Usage:
    python scripts/eval_generate_hf.py \
        --checkpoint /opt/dlami/nvme/checkpoints/training_pairs_v2/checkpoint-10000 \
        --n 50 --output eval_results_hf.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.models.hf_plasmid_lm import PlasmidLMForCausalLM, PlasmidLMTokenizer

PLANNOTATE_BIN = "/opt/dlami/nvme/miniconda3/envs/plannotate/bin/plannotate"
DEFAULT_PARQUET = "/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/training_pairs_v2.parquet"
DEFAULT_VOCAB = "/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/token_vocabulary.json"


def clean_dna(seq: str) -> str:
    return re.sub(r"[^ATCGNatcgn]", "", seq)


def parse_prompt_tags(prompt: str) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    for m in re.finditer(r"<([A-Z]+)_([A-Z0-9_]+)>", prompt):
        tags.setdefault(m.group(1), []).append(m.group(2))
    return tags


def run_plannotate(dna: str, label: str, idx: int) -> list[str]:
    if len(dna) < 100:
        print(f"  [{label}] idx={idx}: too short ({len(dna)} bp), skipping")
        return []
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = Path(tmpdir) / "input.fa"
        fasta.write_text(f">seq_{idx}_{label}\n{dna}\n")
        result = subprocess.run(
            [PLANNOTATE_BIN, "batch", "-i", str(fasta), "-o", tmpdir, "-c"],
            capture_output=True, text=True, timeout=300,
        )
        csv_files = list(Path(tmpdir).glob("*_pLann.csv"))
        if csv_files:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            if "Feature" in df.columns:
                return df["Feature"].tolist()
        if result.returncode != 0:
            print(f"  plannotate error ({label} {idx}): {result.stderr[:200]}")
        return []


def get_val_indices(n_total: int, val_split: float, seed: int) -> list[int]:
    """Replicate the exact train/val split from data.py::train_val_split."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    val_size = int(n_total * val_split)
    return indices[:val_size]


def main():
    parser = argparse.ArgumentParser(description="Generate + annotate plasmids from HF checkpoint")
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint directory")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--n", type=int, default=50, help="Number of val samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_results_hf.csv")
    parser.add_argument("--skip-plannotate", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    print(f"Loading tokenizer from {args.vocab}")
    tokenizer = PlasmidLMTokenizer(args.vocab)

    print(f"Loading model from {args.checkpoint}")
    model = PlasmidLMForCausalLM.from_pretrained(args.checkpoint)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | device={device}")

    # --- Load data ---
    print(f"Loading data from {args.parquet}")
    table = pq.read_table(args.parquet)
    prompts = table.column("token_prompt").to_pylist()
    col = "token_completion" if "token_completion" in table.column_names else "sequence"
    completions = table.column(col).to_pylist()

    # Replicate exact val split from training
    val_indices = get_val_indices(len(prompts), args.val_split, args.seed)
    print(f"Val set: {len(val_indices)} samples | selecting {args.n}")

    # Random sample from val set
    torch.manual_seed(args.seed + 1)
    picks = torch.randperm(len(val_indices))[:args.n].tolist()
    selected = [val_indices[p] for p in picks]

    # --- Generate ---
    results = []
    for i, idx in enumerate(selected):
        prompt_text = prompts[idx]
        true_seq = clean_dna(completions[idx])
        tags = parse_prompt_tags(prompt_text)

        # Input format matches training: BOS + prompt + SEP
        prompt_ids = tokenizer.encode(prompt_text)
        bos_id = tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id
        input_ids = torch.tensor(
            [[bos_id] + prompt_ids + [sep_id]],
            device=device, dtype=torch.long,
        )

        print(f"\n[{i+1}/{args.n}] idx={idx} | {tags} | true={len(true_seq)}bp")

        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = output[0, input_ids.shape[1]:].tolist()
            eos = tokenizer.eos_token_id
            if eos in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos)]
            gen_dna = clean_dna(tokenizer.decode(gen_ids))
        except Exception as e:
            print(f"  Generation failed: {e}")
            gen_dna = ""

        print(f"  Generated: {len(gen_dna)} bp")

        results.append({
            "idx": idx,
            "prompt": prompt_text,
            "prompt_tags": json.dumps(tags),
            "true_seq": true_seq,
            "gen_seq": gen_dna,
            "true_len": len(true_seq),
            "gen_len": len(gen_dna),
            "temperature": args.temperature,
            "top_k": args.top_k,
        })

    # --- Annotate ---
    if not args.skip_plannotate:
        print("\n=== Running pLannotate ===")
        for r in results:
            idx = r["idx"]
            gen_feats = run_plannotate(r["gen_seq"], "gen", idx)
            true_feats = run_plannotate(r["true_seq"], "true", idx)

            gen_set = set(gen_feats)
            true_set = set(true_feats)
            overlap = gen_set & true_set

            r["gen_features"] = json.dumps(sorted(gen_feats))
            r["true_features"] = json.dumps(sorted(true_feats))
            r["feature_overlap"] = json.dumps(sorted(overlap))
            r["n_gen_features"] = len(gen_set)
            r["n_true_features"] = len(true_set)
            r["n_overlap"] = len(overlap)
            r["recall"] = round(len(overlap) / max(len(true_set), 1), 4)
            r["precision"] = round(len(overlap) / max(len(gen_set), 1), 4)

            tags = json.loads(r["prompt_tags"])
            print(
                f"  idx={idx}: true={len(true_set)}, gen={len(gen_set)}, "
                f"overlap={len(overlap)}, recall={r['recall']:.2f} | {tags}"
            )

    # --- Write results ---
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {output}")
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY  (n={len(results)}, checkpoint={Path(args.checkpoint).name})")
    print(f"{'='*60}")
    print(f"Mean true len:   {sum(r['true_len'] for r in results)/len(results):.0f} bp")
    print(f"Mean gen len:    {sum(r['gen_len'] for r in results)/len(results):.0f} bp")

    if "n_overlap" in results[0]:
        n_true = [r["n_true_features"] for r in results]
        n_gen = [r["n_gen_features"] for r in results]
        n_ov = [r["n_overlap"] for r in results]
        recalls = [r["recall"] for r in results]
        precs = [r["precision"] for r in results]
        print(f"Mean true feats: {sum(n_true)/len(n_true):.1f}")
        print(f"Mean gen feats:  {sum(n_gen)/len(n_gen):.1f}")
        print(f"Mean overlap:    {sum(n_ov)/len(n_ov):.1f}")
        print(f"Mean recall:     {sum(recalls)/len(recalls):.3f}")
        print(f"Mean precision:  {sum(precs)/len(precs):.3f}")

        # Per-sample table
        print(f"\n{'idx':>6}  {'true':>5}  {'gen':>5}  {'ovlp':>5}  {'rec':>5}  tags")
        print("-" * 70)
        for r in results:
            tags = json.loads(r["prompt_tags"])
            tag_str = " ".join(f"{k}:{'+'.join(v)}" for k, v in tags.items())
            print(
                f"{r['idx']:>6}  {r['n_true_features']:>5}  {r['n_gen_features']:>5}  "
                f"{r['n_overlap']:>5}  {r['recall']:>5.2f}  {tag_str}"
            )


if __name__ == "__main__":
    main()
