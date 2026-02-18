"""Generate sequences from a checkpoint and annotate with pLannotate.

Usage:
    python scripts/eval_generate.py \
        --checkpoint checkpoints/best.pt \
        --n 5 --temperature 0.8 --top-k 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.models import build_model
from plasmid_llm.tokenizer import PlasmidTokenizer


# ── Tag parsing ──────────────────────────────────────────────────────────────

TAG_CATEGORIES = {
    "AMR": "Antibiotic Resistance",
    "ORI": "Origin of Replication",
    "PROM": "Promoter",
    "COPY": "Copy Number",
    "VEC": "Vector Type",
    "SPECIES": "Species",
    "GC": "GC Content",
    "SIZE": "Size",
    "ELEM": "Regulatory Element",
    "REPORTER": "Reporter",
}


def parse_prompt_tags(prompt: str) -> dict[str, list[str]]:
    """Extract functional element tags from a prompt string."""
    tags: dict[str, list[str]] = {}
    for match in re.finditer(r"<([A-Z_]+?)_([A-Z0-9_]+)>", prompt):
        category, value = match.group(1), match.group(2)
        tags.setdefault(category, []).append(value)
    return tags


def clean_dna(seq: str) -> str:
    """Strip non-DNA characters from a generated sequence."""
    return re.sub(r"[^ATCGNatcgn]", "", seq)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate + annotate plasmids")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--parquet", default="/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/training_pairs.parquet")
    parser.add_argument("--vocab", default="/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/token_vocabulary.json")
    parser.add_argument("--n", type=int, default=5, help="Number of val samples")
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_results.csv")
    parser.add_argument("--skip-plannotate", action="store_true", help="Skip annotation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoint ──────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Find model config in checkpoint (various formats)
    model_cfg = ckpt.get("model_config")
    if model_cfg is None:
        raw = ckpt.get("config", {})
        model_cfg = raw.get("model") if isinstance(raw, dict) else getattr(raw, "model", None)
    if model_cfg is None:
        raise RuntimeError(f"Cannot find model config in checkpoint. Keys: {list(ckpt.keys())}")

    # Handle both OmegaConf dict and SimpleNamespace
    if isinstance(model_cfg, dict):
        cfg = SimpleNamespace(**model_cfg)
    else:
        cfg = model_cfg

    tokenizer = PlasmidTokenizer(args.vocab)
    model = build_model(cfg, tokenizer.vocab_size)

    # Load weights
    state = ckpt.get("model_state_dict") or ckpt.get("model")
    model.load_state_dict(state)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    print(f"Model loaded: {cfg.arch}, {sum(p.numel() for p in model.parameters()):,} params")

    # ── Get validation prompts ───────────────────────────────────────────
    print(f"Loading data: {args.parquet}")
    table = pq.read_table(args.parquet)
    prompts = table.column("token_prompt").to_pylist()
    col = "token_completion" if "token_completion" in table.column_names else "sequence"
    completions = table.column(col).to_pylist()

    n_total = len(prompts)
    n_val = int(n_total * args.val_split)
    torch.manual_seed(args.seed)
    indices = torch.randperm(n_total).tolist()
    val_indices = indices[-n_val:]

    # Pick samples
    torch.manual_seed(args.seed + 1)
    picks = torch.randperm(len(val_indices))[: args.n].tolist()
    selected = [val_indices[i] for i in picks]

    # ── Generate ─────────────────────────────────────────────────────────
    results = []
    for i, idx in enumerate(selected):
        prompt_text = prompts[idx] + "<SEP>"
        true_seq = completions[idx]
        prompt_tags = parse_prompt_tags(prompt_text)

        prompt_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([prompt_ids], device=device)

        print(f"\n[{i+1}/{args.n}] Prompt: {prompts[idx][:100]}...")
        print(f"  Tags: {dict(prompt_tags)}")
        print(f"  True length: {len(true_seq)} bp")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        # Decode only the generated portion (after prompt)
        gen_ids = generated_ids[0, len(prompt_ids):].tolist()
        gen_text = tokenizer.decode(gen_ids)
        gen_dna = clean_dna(gen_text)

        print(f"  Generated: {len(gen_dna)} bp")

        results.append({
            "idx": idx,
            "prompt": prompts[idx],
            "prompt_tags": json.dumps(prompt_tags),
            "true_seq": true_seq,
            "gen_seq": gen_dna,
            "true_len": len(true_seq),
            "gen_len": len(gen_dna),
            "temperature": args.temperature,
            "top_k": args.top_k,
        })

    # ── Annotate with pLannotate (conda CLI) ────────────────────────────
    PLANNOTATE_BIN = "/opt/dlami/nvme/miniconda3/envs/plannotate/bin/plannotate"

    if not args.skip_plannotate:
        import subprocess
        import tempfile

        print("\n=== Running pLannotate annotation ===")

        for r in results:
            idx = r["idx"]
            prompt_tags = json.loads(r["prompt_tags"])

            for label, seq_key in [("gen", "gen_seq"), ("true", "true_seq")]:
                dna = clean_dna(r[seq_key]) if label == "true" else r[seq_key]
                features = []

                if len(dna) < 100:
                    print(f"  {label} seq {idx} too short ({len(dna)} bp), skip")
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        fasta_path = Path(tmpdir) / "input.fa"
                        fasta_path.write_text(f">seq_{idx}_{label}\n{dna}\n")

                        result = subprocess.run(
                            [PLANNOTATE_BIN, "batch", "-i", str(fasta_path),
                             "-o", tmpdir, "-c"],
                            capture_output=True, text=True, timeout=120,
                        )

                        # Find the CSV output
                        csv_files = list(Path(tmpdir).glob("*_pLann.csv"))
                        if csv_files:
                            import pandas as pd
                            ann_df = pd.read_csv(csv_files[0])
                            if "Feature" in ann_df.columns:
                                features = ann_df["Feature"].tolist()

                        if result.returncode != 0 and not features:
                            print(f"  pLannotate {label} seq {idx}: {result.stderr[:200]}")

                r[f"{label}_features"] = json.dumps(features)

            # Compare features
            gen_features = json.loads(r["gen_features"])
            true_features = json.loads(r["true_features"])
            gen_set = set(gen_features)
            true_set = set(true_features)
            overlap = gen_set & true_set
            r["feature_overlap"] = json.dumps(sorted(overlap))
            r["n_gen_features"] = len(gen_features)
            r["n_true_features"] = len(true_features)
            r["n_overlap"] = len(overlap)

            print(f"\n  Sample {idx}:")
            print(f"    Prompt tags: {prompt_tags}")
            print(f"    True features ({len(true_features)}): {true_features}")
            print(f"    Gen features  ({len(gen_features)}): {gen_features}")
            print(f"    Overlap ({len(overlap)}): {sorted(overlap)}")
    else:
        print("\nSkipping pLannotate (--skip-plannotate)")

    # ── Write results ────────────────────────────────────────────────────
    print(f"\nWriting results to {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        prompt_tags = json.loads(r["prompt_tags"])
        tag_str = ", ".join(f"{k}: {v}" for k, v in prompt_tags.items())
        print(f"\nSample {r['idx']} ({r['true_len']}bp true, {r['gen_len']}bp gen)")
        print(f"  Tags: {tag_str}")
        if "n_overlap" in r:
            print(f"  True features: {r.get('n_true_features', '?')}")
            print(f"  Gen features:  {r.get('n_gen_features', '?')}")
            print(f"  Overlap:       {r.get('n_overlap', '?')}")
            print(f"  Overlapping:   {r.get('feature_overlap', '[]')}")


if __name__ == "__main__":
    main()
