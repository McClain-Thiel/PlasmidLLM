#!/usr/bin/env python3
"""Unconditional generation from the completion-only plasmid LMs.

PlasmidGPT (Shao 2024), PlasmidGPT-SFT (Cunningham) and PlasmidGPT-GRPO are ordinary
GPT-2s over a BPE vocabulary of DNA. They have no interface for saying which components a
plasmid should contain — you seed them with sequence and they continue. That is exactly the
point of the comparison: `baselines.py completion` scores these samples against the same
specs to measure the base rate an unsteerable generator hits by chance.

Seeding: a short fixed prefix, since these models expect sequence rather than a prompt.
The seed is deliberately generic ("ATG", as in the model card's quick-start) so it does not
smuggle in component information and quietly turn this into a conditional baseline.

Separate from eval/scripts/generate.py because the tokenizer, the absence of prompts and
the stopping behaviour all differ; forcing both through one script would make neither clear.

Usage:
  python generate_completion.py --model UCL-CSSB/PlasmidGPT --out pg_base \\
      --n 1000 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed-text", default="ATG")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ids = tok(args.seed_text, return_tensors="pt").input_ids.to(args.device)
    rows, t0 = [], time.time()
    for start in range(0, args.n, args.batch_size):
        k = min(args.batch_size, args.n - start)
        batch = ids.repeat(k, 1)
        with torch.no_grad():
            out = model.generate(
                batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        for row in out:
            dna = "".join(
                c for c in tok.decode(row, skip_special_tokens=True).upper()
                if c in "ACGT"
            )
            gc = (dna.count("G") + dna.count("C")) / len(dna) if dna else 0.0
            rows.append({"sequence": dna, "length": len(dna), "gc": gc})
        done = len(rows)
        print(f"  [{done}/{args.n}] {done/(time.time()-t0)*60:.1f} seq/min", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "generations.parquet", index=False)
    with open(args.out / "generations.fasta", "w") as fh:
        for i, s in enumerate(df.sequence):
            if s:
                fh.write(f">gen_{i}\n{s}\n")
    summary = {
        "model": args.model,
        "n_generated": len(df),
        "mean_length": float(df.length.mean()),
        "median_length": float(df.length.median()),
        "mean_gc": float(df.gc.mean()),
        "total_time_s": round(time.time() - t0, 1),
    }
    (args.out / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
