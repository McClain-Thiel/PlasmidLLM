#!/usr/bin/env python3
"""Compute per-sequence ViennaRNA MFE density (DNA Mathews2004).

Writes one row per input sequence so downstream figures (violins, scatter)
have full distributions instead of just summary stats.

For sequences ≤ 5,000 bp: single full-length fold.
For sequences > 5,000 bp: average of 5 random 2,000 bp windows (deterministic
  per-sequence seed — same windows on a re-run).

Reproduces the aggregation behavior of `compute_extra_metrics.compute_mfe_density`
but persists the per-seq values.

Inputs accepted:
  - generations.parquet (cols: prompt, sequence, length, ...)
  - FASTA file

Usage:
  conda run -n plannotate python eval/scripts/compute_per_seq_mfe.py \\
      --input runs/grpo_plannotate_full_20260408/generations.parquet \\
      --output runs/grpo_plannotate_full_20260408/metrics/mfe_per_seq.parquet

  conda run -n plannotate python eval/scripts/compute_per_seq_mfe.py \\
      --input reference/addgene_500.fasta \\
      --output reference/addgene_500_mfe_per_seq.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 2000
N_WINDOWS = 5
LONG_SEQ_THRESHOLD = 5000
MIN_LEN = 50
MAX_LEN = 15000


def iter_fasta(path: Path):
    seq_id, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq)
                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if seq_id is not None:
            yield seq_id, "".join(seq)


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        if "sequence" not in df.columns:
            raise ValueError(f"parquet missing 'sequence' column: {path}")
        ids = (
            df["seq_id"].astype(str).tolist()
            if "seq_id" in df.columns
            else [f"gen_{i}" for i in range(len(df))]
        )
        return pd.DataFrame({"seq_id": ids, "sequence": df["sequence"].astype(str).tolist()})
    if path.suffix in {".fasta", ".fa", ".fna"}:
        rows = list(iter_fasta(path))
        return pd.DataFrame(rows, columns=["seq_id", "sequence"])
    raise ValueError(f"unsupported input format: {path.suffix}")


def fold_one(rna_module, seq: str, seed_idx: int) -> tuple[float | None, int, str]:
    """Return (mfe_density_kcal_mol_nt, n_windows, status)."""
    L = len(seq)
    if L < MIN_LEN or L > MAX_LEN:
        return None, 0, f"length_out_of_range_{L}"
    try:
        if L > LONG_SEQ_THRESHOLD:
            rng = np.random.RandomState(seed_idx)
            window_mfes = []
            for _ in range(N_WINDOWS):
                start = rng.randint(0, L - WINDOW_SIZE)
                window = seq[start:start + WINDOW_SIZE]
                _, mfe = rna_module.fold(window)
                window_mfes.append(mfe / len(window))
            return float(np.mean(window_mfes)), N_WINDOWS, "ok_windowed"
        _, mfe = rna_module.fold(seq)
        return mfe / L, 1, "ok_full"
    except Exception as exc:
        return None, 0, f"error:{exc.__class__.__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="generations.parquet or .fasta")
    parser.add_argument("--output", type=Path, required=True,
                        help="output parquet path")
    parser.add_argument("--n", type=int, default=None,
                        help="subsample N rows (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for subsample (default 42)")
    args = parser.parse_args()

    try:
        import RNA  # type: ignore
    except ImportError:
        raise SystemExit(
            "ViennaRNA not installed in this env. "
            "Run inside the plannotate conda env: "
            "`conda run -n plannotate python eval/scripts/compute_per_seq_mfe.py ...`"
        )

    RNA.params_load_DNA_Mathews2004()
    print("loaded DNA params (Mathews et al. 2004)", flush=True)

    df = load_input(args.input)
    print(f"loaded {len(df)} sequences from {args.input}", flush=True)

    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"subsampled to n={len(df)} (seed={args.seed})", flush=True)

    out_rows = []
    t_start = time.time()
    for idx, row in df.iterrows():
        seq = row["sequence"]
        density, n_windows, status = fold_one(RNA, seq, seed_idx=idx)
        out_rows.append({
            "seq_id": row["seq_id"],
            "length": len(seq),
            "mfe_density": density,
            "n_windows": n_windows,
            "status": status,
        })
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1e-3)
            remaining = (len(df) - idx - 1) / max(rate, 1e-3)
            print(f"  [{idx+1}/{len(df)}] {rate:.1f} seq/s  ETA {remaining/60:.1f} min",
                  flush=True)

    out_df = pd.DataFrame(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    valid = out_df.dropna(subset=["mfe_density"])
    print(f"\nwrote {len(out_df)} rows ({len(valid)} valid) to {args.output}")
    if len(valid):
        print(f"  mean   = {valid['mfe_density'].mean():+.4f} kcal/mol/nt")
        print(f"  median = {valid['mfe_density'].median():+.4f}")
        print(f"  std    = {valid['mfe_density'].std():.4f}")


if __name__ == "__main__":
    main()
