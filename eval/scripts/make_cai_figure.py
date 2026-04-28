#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=14.0", "numpy>=1.26", "matplotlib>=3.7"]
# ///
"""Forest plot of (CAI_yeast − CAI_ecoli) per (source × SP token).

Reads `cai_per_seq_*.parquet` produced by `compute_cai_by_host.py` from --cache,
which should contain:
    cai_per_seq_training_sample.parquet      — real plasmids stratified by SP
    cai_per_seq_forced_base.parquet          — forced SP_ECOLI/SP_YEAST gen, base
    cai_per_seq_forced_grpo.parquet          — same, post-GRPO
    cai_per_seq_grpo_plannotate_full_20260408.parquet — paper eval set, GRPO
    cai_per_seq_base_kmer6_matched_20260420_1303.parquet — paper eval set, base

Writes:
    cai_yeast_minus_ecoli_forest.{pdf,png}

Usage:
    uv run --script eval/scripts/make_cai_figure.py \\
        --cache eval/figures/paper/cai \\
        --out eval/figures/paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

COLORS = {
    "real":       "#1f4e79",  # deep blue
    "base":       "#6b7280",  # grey
    "grpo":       "#059669",  # green
}

SOURCES = [
    ("real",   "real plasmids (training sample)",   "training_sample"),
    ("base",   "supervised base (eval-set, n=1000)", "base_kmer6_matched_20260420_1303"),
    ("grpo",   "post-GRPO (eval-set, n=1000)",       "grpo_plannotate_full_20260408"),
    ("base",   "supervised base (forced SP_ECOLI / SP_YEAST)", "forced_base"),
    ("grpo",   "post-GRPO (forced SP_ECOLI / SP_YEAST)",       "forced_grpo"),
]
SP_TOKENS = ["SP_ECOLI", "SP_YEAST", "SP_HUMAN", "SP_MOUSE"]


def boot_mean_ci(x: np.ndarray, B: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for kind, label, run in SOURCES:
        path = args.cache / f"cai_per_seq_{run}.parquet"
        if not path.is_file():
            print(f"  skip (missing): {path}")
            continue
        df = pd.read_parquet(path).dropna(subset=["cai_ecoli_wmean", "cai_yeast_wmean"])
        df["d"] = df["cai_yeast_wmean"] - df["cai_ecoli_wmean"]
        for sp in SP_TOKENS:
            sub = df[df["sp_token"] == sp]
            if len(sub) == 0:
                continue
            # Skip tiny-n cells from the original eval set for SP_ECOLI/SP_YEAST —
            # the forced runs (n≈100–266) are the right comparison.
            if (run in ("base_kmer6_matched_20260420_1303",
                        "grpo_plannotate_full_20260408")
                and sp in ("SP_ECOLI", "SP_YEAST")):
                continue
            m, lo, hi = boot_mean_ci(sub["d"].to_numpy())
            rows.append({"kind": kind, "label": label, "run": run,
                         "sp": sp, "n": len(sub), "mean": m, "lo": lo, "hi": hi})

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    # Print summary table to stdout for inclusion in the paper
    print("\n=== summary (paper text) ===")
    for _, r in tbl.iterrows():
        sig = "***" if (r["lo"] > 0 or r["hi"] < 0) else "n.s."
        print(f"{r['label']:<48s} {r['sp']:<10s} n={int(r['n']):>3d}  "
              f"Δ = {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]  {sig}")

    # ── forest plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # Order: by SP token first, then by source within SP
    order_keys = []
    for sp in SP_TOKENS:
        for kind, label, run in SOURCES:
            sub = tbl[(tbl["sp"] == sp) & (tbl["run"] == run)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            order_keys.append((sp, kind, label, run, r["mean"], r["lo"], r["hi"], r["n"]))

    n_rows = len(order_keys)
    y_positions = list(range(n_rows, 0, -1))
    y_labels = []
    last_sp = None
    for (sp, kind, label, run, m, lo, hi, n), y in zip(order_keys, y_positions):
        col = COLORS[kind]
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]],
                     fmt="o", markersize=6, capsize=4,
                     color=col, ecolor=col, linewidth=1.4)
        # SP label in bold once, then source label indented
        if sp != last_sp:
            y_labels.append(f"$\\bf{{{sp.replace('SP_','')}}}$  ·  {label}  (n={n})")
            last_sp = sp
        else:
            y_labels.append(f"     {label}  (n={n})")

    ax.set_yticks(y_positions, y_labels)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$\overline{CAI}_\mathrm{yeast} - \overline{CAI}_\mathrm{ecoli}$"
                   r"  per sequence  (mean ± 95% bootstrap CI)")
    ax.set_title("Host-specific codon usage in real and generated plasmids "
                  "stratified by SP token", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    # legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color=COLORS["real"], linestyle="",
               markersize=6, label="real plasmids"),
        Line2D([0], [0], marker="o", color=COLORS["base"], linestyle="",
               markersize=6, label="supervised base"),
        Line2D([0], [0], marker="o", color=COLORS["grpo"], linestyle="",
               markersize=6, label="post-GRPO"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "cai_yeast_minus_ecoli_forest.pdf")
    fig.savefig(args.out / "cai_yeast_minus_ecoli_forest.png", dpi=200)
    plt.close(fig)
    print(f"\nwrote {args.out}/cai_yeast_minus_ecoli_forest.pdf")


if __name__ == "__main__":
    main()
