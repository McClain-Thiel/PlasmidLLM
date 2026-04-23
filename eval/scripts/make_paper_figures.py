#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas==2.3.3",
#     "pyarrow>=14.0",
#     "numpy==2.2.6",
#     "matplotlib==3.10.8",
# ]
# ///
"""Generate paper figures that complement the main workflow.

Produces two PDFs into the output dir:

    fidelity_scatter.{png,pdf}
        Per-prompt base fidelity (x) vs GRPO fidelity (y) for every prompt
        with n_requested > 0. Hex-binned to handle the discrete-fraction
        overplotting (fidelity values cluster at 0/1, 1/2, 2/3, etc.).
        Marginal histograms break out the shift on each axis; color ramp
        encodes prompt complexity (n_requested).

    per_token_hit_rates.{png,pdf}
        All hard tokens from the motif registry, rank-ordered by the max of
        base / GRPO hit rate. Dumbbell plot: base (grey) and GRPO (green)
        dots connected by a line. Token labels carry the n_requested count.
        The figure visualizes the long-tail distribution: a few tokens
        dominate, many are ignored entirely (hit rate 0).

Both figures share the same paired cache used by the soft-token analysis:

    /tmp/plasmidlm_eval_cache/runs__base_kmer6_matched_20260420_1303
    /tmp/plasmidlm_eval_cache/runs__grpo_plannotate_full_20260408

Usage:
    uv run --script eval/scripts/make_paper_figures.py --out eval/figures/paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

CACHE = Path("/tmp/plasmidlm_eval_cache")
BASE_DIR = CACHE / "runs__base_kmer6_matched_20260420_1303"
GRPO_DIR = CACHE / "runs__grpo_plannotate_full_20260408"


# ──────────────────────────────────────────────────────────────────────────
# Figure A — per-prompt fidelity scatter (paired)
# ──────────────────────────────────────────────────────────────────────────

def plot_fidelity_scatter(out_dir: Path) -> None:
    base_fid = pd.read_parquet(BASE_DIR / "metrics/prompt_fidelity_per_seq.parquet").reset_index(drop=True)
    grpo_fid = pd.read_parquet(GRPO_DIR / "metrics/prompt_fidelity_per_seq.parquet").reset_index(drop=True)
    assert (base_fid["prompt"].values == grpo_fid["prompt"].values).all(), \
        "prompt order mismatch between runs"

    mask = (base_fid["n_requested"] > 0) & (grpo_fid["n_requested"] > 0)
    n_req = base_fid.loc[mask, "n_requested"].to_numpy()
    x = (base_fid.loc[mask, "n_found"] / base_fid.loc[mask, "n_requested"]).to_numpy()
    y = (grpo_fid.loc[mask, "n_found"] / grpo_fid.loc[mask, "n_requested"]).to_numpy()

    # Jitter by a small amount so points at identical fractions are visible
    # but the underlying density is still readable.
    rng = np.random.default_rng(0)
    jitter = 0.012
    xj = np.clip(x + rng.normal(0, jitter, size=x.size), 0.0, 1.0)
    yj = np.clip(y + rng.normal(0, jitter, size=y.size), 0.0, 1.0)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[5, 1.2], height_ratios=[1.2, 5],
        hspace=0.04, wspace=0.04,
        left=0.10, right=0.94, top=0.88, bottom=0.10,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter.
    sc = ax_main.scatter(
        xj, yj,
        c=n_req, cmap="viridis", s=14, alpha=0.45, linewidths=0,
        vmin=np.quantile(n_req, 0.02), vmax=np.quantile(n_req, 0.98),
    )
    ax_main.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_main.set_xlim(-0.03, 1.03); ax_main.set_ylim(-0.03, 1.03)
    ax_main.set_xlabel("base fidelity (n_found / n_requested per prompt)")
    ax_main.set_ylabel("GRPO fidelity (n_found / n_requested per prompt)")
    ax_main.grid(alpha=0.25)

    # Mean markers.
    ax_main.axvline(float(x.mean()), color="#6b7280", linestyle=":", linewidth=1, alpha=0.8)
    ax_main.axhline(float(y.mean()), color="#059669", linestyle=":", linewidth=1, alpha=0.8)
    ax_main.text(
        float(x.mean()) + 0.02, 0.02,
        f"base mean={x.mean():.2f}",
        color="#374151", fontsize=9,
    )
    ax_main.text(
        0.02, float(y.mean()) + 0.02,
        f"GRPO mean={y.mean():.2f}",
        color="#065f46", fontsize=9,
    )

    # Marginal histograms.
    bins = np.linspace(0, 1, 41)
    ax_top.hist(x, bins=bins, color="#9ca3af", alpha=0.9, edgecolor="white", linewidth=0.4)
    ax_top.set_ylabel("base\ncount", fontsize=9)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(axis="y", alpha=0.25)
    ax_right.hist(y, bins=bins, orientation="horizontal",
                  color="#059669", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_right.set_xlabel("GRPO\ncount", fontsize=9)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(axis="x", alpha=0.25)

    # Colorbar — use figure-level placement that won't fight the grid.
    cax = fig.add_axes([0.42, 0.935, 0.40, 0.018])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("n_requested tokens per prompt", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Fraction above/below diagonal.
    above = int((y > x + 1e-9).sum())
    below = int((y < x - 1e-9).sum())
    equal = int((np.abs(y - x) <= 1e-9).sum())
    ax_main.text(
        0.55, 0.06,
        f"above diag: {above}  ({above / mask.sum() * 100:.1f}%)\n"
        f"below diag: {below}  ({below / mask.sum() * 100:.1f}%)\n"
        f"equal:      {equal}",
        fontsize=9, color="#111827",
        bbox=dict(facecolor="white", edgecolor="#d1d5db", boxstyle="round,pad=0.4"),
    )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fidelity_scatter.{ext}", dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure B — per-token dumbbell (long tail)
# ──────────────────────────────────────────────────────────────────────────

def _load_per_token(run_dir: Path) -> pd.DataFrame:
    with open(run_dir / "metrics/extra_metrics.json") as f:
        d = json.load(f)
    rows = [
        {"token": k, "hit_rate": v["hit_rate"],
         "n_requested": v["n_requested"], "n_found": v["n_found"]}
        for k, v in d["prompt_fidelity"]["per_token"].items()
    ]
    return pd.DataFrame(rows)


def plot_per_token_dumbbell(out_dir: Path) -> None:
    base = _load_per_token(BASE_DIR)
    grpo = _load_per_token(GRPO_DIR)
    merged = base.merge(
        grpo, on="token", suffixes=("_base", "_grpo"), how="outer"
    ).fillna({"hit_rate_base": 0, "hit_rate_grpo": 0,
              "n_requested_base": 0, "n_requested_grpo": 0,
              "n_found_base": 0, "n_found_grpo": 0})
    # Use max(n_requested) as the authoritative count (prompts are matched).
    merged["n_requested"] = merged[["n_requested_base", "n_requested_grpo"]].max(axis=1).astype(int)
    merged = merged[merged["n_requested"] > 0].copy()
    merged["max_rate"] = merged[["hit_rate_base", "hit_rate_grpo"]].max(axis=1)

    # Order: highest max_rate first (descending), then alphabetical for ties.
    merged = merged.sort_values(["max_rate", "token"], ascending=[False, True]).reset_index(drop=True)

    # Category colour for the left-side accent.
    cat_colors = {
        "ORI":      "#1f77b4",
        "AMR":      "#d62728",
        "PROM":     "#2ca02c",
        "ELEM":     "#9467bd",
        "REPORTER": "#ff7f0e",
        "TAG":      "#8c564b",
        "REP":      "#7f7f7f",
    }

    def _cat(tok): return tok.split("_", 1)[0]

    n = len(merged)
    fig_h = max(6.5, 0.18 * n + 1.2)
    fig = plt.figure(figsize=(9.5, fig_h))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[5, 1],
        wspace=0.02, left=0.22, right=0.97, top=0.93, bottom=0.06,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1], sharey=ax)

    y = np.arange(n)
    base_rates = merged["hit_rate_base"].to_numpy()
    grpo_rates = merged["hit_rate_grpo"].to_numpy()

    # Connector line between base and GRPO per token.
    for i in range(n):
        ax.plot([base_rates[i], grpo_rates[i]], [y[i], y[i]],
                color="#d1d5db", linewidth=1.2, zorder=1)
    ax.scatter(base_rates, y, s=32, color="#6b7280", zorder=3, label="base", edgecolor="white", linewidth=0.4)
    ax.scatter(grpo_rates, y, s=32, color="#059669", zorder=3, label="GRPO", edgecolor="white", linewidth=0.4)

    # Category colour strip on the far left (before axis labels), as short
    # coloured rectangles.
    for i, tok in enumerate(merged["token"]):
        ax.barh(y[i], 0.012, left=-0.020, color=cat_colors.get(_cat(tok), "#888"),
                height=0.75, zorder=2, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["token"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("per-token hit rate")
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(
        "Per-token fidelity, rank-ordered — long tail of zero-hit tokens "
        "(AMR_CHLORAMPHENICOL, ORI_P15A, ORI_PSC101,\nshort epitope tags). "
        "Base (grey) → GRPO (green) shift.",
        fontsize=10,
    )

    # Right-side n_requested panel (log scale emphasises the long tail).
    n_req = merged["n_requested"].to_numpy()
    ax_n.barh(y, n_req, color="#111827", height=0.7)
    ax_n.set_xscale("log")
    ax_n.set_xlabel("n_requested")
    ax_n.tick_params(axis="y", labelleft=False)
    ax_n.grid(axis="x", alpha=0.25, which="both")

    # Category legend in corner (abbreviated).
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_colors.values()]
    fig.legend(handles, list(cat_colors.keys()),
               loc="lower center", ncol=len(cat_colors),
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"per_token_hit_rates.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_fidelity_scatter(args.out)
    plot_per_token_dumbbell(args.out)
    print(f"wrote figures to {args.out}:")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
