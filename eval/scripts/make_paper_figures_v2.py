#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow>=14.0",
#     "numpy>=1.26",
#     "matplotlib>=3.7",
#     "statsmodels>=0.14",
# ]
# ///
"""Generate the additional paper figures (April 2026 round).

Reads from a flat bucket cache (e.g. `.paper-audit/bucket/` after
`hf buckets cp`-ing the relevant runs locally, or any directory laid out the
way the bucket itself is). The cache root must contain at least:

    addgene_reference_metrics.csv
    addgene_500_mfe_per_seq.parquet
    grpo_plannotate_full_20260408/
        metrics/{tier1_distributional,tier3_essentials,prompt_fidelity_per_seq,mfe_per_seq}.parquet
    base_kmer6_matched_20260420_1303/
        metrics/{...}
    bon_{base,grpo}_seed{43,44,45}_20260420_1303/
        metrics/{tier3_essentials,prompt_fidelity_per_seq}.parquet

Produces three PDFs (+PNGs) into --out:
    bok_useful_curve.pdf            — main-text BoK curve, replaces appendix figure
    distribution_violins.pdf        — length / GC / MFE, three populations
    vocab_treemap.pdf               — 57 hard-token treemap, area=requested, color=GRPO hit rate
                                      (skipped if --skip-treemap or `squarify` not installed)

Usage:
    uv run --script eval/scripts/make_paper_figures_v2.py \\
        --cache .paper-audit/bucket \\
        --out eval/figures/paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

# ── styling shared across figures ──────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "real": "#1f4e79",       # deep blue
    "base": "#6b7280",       # grey
    "grpo": "#059669",       # green
}

BASE_RUN = "base_kmer6_matched_20260420_1303"
GRPO_RUN = "grpo_plannotate_full_20260408"
BON_SEEDS = {
    "base": [BASE_RUN] + [f"bon_base_seed{s}_20260420_1303" for s in (43, 44, 45)],
    "grpo": [GRPO_RUN] + [f"bon_grpo_seed{s}_20260420_1303" for s in (43, 44, 45)],
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


# ── Fig 5 — Best-of-K useful curve, promoted to main text ──────────────────

def fig_bok_useful_curve(cache: Path, out_dir: Path) -> None:
    """Two-panel BoK curve: plausibility (left), useful @ T=0.5 (right).
    Both panels share an x-axis of K ∈ {1, 2, 4} and overlay base, GRPO, and
    a horizontal real-Addgene plausibility reference at 95.0% (left panel
    only — useful@T=0.5 was not measured on real plasmids).
    """
    def load_seed(run: str) -> pd.DataFrame:
        t3 = pd.read_parquet(cache / run / "metrics/tier3_essentials.parquet").reset_index(drop=True)
        f = pd.read_parquet(cache / run / "metrics/prompt_fidelity_per_seq.parquet").reset_index(drop=True)
        df = pd.DataFrame({
            "plaus": t3["plausibility_pass"].astype(bool).values,
            "n_req": f["n_requested"].values,
            "n_fnd": f["n_found"].values,
        })
        df["fid"] = np.where(df["n_req"] > 0,
                             df["n_fnd"] / np.maximum(df["n_req"], 1),
                             np.nan)
        df["useful"] = df["plaus"] & (df["fid"] >= 0.5)
        return df

    base_seeds = [load_seed(r) for r in BON_SEEDS["base"]]
    grpo_seeds = [load_seed(r) for r in BON_SEEDS["grpo"]]

    def best_of_k(seeds: list[pd.DataFrame], K: int) -> dict[str, np.ndarray]:
        sub = seeds[:K]
        stack = np.stack([s["n_fnd"].values for s in sub], axis=0)
        plaus = np.stack([s["plaus"].astype(int).values for s in sub], axis=0)
        fid = np.stack([np.nan_to_num(s["fid"].values, nan=-1.0) for s in sub], axis=0)
        key = stack * 1e6 + plaus * 1e3 + fid
        idx = key.argmax(axis=0)
        out = {}
        for col in ("plaus", "useful"):
            v = np.stack([s[col].values for s in sub], axis=0)
            out[col] = v[idx, np.arange(stack.shape[1])].astype(bool)
        return out

    Ks = [1, 2, 4]
    rates: dict[tuple[str, str], list[float]] = {("base", "plaus"): [], ("grpo", "plaus"): [],
                                                   ("base", "useful"): [], ("grpo", "useful"): []}
    cis: dict[tuple[str, str], list[tuple[float, float]]] = {k: [] for k in rates}
    for K in Ks:
        b = best_of_k(base_seeds, K)
        g = best_of_k(grpo_seeds, K)
        for name, dct in [("base", b), ("grpo", g)]:
            for metric in ("plaus", "useful"):
                v = dct[metric]
                k = int(v.sum()); n = len(v)
                lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
                rates[(name, metric)].append(k / n)
                cis[(name, metric)].append((float(lo), float(hi)))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw={"wspace": 0.25})
    panels = [
        (axes[0], "plaus", "Plausibility (structural)", 95.0,
         "real Addgene plausibility (95.0%)"),
        (axes[1], "useful", r"Useful (plausible $\wedge$ fidelity $\geq 0.5$)", None, None),
    ]
    for ax, metric, title, ref, ref_label in panels:
        for name, label, color in [("base", "supervised base", COLORS["base"]),
                                    ("grpo", "post-GRPO", COLORS["grpo"])]:
            vals = np.array(rates[(name, metric)]) * 100
            ci = np.array(cis[(name, metric)]) * 100
            ax.plot(Ks, vals, "o-", color=color, label=label, linewidth=1.6, markersize=5)
            ax.fill_between(Ks, ci[:, 0], ci[:, 1], color=color, alpha=0.18)
        if ref is not None:
            ax.axhline(ref, color=COLORS["real"], linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(4, ref + 0.6, ref_label, color=COLORS["real"], ha="right", fontsize=8)
        ax.set_xlabel("K (best-of-$K$, oracle ranker)")
        ax.set_ylabel("rate (%)")
        ax.set_title(title)
        ax.set_xticks(Ks)
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right", framealpha=0.9, frameon=False)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save(fig, out_dir, "bok_useful_curve")
    print(f"  bok_useful_curve.pdf — base: {[round(r*100,1) for r in rates[('base','useful')]]} | "
          f"GRPO: {[round(r*100,1) for r in rates[('grpo','useful')]]}")


# ── Fig 1 — Distribution violins (length / GC / MFE) ───────────────────────

def fig_distribution_violins(cache: Path, out_dir: Path) -> None:
    """Three-panel violin: length, GC, MFE density across real / base / GRPO."""
    # Length & GC come from per-seq tier1_distributional.parquet
    base_t1 = pd.read_parquet(cache / BASE_RUN / "metrics/tier1_distributional.parquet")
    grpo_t1 = pd.read_parquet(cache / GRPO_RUN / "metrics/tier1_distributional.parquet")
    real_ref = pd.read_csv(cache / "addgene_reference_metrics.csv")

    # MFE comes from the per-seq parquets we generated 2026-04-27
    base_mfe = pd.read_parquet(cache / BASE_RUN / "metrics/mfe_per_seq.parquet")
    grpo_mfe = pd.read_parquet(cache / GRPO_RUN / "metrics/mfe_per_seq.parquet")
    real_mfe = pd.read_parquet(cache / "addgene_500_mfe_per_seq.parquet")

    populations = [
        ("real\nAddgene", "real", real_ref["length"].to_numpy(),
         real_ref["gc"].to_numpy() * 100,
         real_mfe["mfe_density"].dropna().to_numpy()),
        ("base\n(pre-GRPO)", "base", base_t1["length"].to_numpy(),
         base_t1["gc"].to_numpy() * 100,
         base_mfe["mfe_density"].dropna().to_numpy()),
        ("post-GRPO", "grpo", grpo_t1["length"].to_numpy(),
         grpo_t1["gc"].to_numpy() * 100,
         grpo_mfe["mfe_density"].dropna().to_numpy()),
    ]
    labels = [p[0] for p in populations]
    colors = [COLORS[p[1]] for p in populations]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2),
                              gridspec_kw={"wspace": 0.32})

    panel_specs = [
        (axes[0], "length (kb)",  [p[2] / 1000 for p in populations], None),
        (axes[1], "GC content (%)", [p[3] for p in populations], None),
        (axes[2], "MFE density (kcal/mol/nt)",
         [p[4] for p in populations], None),
    ]

    for ax, ylabel, data, _ in panel_specs:
        positions = list(range(1, len(data) + 1))
        parts = ax.violinplot(data, positions=positions, widths=0.78,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(0.55)
        # Box / median markers on top
        bp = ax.boxplot(data, positions=positions, widths=0.18,
                        patch_artist=True, manage_ticks=False, showfliers=False,
                        whiskerprops={"color": "#222", "linewidth": 0.9},
                        capprops={"color": "#222", "linewidth": 0.9},
                        medianprops={"color": "white", "linewidth": 1.4})
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c); box.set_edgecolor("#222"); box.set_alpha(0.95)
        # Reference horizontal line at real-population median
        real_median = float(np.median(data[0]))
        ax.axhline(real_median, color=COLORS["real"], linestyle=":", linewidth=0.9, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    # Per-panel small caption with means underneath the labels
    panel_captions = ["mean (kb)", "mean (%)", "mean (kcal/mol/nt)"]
    for ax, (_, _, data, _), cap in zip(axes, panel_specs, panel_captions):
        means = [np.mean(d) for d in data]
        ax.set_xlabel(
            f"{cap}: {means[0]:.2f} / {means[1]:.2f} / {means[2]:.2f}",
            fontsize=8, color="#444", labelpad=8
        )
        ax.tick_params(axis="x", pad=4)

    _save(fig, out_dir, "distribution_violins")
    real, base, grpo = [p[3] for p in populations]
    print(f"  distribution_violins.pdf — GC mean (real/base/GRPO): "
          f"{np.mean(real):.1f} / {np.mean(base):.1f} / {np.mean(grpo):.1f}; "
          f"MFE mean: {np.mean(populations[0][4]):.3f} / {np.mean(populations[1][4]):.3f} / {np.mean(populations[2][4]):.3f}")


# ── Fig 4 — Vocabulary treemap (optional) ──────────────────────────────────

def fig_vocab_treemap(cache: Path, out_dir: Path) -> None:
    """Treemap of the 57 hard-token component types.
    Tile area = n_requested across the n=1000 evaluation prompts.
    Tile color = post-GRPO hit rate (red→green colormap, anchored 0–60%).
    Grouping by category is shown via a colored stripe in each tile.
    """
    try:
        import squarify  # type: ignore
    except ImportError:
        print("  vocab_treemap.pdf — SKIPPED (install `squarify` to enable)")
        return

    base = pd.read_parquet(cache / BASE_RUN / "metrics/prompt_fidelity_per_seq.parquet")
    grpo = pd.read_parquet(cache / GRPO_RUN / "metrics/prompt_fidelity_per_seq.parquet")

    def per_token(df: pd.DataFrame) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for req_list, fnd_list in zip(df["requested_tokens"], df["found_tokens"]):
            for t in req_list:
                counts.setdefault(t, {"req": 0, "fnd": 0})["req"] += 1
            for t in fnd_list:
                counts.setdefault(t, {"req": 0, "fnd": 0})["fnd"] += 1
        return counts

    base_counts = per_token(base)
    grpo_counts = per_token(grpo)
    tokens = sorted(set(base_counts) | set(grpo_counts),
                    key=lambda t: -(grpo_counts.get(t, {}).get("req", 0)))
    tokens = [t for t in tokens if grpo_counts.get(t, {}).get("req", 0) > 0]

    rows = []
    for t in tokens:
        req = grpo_counts.get(t, {}).get("req", 0)
        fnd = grpo_counts.get(t, {}).get("fnd", 0)
        cat = t.split("_", 1)[0]
        rows.append({"token": t, "category": cat, "n_requested": req, "n_found": fnd,
                     "hit_rate": fnd / max(req, 1)})
    df = pd.DataFrame(rows).sort_values("n_requested", ascending=False).reset_index(drop=True)

    cmap = plt.colormaps["RdYlGn"]
    norm = plt.Normalize(vmin=0.0, vmax=0.6)
    colors_tile = [cmap(norm(r)) for r in df["hit_rate"]]
    sizes = df["n_requested"].astype(float).clip(lower=1).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects = squarify.normalize_sizes(sizes, 100, 60)
    rects = squarify.squarify(rects, x=0, y=0, dx=100, dy=60)

    for r, color, _, row in zip(rects, colors_tile, df["hit_rate"], df.itertuples()):
        ax.add_patch(plt.Rectangle((r["x"], r["y"]), r["dx"], r["dy"],
                                    facecolor=color, edgecolor="white", linewidth=1.2))
        # label only if tile is large enough
        if r["dx"] * r["dy"] > 20:
            label = row.token.replace("_", "\n", 1)
            ax.text(r["x"] + r["dx"] / 2, r["y"] + r["dy"] / 2,
                    f"{label}\n{row.hit_rate*100:.0f}%",
                    ha="center", va="center", fontsize=7,
                    color="black" if row.hit_rate > 0.25 else "white")

    ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Hard-token vocabulary — area = n_requested in eval, color = post-GRPO hit rate")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("post-GRPO hit rate")
    _save(fig, out_dir, "vocab_treemap")
    print(f"  vocab_treemap.pdf — {len(df)} tokens, "
          f"top by requested: {df.iloc[0]['token']} ({df.iloc[0]['n_requested']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True,
                        help="Local bucket cache root (flat layout)")
    parser.add_argument("--out", type=Path, default=Path("eval/figures/paper"))
    parser.add_argument("--skip-treemap", action="store_true")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=["bok", "violins", "treemap"])
    args = parser.parse_args()

    if not args.cache.is_dir():
        raise SystemExit(f"cache dir does not exist: {args.cache}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {args.out}/")

    plan = args.only or ["bok", "violins", "treemap"]
    if "bok" in plan:
        fig_bok_useful_curve(args.cache, args.out)
    if "violins" in plan:
        fig_distribution_violins(args.cache, args.out)
    if "treemap" in plan and not args.skip_treemap:
        fig_vocab_treemap(args.cache, args.out)


if __name__ == "__main__":
    main()
