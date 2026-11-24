#!/usr/bin/env python3
"""Regenerate all publication figures deterministically from cluster_v3 rigor archives."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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
    "savefig.pad_inches": 0.06,
})

COLORS = {
    "real": "#1f4e79",
    "base": "#6b7280",
    "grpo": "#059669",
}

CATEGORIES = ["ORI", "PROM", "AMR", "ELEM", "REPORTER", "TAG"]

FEATURE_TYPE_COLOR = {
    "rep_origin": "#2563eb",
    "CDS": "#7c3aed",
    "promoter": "#059669",
    "terminator": "#d97706",
    "polyA_signal": "#d97706",
    "enhancer": "#0d9488",
    "LTR": "#e11d48",
    "misc_feature": "#64748b",
    "ncRNA": "#ec4899",
    "protein_bind": "#0ea5e9",
    "oriT": "#2563eb",
    "RBS": "#f59e0b",
}

SOFT_TO_HARD = {
    "COPY": {
        "HIGH": {"ORI_COLE1", "ORI_RSF"},
        "LOW": {"ORI_P15A", "ORI_PSC101"},
    },
    "VEC": {
        "MAMMALIAN": {
            "PROM_CMV",
            "PROM_EF1A",
            "PROM_CAG",
            "ELEM_POLYA_BGH",
            "ELEM_POLYA_SV40",
            "ELEM_CMV_ENHANCER",
        },
        "BACTERIAL": {"PROM_T7", "PROM_T5", "PROM_LAC"},
        "LENTIVIRAL": {"ELEM_LTR_3", "ELEM_LTR_5", "ELEM_PSI", "ELEM_CPPT", "ELEM_WPRE"},
        "CRISPR": {"ELEM_GRNA_SCAFFOLD", "ELEM_TRACRRNA", "PROM_U6"},
        "AAV": {"ELEM_AAV_ITR"},
    },
    "SP": {
        "HUMAN": {"PROM_CMV", "PROM_EF1A", "ELEM_POLYA_BGH"},
        "ECOLI": {"PROM_T7", "PROM_T5", "PROM_LAC"},
        "YEAST": {"ORI_2MU"},
    },
    "BB": {
        "PUC19": {"AMR_AMPICILLIN", "ORI_COLE1"},
        "PCDNA3": {"PROM_CMV", "ELEM_POLYA_BGH", "AMR_NEOMYCIN", "ORI_SV40"},
        "PEGFP_C1": {"REPORTER_EGFP", "PROM_CMV", "AMR_KANAMYCIN"},
        "PEGFP_N1": {"REPORTER_EGFP", "PROM_CMV", "AMR_KANAMYCIN"},
        "PAAV": {"ELEM_AAV_ITR"},
        "PAAV2": {"ELEM_AAV_ITR"},
    },
}
SOFT_RE = re.compile(r"<(BB|COPY|SP|VEC)_([^>]+)>")
ALL_TOKEN_RE = re.compile(r"<([^>]+)>")


def resolve_default_paths() -> tuple[Path, Path, Path]:
    """Resolve default rigor directory, motif registry, and output directory."""
    here = Path(__file__).resolve()
    if here.parent.name == "scripts" and here.parents[1].name == "eval":
        repo_root = here.parents[2]
        rigor_dir = repo_root / "eval" / "runs" / "rigor_20261116"
        registry = repo_root / "data" / "motif_registry_combined.parquet"
        out_dir = repo_root / "eval" / "figures" / "paper"
        return rigor_dir, registry, out_dir
    paper_dir = here.parent
    repo_root = paper_dir.parent / "PlasmidLLM"
    return (
        repo_root / "eval" / "runs" / "rigor_20261116",
        repo_root / "data" / "motif_registry_combined.parquet",
        paper_dir / "figures",
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, both: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    if both:
        fig.savefig(out_dir / f"{stem}.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Exemplar gallery (Figure 1)
# ---------------------------------------------------------------------------


def make_exemplar_gallery(obs_df: pd.DataFrame, annot_dir: Path, out_dir: Path) -> None:
    """Render three held-out PlasmidLM (post-GRPO) exemplar plasmid maps."""
    pn_path = annot_dir / "exp2_v2_s44" / "decode_s42" / "plannotate_results.json"
    with pn_path.open() as f:
        pn_map = {e["seq_id"]: e for e in json.load(f)}

    sub = obs_df[(obs_df["model_name"] == "exp2_v2_s44") & (obs_df["seed"] == 42)]
    exemplars = [
        {
            "seq_id": "gen_388",
            "title": "(a) High-fidelity generation (6/7, F = 0.86)",
            "subtitle": "0 extra · 12,291 bp · viable (<TRACRRNA> unannotated)",
        },
        {
            "seq_id": "gen_360",
            "title": "(b) Typical useful design (3/5, F = 0.60)",
            "subtitle": "0 extra · 3,966 bp (<EOS>) · viable Gateway/T7 vector",
        },
        {
            "seq_id": "gen_153",
            "title": "(c) Minority / <COPY_LOW> failure (3/7, F = 0.43)",
            "subtitle": "4 extra · misses p15A & CmR · 7,239 bp",
        },
    ]

    fig = plt.figure(figsize=(12.4, 8.9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 2.58], hspace=0.55, wspace=0.22)

    hard_prefixes = {"AMR", "ORI", "PROM", "ELEM", "REPORTER", "TAG", "REP"}

    for r_idx, ex in enumerate(exemplars):
        sid = ex["seq_id"]
        row = sub[sub["seq_id"] == sid].iloc[0]
        toks = [t for t in ALL_TOKEN_RE.findall(row["prompt"]) if t not in {"BOS", "SEQ"}]
        det = {t.strip("<>") for t in json.loads(row["detected_tokens_json"])}

        soft_toks = [t for t in toks if t.split("_", 1)[0] not in hard_prefixes]
        hard_toks = [t for t in toks if t.split("_", 1)[0] in hard_prefixes]

        ax_l = fig.add_subplot(gs[r_idx, 0])
        ax_l.set_xlim(0, 1)
        ax_l.set_ylim(0, 1)
        ax_l.axis("off")
        ax_l.text(
            0.0,
            1.13,
            ex["title"],
            transform=ax_l.transAxes,
            fontsize=10.0,
            fontweight="bold",
            color="#111827",
        )
        ax_l.text(
            0.0,
            1.03,
            ex["subtitle"],
            transform=ax_l.transAxes,
            fontsize=8.0,
            color="#4b5563",
        )

        items: list[tuple[str, str, str]] = []
        if soft_toks:
            items.append(("CONTEXT (SOFT TOKENS)", "", "header"))
            for t in soft_toks:
                items.append(("dot", f"<{t}>", "soft"))
        items.append(("REQUESTED COMPONENTS", "", "header"))
        for t in hard_toks:
            ok = t in det
            items.append(("ok" if ok else "miss", f"<{t}>", "ok" if ok else "miss"))

        n_items = len(items)
        step = 0.96 / max(n_items, 1)
        for i, (mark, text, kind) in enumerate(items):
            y = 0.95 - i * step
            if kind == "header":
                ax_l.text(
                    0.0,
                    y,
                    mark,
                    fontsize=7.5,
                    fontweight="bold",
                    color="#6b7280",
                    va="center",
                    transform=ax_l.transAxes,
                )
            else:
                tc = "#111827" if kind != "soft" else "#475569"
                if kind == "ok":
                    ax_l.scatter([0.032], [y], s=42, color="#059669", marker="s", zorder=3)
                    ax_l.text(
                        0.032,
                        y,
                        "+",
                        fontsize=7.5,
                        fontweight="bold",
                        color="white",
                        ha="center",
                        va="center",
                        zorder=4,
                    )
                elif kind == "miss":
                    ax_l.scatter([0.032], [y], s=42, color="#dc2626", marker="s", zorder=3)
                    ax_l.text(
                        0.032,
                        y,
                        "x",
                        fontsize=7.2,
                        fontweight="bold",
                        color="white",
                        ha="center",
                        va="center",
                        zorder=4,
                    )
                else:
                    ax_l.scatter([0.032], [y], s=14, color="#64748b", marker="o", zorder=3)
                ax_l.text(
                    0.09,
                    y,
                    text,
                    fontsize=7.9,
                    fontfamily="monospace",
                    color=tc,
                    va="center",
                    transform=ax_l.transAxes,
                )

        ax_r = fig.add_subplot(gs[r_idx, 1])
        entry = pn_map[sid]
        seq_len = int(entry["seq_length"])

        feats = [f for f in entry["features"] if not f.get("fragment", False)]
        feats.sort(key=lambda x: (int(x["qstart"]), -int(x["qend"])))
        filtered = []
        seen_spans = []
        for f in feats:
            s, e = int(f["qstart"]), int(f["qend"])
            if f.get("Type") == "ncRNA":
                skip = False
                for ps, pe in seen_spans:
                    ov = max(0, min(e, pe) - max(s, ps))
                    if ov / max(1, e - s) > 0.7:
                        skip = True
                        break
                if skip:
                    continue
            filtered.append(f)
            seen_spans.append((s, e))

        ax_r.plot([0, seq_len], [0.0, 0.0], color="#334155", linewidth=2.2, zorder=1)
        ax_r.set_xlim(-0.02 * seq_len, 1.02 * seq_len)
        ax_r.set_ylim(-1.1, 1.35)
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.set_yticks([])
        ax_r.set_xlabel("Position (bp)", fontsize=8.5)
        ax_r.tick_params(axis="x", labelsize=8)

        label_tiers_above = [0.42, 0.72, 1.02]
        label_tiers_below = [-0.42, -0.72]
        occupied: dict[float, list[tuple[float, float]]] = {
            y: [] for y in label_tiers_above + label_tiers_below
        }

        for idx_f, f in enumerate(filtered):
            s = max(0, int(f["qstart"]) - 1)
            e = min(seq_len, int(f["qend"]))
            w = max(e - s, seq_len * 0.012)
            ftype = str(f.get("Type", "misc_feature"))
            col = FEATURE_TYPE_COLOR.get(ftype, "#64748b")
            rect = patches.FancyBboxPatch(
                (s, -0.14),
                w,
                0.28,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=col,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
                zorder=3,
            )
            ax_r.add_patch(rect)

            label = (
                str(f.get("Feature", ""))
                .replace(" (truncated)", "")
                .replace("T-DNA repeat", "T-DNA")
                .replace("\u03a8", r"$\Psi$")
            )
            if len(label) > 19:
                label = label[:17] + "…"
            mid = 0.5 * (s + e)
            half_w = max(0.055 * seq_len, len(label) * seq_len * 0.0046)

            tier_pool = label_tiers_above if idx_f % 2 == 0 else label_tiers_below
            other_pool = label_tiers_below if idx_f % 2 == 0 else label_tiers_above
            chosen_y = None
            for yy in tier_pool + other_pool:
                x0, x1 = mid - half_w, mid + half_w
                if not any(not (x1 < o0 or x0 > o1) for o0, o1 in occupied[yy]):
                    chosen_y = yy
                    occupied[yy].append((x0, x1))
                    break
            if chosen_y is not None:
                va = "bottom" if chosen_y > 0 else "top"
                y_anchor = 0.14 if chosen_y > 0 else -0.14
                ax_r.plot(
                    [mid, mid],
                    [y_anchor, chosen_y * 0.90],
                    color="#94a3b8",
                    linewidth=0.7,
                    zorder=2,
                )
                ax_r.text(
                    mid,
                    chosen_y,
                    label,
                    ha="center",
                    va=va,
                    fontsize=7.5,
                    color="#1f2937",
                    bbox=dict(
                        boxstyle="round,pad=0.14",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                    zorder=4,
                )

        ax_r.set_title(
            f"{seq_len:,} bp · {len(filtered)} intact annotated features",
            loc="left",
            fontsize=9.2,
            color="#374151",
            pad=4,
        )

    legend_handles = [
        patches.Patch(color="#2563eb", label="Origin (ori / f1 / oriT)"),
        patches.Patch(color="#7c3aed", label="CDS / AMR / Cargo"),
        patches.Patch(color="#059669", label="Promoter"),
        patches.Patch(color="#d97706", label="Terminator / poly(A)"),
        patches.Patch(color="#e11d48", label="LTR"),
        patches.Patch(color="#0d9488", label="Enhancer"),
        patches.Patch(color="#64748b", label="Packaging / misc / RBS"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.54, -0.025),
    )
    _save(fig, out_dir, "exemplar_gallery", both=True)


# ---------------------------------------------------------------------------
# 2. Best-of-K curve (Figure 3)
# ---------------------------------------------------------------------------


def make_bok_curve(summary: dict, mps: pd.DataFrame, out_dir: Path) -> None:
    """Render single-shot through Best-of-4 useful-plasmid and viability curves."""
    ks = [1, 2, 4]
    base_main = mps[mps["model_family"] == "cluster_base_s42"].iloc[0]
    grpo_main = mps[mps["model_family"] == "exp2_v2"].iloc[0]
    ref_main = summary["reference_control"]
    bok = summary["best_of_k"]

    base_use = [
        float(base_main["useful_05_mean"]) * 100,
        float(bok["K=2"]["cluster_base_s42"]["useful_05"]["mean"]) * 100,
        float(bok["K=4"]["cluster_base_s42"]["useful_05"]["mean"]) * 100,
    ]
    grpo_use = [
        float(grpo_main["useful_05_mean"]) * 100,
        float(bok["K=2"]["exp2_v2"]["useful_05"]["mean"]) * 100,
        float(bok["K=4"]["exp2_v2"]["useful_05"]["mean"]) * 100,
    ]
    grpo_use_sd = [
        float(grpo_main["useful_05_std"]) * 100,
        float(bok["K=2"]["exp2_v2"]["useful_05"]["std"]) * 100,
        float(bok["K=4"]["exp2_v2"]["useful_05"]["std"]) * 100,
    ]

    base_via = [
        float(base_main["viable_mean"]) * 100,
        float(bok["K=2"]["cluster_base_s42"]["viable"]["mean"]) * 100,
        float(bok["K=4"]["cluster_base_s42"]["viable"]["mean"]) * 100,
    ]
    grpo_via = [
        float(grpo_main["viable_mean"]) * 100,
        float(bok["K=2"]["exp2_v2"]["viable"]["mean"]) * 100,
        float(bok["K=4"]["exp2_v2"]["viable"]["mean"]) * 100,
    ]
    grpo_via_sd = [
        float(grpo_main["viable_std"]) * 100,
        float(bok["K=2"]["exp2_v2"]["viable"]["std"]) * 100,
        float(bok["K=4"]["exp2_v2"]["viable"]["std"]) * 100,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
    panels = [
        (
            axes[0],
            base_use,
            grpo_use,
            grpo_use_sd,
            float(ref_main["useful_05"]) * 100,
            "(a) Useful-plasmid rate ($T = 0.5$)",
            "Useful plasmids (%)",
            (38, 102),
        ),
        (
            axes[1],
            base_via,
            grpo_via,
            grpo_via_sd,
            float(ref_main["viable"]) * 100,
            "(b) Structural viability",
            "Viable plasmids (%)",
            (65, 104),
        ),
    ]

    for ax, b_y, g_y, g_sd, ref_y, title, ylabel, ylim in panels:
        ax.axhline(
            ref_y,
            color=COLORS["real"],
            linestyle="--",
            linewidth=1.3,
            label=f"Held-out Addgene ({ref_y:.1f}%)",
        )
        ax.plot(
            ks,
            b_y,
            marker="s",
            markersize=6,
            linewidth=2.0,
            color=COLORS["base"],
            label="PlasmidLM (supervised base)",
        )
        ax.errorbar(
            ks,
            g_y,
            yerr=g_sd,
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            capsize=3.5,
            color=COLORS["grpo"],
            ecolor=COLORS["grpo"],
            label="PlasmidLM (post-GRPO, 3-seed mean ± SD)",
        )
        for k, by, gy, gsd in zip(ks, b_y, g_y, g_sd):
            ax.annotate(
                f"{by:.1f}%",
                (k, by),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#374151",
            )
            ax.annotate(
                f"{gy:.1f}%",
                (k, gy + gsd),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
                color=COLORS["grpo"],
            )
        ax.set_xticks(ks)
        ax.set_xticklabels(["1\n(single-shot mean)", "2\n(best-of-2)", "4\n(best-of-4)"])
        ax.set_ylim(*ylim)
        ax.set_xlim(0.65, 4.35)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.0)

    fig.tight_layout()
    _save(fig, out_dir, "bok_useful_curve", both=True)


# ---------------------------------------------------------------------------
# 3. Sequence distributions violin/box figure (Figure 4)
# ---------------------------------------------------------------------------


def make_distribution_violins(obs_df: pd.DataFrame, out_dir: Path) -> None:
    """Render length, GC content, and low-complexity distributions."""
    main = obs_df[obs_df["panel_main"].astype(bool)]
    ref = main[main["model_name"] == "reference_control"]
    base = main[main["model_family"] == "cluster_base_s42"]
    grpo = main[main["model_family"] == "exp2_v2"]

    groups = [
        ("Real Addgene", ref, COLORS["real"]),
        ("Supervised base", base, COLORS["base"]),
        ("Post-GRPO", grpo, COLORS["grpo"]),
    ]

    metrics = [
        ("length", 1e-3, "(a) Plasmid length (kb)", "Length (kb)", (0, 15.6)),
        ("gc", 100.0, "(b) GC content (%)", "GC (%)", (25, 75)),
        (
            "low_complexity_frac",
            100.0,
            "(c) Low-complexity masked DNA (%)",
            "Dustmasker fraction (%)",
            (-2, 55),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, (col, scale, title, ylabel, ylim) in zip(axes, metrics):
        data = [g[col].to_numpy(dtype=float) * scale for _, g, _ in groups]
        parts = ax.violinplot(
            data, positions=[1, 2, 3], showmeans=False, showextrema=False, widths=0.72
        )
        for body, (_, _, col_hex) in zip(parts["bodies"], groups):
            body.set_facecolor(col_hex)
            body.set_edgecolor(col_hex)
            body.set_alpha(0.32)
        bp = ax.boxplot(
            data,
            positions=[1, 2, 3],
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#111827", linewidth=1.5),
            whiskerprops=dict(color="#4b5563", linewidth=1.0),
            capprops=dict(color="#4b5563", linewidth=1.0),
        )
        for box, (_, _, col_hex) in zip(bp["boxes"], groups):
            box.set_facecolor(col_hex)
            box.set_alpha(0.75)
            box.set_edgecolor("#111827")

        for pos, vals in enumerate(data, start=1):
            mean_val = float(np.mean(vals))
            ax.scatter([pos], [mean_val], color="white", edgecolor="#111827", s=36, zorder=5)
            unit = "kb" if col == "length" else "%"
            ax.text(
                pos,
                ylim[1] - 0.05 * (ylim[1] - ylim[0]),
                f"μ={mean_val:.1f}{unit}",
                ha="center",
                va="top",
                fontsize=8.3,
                color="#1f2937",
            )

        if col == "length":
            ax.axhline(
                12.288,
                color="#dc2626",
                linestyle=":",
                linewidth=1.1,
                label="4,096-token cap (12.3 kb)",
            )
            ax.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.88)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Real\nAddgene", "Supervised\nbase", "Post-\nGRPO"], fontsize=8.8)
        ax.set_title(title, fontsize=10.3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "distribution_violins", both=True)


# ---------------------------------------------------------------------------
# 4. Threshold sweep / ceiling figure (Appendix B Figure 5)
# ---------------------------------------------------------------------------


def make_threshold_ceiling(obs_df: pd.DataFrame, out_dir: Path) -> None:
    """Render useful-plasmid yield across faithfulness thresholds T in [0, 1]."""
    main = obs_df[obs_df["panel_main"].astype(bool)].copy()
    thresholds = [0.00, 0.25, 0.50, 0.75, 1.00]
    ref = main[main["model_name"] == "reference_control"]
    base = main[main["model_family"] == "cluster_base_s42"]
    grpo = main[main["model_family"] == "exp2_v2"]

    ref_y, base_y, grpo_mean, grpo_sd = [], [], [], []
    for t in thresholds:
        tol = t - 1e-9 if t == 1.0 else t
        ref_y.append(float(((ref["viable"] > 0.5) & (ref["faithfulness"] >= tol)).mean()) * 100)
        base_y.append(float(((base["viable"] > 0.5) & (base["faithfulness"] >= tol)).mean()) * 100)
        seed_rates = []
        for s in (42, 43, 44):
            gs = grpo[grpo["training_seed"] == s]
            seed_rates.append(
                float(((gs["viable"] > 0.5) & (gs["faithfulness"] >= tol)).mean()) * 100
            )
        grpo_mean.append(float(np.mean(seed_rates)))
        grpo_sd.append(float(np.std(seed_rates, ddof=1)))

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.plot(
        thresholds,
        ref_y,
        marker="^",
        markersize=6.5,
        linewidth=2.0,
        linestyle="--",
        color=COLORS["real"],
        label="Held-out real Addgene (pLannotate ceiling)",
    )
    ax.plot(
        thresholds,
        base_y,
        marker="s",
        markersize=5.8,
        linewidth=2.0,
        color=COLORS["base"],
        label="PlasmidLM (supervised base)",
    )
    ax.errorbar(
        thresholds,
        grpo_mean,
        yerr=grpo_sd,
        marker="o",
        markersize=6.2,
        linewidth=2.2,
        capsize=3.5,
        color=COLORS["grpo"],
        ecolor=COLORS["grpo"],
        label="PlasmidLM (post-GRPO, 3-seed mean ± SD)",
    )
    ax.fill_between(thresholds, base_y, grpo_mean, color=COLORS["grpo"], alpha=0.10)
    ax.axvline(0.50, color="#94a3b8", linestyle=":", linewidth=1.2)
    ax.text(0.52, 75, "Primary threshold\n($T = 0.50$)", fontsize=8.3, color="#475569")

    ax.annotate(
        f"{grpo_mean[2]:.1f}%",
        (0.50, grpo_mean[2]),
        xytext=(-12, 8),
        textcoords="offset points",
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["grpo"],
    )
    ax.annotate(
        f"{base_y[2]:.1f}%",
        (0.50, base_y[2]),
        xytext=(-26, -13),
        textcoords="offset points",
        fontsize=8.5,
        color=COLORS["base"],
    )

    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("Minimum faithfulness threshold $T$ (viable $\\wedge\\, F \\geq T$)")
    ax.set_ylabel("Useful-plasmid rate (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("Useful-plasmid rate across faithfulness thresholds ($n = 468$ clusters)")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)
    fig.tight_layout()
    _save(fig, out_dir, "prompt_fidelity_ceiling", both=True)


# ---------------------------------------------------------------------------
# 5. Category recall bar chart (Appendix B Figure 6)
# ---------------------------------------------------------------------------


def make_category_bars(cat_df: pd.DataFrame, obs_df: pd.DataFrame, out_dir: Path) -> None:
    """Render per-category hard-token detection rates across all six component families."""
    main = obs_df[(obs_df["panel_main"].astype(bool)) & (obs_df["model_family"] == "exp2_v2")]

    grpo_seed_rates: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for s in (42, 43, 44):
        req_c: dict[str, int] = defaultdict(int)
        hit_c: dict[str, int] = defaultdict(int)
        for row in main[main["training_seed"] == s].itertuples():
            req = json.loads(row.requested_tokens_json)
            det = set(json.loads(row.detected_tokens_json))
            for tok in req:
                c = tok.strip("<>").split("_", 1)[0]
                if c in req_c or c in CATEGORIES:
                    req_c[c] += 1
                    hit_c[c] += int(tok in det)
        for c in CATEGORIES:
            grpo_seed_rates[c].append(hit_c[c] / max(req_c[c], 1) * 100.0)

    def _rates(fam: str) -> list[float]:
        sub = cat_df[(cat_df["model_family"] == fam) & (cat_df["kind"] == "category")]
        m = dict(zip(sub["name"], sub["plannotate_rate"] * 100.0))
        return [float(m[c]) for c in CATEGORIES]

    ref_r = _rates("reference_control")
    base_r = _rates("cluster_base_s42")
    grpo_r = _rates("exp2_v2")
    grpo_sd = [float(np.std(grpo_seed_rates[c], ddof=1)) for c in CATEGORIES]

    x = np.arange(len(CATEGORIES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.bar(x - w, base_r, width=w, color=COLORS["base"], label="PlasmidLM (supervised base)")
    ax.bar(
        x,
        grpo_r,
        width=w,
        yerr=grpo_sd,
        capsize=3,
        color=COLORS["grpo"],
        label="PlasmidLM (post-GRPO, 3-seed mean ± SD)",
    )
    ax.bar(
        x + w,
        ref_r,
        width=w,
        color=COLORS["real"],
        alpha=0.85,
        label="Held-out real Addgene (ceiling)",
    )

    for i in range(len(CATEGORIES)):
        ax.text(
            x[i] - w,
            base_r[i] + 1.5,
            f"{base_r[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#374151",
        )
        ax.text(
            x[i],
            grpo_r[i] + grpo_sd[i] + 1.5,
            f"{grpo_r[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=COLORS["grpo"],
        )
        ax.text(
            x[i] + w,
            ref_r[i] + 1.5,
            f"{ref_r[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=COLORS["real"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Hard-token detection rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title(
        "Per-category hard-token detection across all 6 token families ($n = 468$ clusters)",
        pad=24,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8.3)
    fig.tight_layout()
    _save(fig, out_dir, "hit_rate_by_category", both=False)


# ---------------------------------------------------------------------------
# 6. Per-token hit rates (Appendix B Figure 7)
# ---------------------------------------------------------------------------


def make_per_token_figure(tok_df: pd.DataFrame, out_dir: Path) -> None:
    """Render per-token detection dumbbell chart for tokens with n_req >= 5."""
    ref = tok_df[tok_df["model_family"] == "reference_control"].copy()
    base = tok_df[tok_df["model_family"] == "cluster_base_s42"].copy()
    grpo = tok_df[tok_df["model_family"] == "exp2_v2"].copy()

    elig = ref[ref["requested"] >= 5]["token"].tolist()
    m = (
        ref[ref["token"].isin(elig)][["token", "category", "requested", "hit_rate"]]
        .rename(columns={"hit_rate": "ref_rate"})
        .merge(
            base[["token", "hit_rate"]].rename(columns={"hit_rate": "base_rate"}),
            on="token",
        )
        .merge(
            grpo[["token", "hit_rate"]].rename(columns={"hit_rate": "grpo_rate"}),
            on="token",
        )
    )
    cat_rank = {c: i for i, c in enumerate(CATEGORIES)}
    m["cat_rank"] = m["category"].map(cat_rank)
    m = m.sort_values(["cat_rank", "grpo_rate"], ascending=[True, True]).reset_index(drop=True)

    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8.8, 0.25 * len(m) + 1.4))

    for idx_r, r in m.iterrows():
        ax.plot(
            [r["base_rate"] * 100, r["grpo_rate"] * 100],
            [idx_r, idx_r],
            color="#cbd5e1",
            linewidth=2.0,
            zorder=1,
        )
    ax.scatter(
        m["base_rate"] * 100,
        y,
        color=COLORS["base"],
        marker="s",
        s=34,
        zorder=3,
        label="Supervised base",
    )
    ax.scatter(
        m["grpo_rate"] * 100,
        y,
        color=COLORS["grpo"],
        marker="o",
        s=40,
        zorder=4,
        label="Post-GRPO (3-seed mean)",
    )
    ax.scatter(
        m["ref_rate"] * 100,
        y,
        color=COLORS["real"],
        marker="^",
        s=34,
        zorder=2,
        alpha=0.85,
        label="Held-out Addgene ceiling",
    )

    ylabels = [f"{r['token'].strip('<>')} (n={int(r['requested'])})" for _, r in m.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=7.6)
    ax.set_xlim(-3, 105)
    ax.set_xlabel("Detection rate when requested (%)")
    ax.set_title("Per-token recall across requested hard tokens ($n \\geq 5$ prompts in test panel)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)
    fig.tight_layout()
    _save(fig, out_dir, "per_token_hit_rates", both=False)


# ---------------------------------------------------------------------------
# 7. Soft-token forest & COPY buckets (Appendix B Figures 8 & 9)
# ---------------------------------------------------------------------------


def _load_sseqids_for_main(main_seq_ids_ordered: list[str], pn_path: Path) -> list[set[str]]:
    with pn_path.open() as f:
        pn = json.load(f)
    by_sid = {
        e["seq_id"]: {
            ft["sseqid"]
            for ft in e.get("features", [])
            if isinstance(ft.get("sseqid"), str) and ft["sseqid"]
        }
        for e in pn
    }
    return [by_sid.get(sid, set()) for sid in main_seq_ids_ordered]


def make_soft_token_forest(
    obs_df: pd.DataFrame,
    annot_dir: Path,
    registry_path: Path,
    out_dir: Path,
) -> None:
    """Render soft metadata token conditional lift forest plot on the 468-cluster panel."""
    reg = pd.read_parquet(registry_path, columns=["token", "sseqid"])
    tok_sseqids: dict[str, set[str]] = defaultdict(set)
    for t, sid in zip(reg["token"], reg["sseqid"]):
        if isinstance(t, str) and isinstance(sid, str):
            tok_sseqids[t.strip("<>")].add(sid)

    probes = {
        cat: {
            val: set().union(*(tok_sseqids.get(t, set()) for t in toks))
            for val, toks in by_val.items()
        }
        for cat, by_val in SOFT_TO_HARD.items()
    }

    main_ref = obs_df[
        (obs_df["panel_main"].astype(bool)) & (obs_df["model_name"] == "reference_control")
    ].copy()
    main_ref = main_ref.sort_values("seq_id").reset_index(drop=True)
    prompts = main_ref["prompt"].tolist()
    seq_ids = main_ref["seq_id"].tolist()

    def _load_arm(model_names: list[str]) -> list[list[set[str]]]:
        runs = []
        for mname in model_names:
            for dseed in (42, 43, 44, 45):
                runs.append(
                    _load_sseqids_for_main(
                        seq_ids, annot_dir / mname / f"decode_s{dseed}" / "plannotate_results.json"
                    )
                )
        return runs

    base_runs = _load_arm(["cluster_base_s42"])
    grpo_runs = _load_arm(["exp2_v2_s42", "exp2_v2_s43", "exp2_v2_s44"])

    order = [
        ("VEC", "LENTIVIRAL"),
        ("VEC", "AAV"),
        ("VEC", "CRISPR"),
        ("VEC", "MAMMALIAN"),
        ("VEC", "BACTERIAL"),
        ("SP", "HUMAN"),
        ("SP", "ECOLI"),
        ("SP", "YEAST"),
        ("COPY", "HIGH"),
        ("COPY", "LOW"),
        ("BB", "PAAV"),
        ("BB", "PAAV2"),
        ("BB", "PEGFP_C1"),
        ("BB", "PEGFP_N1"),
        ("BB", "PCDNA3"),
        ("BB", "PUC19"),
    ]

    rng = np.random.default_rng(42)

    def _stats_for(runs: list[list[set[str]]], cat: str, val: str):
        tags = []
        for p in prompts:
            found = None
            for c, v in SOFT_RE.findall(p):
                if c == cat and found is None:
                    found = v
            tags.append(found)
        tags_arr = np.array(tags, dtype=object)
        mask_a = tags_arr == val
        mask_b = (tags_arr != val) & pd.notna(tags_arr)
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        if n_a < 3 or n_b < 3:
            return None
        probe = probes[cat][val]
        per_prompt_hit = np.mean(
            [[1.0 if (r[i] & probe) else 0.0 for i in range(len(seq_ids))] for r in runs],
            axis=0,
        )
        vals_a = per_prompt_hit[mask_a]
        vals_b = per_prompt_hit[mask_b]
        delta = float(vals_a.mean() - vals_b.mean())
        boot = np.empty(4000)
        for b in range(4000):
            ia = rng.integers(0, n_a, size=n_a)
            ib = rng.integers(0, n_b, size=n_b)
            boot[b] = vals_a[ia].mean() - vals_b[ib].mean()
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return delta, float(lo), float(hi), n_a

    rows_plot = []
    for cat, val in order:
        res_b = _stats_for(base_runs, cat, val)
        res_g = _stats_for(grpo_runs, cat, val)
        if res_b is None and res_g is None:
            continue
        rows_plot.append((cat, val, res_b, res_g))

    fig, ax = plt.subplots(figsize=(8.0, 0.38 * len(rows_plot) + 1.4))
    y = np.arange(len(rows_plot))
    offset = 0.17

    ytick_labels = []
    for idx_o, (cat, val, res_b, res_g) in enumerate(rows_plot):
        n_a = res_g[3] if res_g else res_b[3]
        ytick_labels.append(f"<{cat}_{val}> (n={n_a})")
        if res_b is not None:
            d, lo, hi, _ = res_b
            ax.errorbar(
                d * 100,
                idx_o - offset,
                xerr=[[(d - lo) * 100], [(hi - d) * 100]],
                fmt="s",
                color=COLORS["base"],
                ecolor=COLORS["base"],
                capsize=2.5,
                markersize=4.5,
                label="Supervised base" if idx_o == 0 else None,
            )
        if res_g is not None:
            d, lo, hi, _ = res_g
            ax.errorbar(
                d * 100,
                idx_o + offset,
                xerr=[[(d - lo) * 100], [(hi - d) * 100]],
                fmt="o",
                color=COLORS["grpo"],
                ecolor=COLORS["grpo"],
                capsize=2.5,
                markersize=4.8,
                label="Post-GRPO (3-seed mean)" if idx_o == 0 else None,
            )

    prev = rows_plot[0][0]
    for j, (cat, _, _, _) in enumerate(rows_plot):
        if cat != prev:
            ax.axhline(j - 0.5, color="#e5e7eb", linewidth=0.9)
            prev = cat

    ax.axvline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ probe hit rate (pp): prompts with tag minus other prompts in category")
    ax.set_title("Soft-token discrimination across context tags ($n = 468$ cluster-disjoint panel)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)
    fig.tight_layout()
    _save(fig, out_dir, "soft_token_forest", both=False)


def make_copy_buckets(obs_df: pd.DataFrame, min_df: pd.DataFrame, out_dir: Path) -> None:
    """Render origin copy-number bucket shares under <COPY_HIGH> vs. <COPY_LOW>."""
    main = obs_df[obs_df["panel_main"].astype(bool)]

    def _high_shares(fam: str):
        sub = main[
            (main["model_family"] == fam)
            & (main["prompt"].str.contains("<COPY_HIGH>", regex=False))
        ]
        req_only = float(sub["copy_low_high_only"].mean())
        conf = float(sub["copy_low_conflicting_high_and_low"].mean())
        wrong_only = float(sub["copy_low_compatible_only"].mean())
        other_none = float((sub["copy_low_unclassified_only"] + sub["copy_low_no_origin"]).mean())
        return [req_only, conf, wrong_only, other_none], int(sub["sample_id"].nunique())

    def _low_shares(fam: str):
        r = min_df[
            (min_df["stratum"] == "minority_copy_low") & (min_df["model_family"] == fam)
        ].iloc[0]
        req_only = float(r["copy_low_compatible_only_mean"])
        conf = float(r["copy_low_conflicting_high_and_low_mean"])
        wrong_only = float(r["copy_low_high_only_mean"])
        other_none = float(r["copy_low_unclassified_only_mean"] + r["copy_low_no_origin_mean"])
        return [req_only, conf, wrong_only, other_none], int(r["n_prompts"])

    b_hi, n_hi = _high_shares("cluster_base_s42")
    g_hi, _ = _high_shares("exp2_v2")
    b_lo, n_lo = _low_shares("cluster_base_s42")
    g_lo, _ = _low_shares("exp2_v2")

    cols = [b_hi, g_hi, b_lo, g_lo]
    arr = np.array(cols) * 100.0
    labels_x = ["Supervised\nbase", "Post-\nGRPO", "Supervised\nbase", "Post-\nGRPO"]
    bucket_names = [
        "Requested copy class only",
        "Conflicting high + low origins",
        "Wrong copy class only",
        "Other / no origin detected",
    ]
    bucket_colors = ["#059669", "#f59e0b", "#dc2626", "#94a3b8"]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(4)
    bottoms = np.zeros(4)
    for b_idx in range(4):
        vals = arr[:, b_idx]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.58,
            color=bucket_colors[b_idx],
            edgecolor="white",
            label=bucket_names[b_idx],
        )
        for xi, v, bot in zip(x, vals, bottoms):
            if v >= 5.5:
                ax.text(
                    xi,
                    bot + 0.5 * v,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if b_idx in (0, 2) else "#111827",
                    fontweight="bold",
                )
        bottoms += vals

    ax.axvline(1.5, color="#cbd5e1", linestyle="--", linewidth=1.0)
    ax.text(
        0.5,
        104.5,
        f"<COPY_HIGH> (n = {n_hi})",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#1f2937",
    )
    ax.text(
        2.5,
        104.5,
        f"<COPY_LOW> (n = {n_lo})",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#1f2937",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=8.8)
    ax.set_ylim(0, 113)
    ax.set_ylabel("Share of generations (%)")
    ax.set_title("Origin copy-number bucket shares: <COPY_HIGH> vs. <COPY_LOW>", pad=8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, "soft_token_copy_buckets", both=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_rigor, default_registry, default_out = resolve_default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigor-dir", type=Path, default=default_rigor)
    parser.add_argument("--registry", type=Path, default=default_registry)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = args.rigor_dir / "results_v1"
    annot_dir = args.rigor_dir / "annotations"

    obs_df = pd.read_parquet(results_dir / "all_observations.parquet")
    summary = json.loads((results_dir / "summary.json").read_text())
    mps = pd.read_csv(results_dir / "tables" / "main_panel_summary.csv")
    cat_df = pd.read_csv(results_dir / "tables" / "category_and_tag_breakdown.csv")
    tok_df = pd.read_csv(results_dir / "tables" / "per_token_breakdown.csv")
    min_df = pd.read_csv(results_dir / "tables" / "minority_panels_summary.csv")

    make_exemplar_gallery(obs_df, annot_dir, args.out_dir)
    make_bok_curve(summary, mps, args.out_dir)
    make_distribution_violins(obs_df, args.out_dir)
    make_threshold_ceiling(obs_df, args.out_dir)
    make_category_bars(cat_df, obs_df, args.out_dir)
    make_per_token_figure(tok_df, args.out_dir)
    make_soft_token_forest(obs_df, annot_dir, args.registry, args.out_dir)
    make_copy_buckets(obs_df, min_df, args.out_dir)
    print(f"Generated all 8 cluster_v3 figures in {args.out_dir}")


if __name__ == "__main__":
    main()
