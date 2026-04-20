# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23",
#     "pandas>=2.0",
#     "pyarrow>=14.0",
#     "numpy>=1.26",
#     "statsmodels>=0.14",
#     "matplotlib>=3.7",
#     "huggingface_hub>=1.0",
# ]
# ///
"""Base vs GRPO-plannotate — paired analysis for the NeurIPS submission.

The story in one paragraph
--------------------------
Two 1000-sequence generation runs using identical prompts and identical sampling
config (temp=0.7, top_p=0.95, top_k=50, seed=42, max_new_tokens=3000) — one from
the pretrained base `McClain/PlasmidLM-kmer6`, one from the post-trained
`McClain/PlasmidLM-kmer6-GRPO-plannotate`. The GRPO model's headline HF-card
claim of "+8.8 pp overall hit rate" was on a 50-prompt best-of-3 setup with no
matched paired baseline. Replicating under matched paired sampling at n=1000:
the plausibility delta is not significant, the prompt-fidelity delta is, and
the composite "plausible AND on-prompt" metric is where GRPO's real gain lives
(+9.8 pp at the 50%-fidelity threshold, p ≈ 6×10⁻⁶).

Data provenance
---------------
Both runs live under `hf://buckets/McClain/PlasmidLMEval/runs/`:
    grpo_plannotate_full_20260408/     (generated 2026-04-08, annotated 2026-04-09)
    base_kmer6_matched_20260420_1303/  (generated + annotated 2026-04-20)

Prompts are identical, extracted from the GRPO run's `generations.parquet` into
`eval/data/paired_prompts_20260420.parquet`. Modeling code was held identical
across both runs (the base HF repo's `modeling_plasmid_lm.py` predates the
newer inference signature — we swapped in the GRPO-repo version without
touching weights).

All analysis here reads straight from bucket parquets; no hidden state.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.contingency_tables import mcnemar

    return mcnemar, mo, np, pd, plt, proportion_confint


@app.cell
def _(mo):
    mo.md(rf"""
    # Base vs GRPO-plannotate — paired analysis

    **Paired design.** Same 1000 prompts for both models, matched sampling config,
    identical annotation pipeline. Differences in outcomes are attributable to the
    GRPO post-training step.

    Contents:

    1. Load matched runs from the HF bucket.
    2. **Plausibility** — Wilson CIs, paired McNemar, paired bootstrap.
    3. **Prompt fidelity** (sseqid intersection via motif registry) — same battery.
    4. **Composite "useful plasmid"** — plausibility AND fidelity ≥ T. This is the
       paper figure.
    5. Per-category fidelity deltas.
    6. Base temperature sweep (context: base at its best vs matched).
    7. Discriminator AUC — base → GRPO collapses from 1.0 to 0.937.
    8. Memorization, MFE density, GC skew — distributional comparisons.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import subprocess

    BUCKET = "McClain/PlasmidLMEval"
    GRPO_RUN = "runs/grpo_plannotate_full_20260408"
    BASE_RUN = "runs/base_kmer6_matched_20260420_1303"

    # Base temperature sweep cells (paired on random 100-prompt subsets, not paired across models).
    SWEEP_CELLS = {
        "t=0.3": "runs/base_kmer6_sweep_20260420_1303/t03",
        "t=0.5": "runs/base_kmer6_sweep_20260420_1303/t05",
        "t=1.0": "runs/base_kmer6_sweep_20260420_1303/t10",
    }

    CACHE = Path("/tmp/plasmidlm_eval_cache")
    CACHE.mkdir(exist_ok=True)

    WANTED_MAIN = [
        "generations.parquet",
        "metrics/tier3_essentials.parquet",
        "metrics/tier1_distributional.parquet",
        "metrics/memorization.parquet",
        "metrics/extra_metrics.json",
        "metrics/prompt_fidelity_per_seq.parquet",
        "metrics/discriminator.json",
        "config.json",
        "generation_summary.json",
    ]
    WANTED_SWEEP = [
        "generations.parquet",
        "metrics/tier3_essentials.parquet",
        "metrics/prompt_fidelity_per_seq.parquet",
        "metrics/extra_metrics.json",
        "metrics/discriminator.json",
    ]

    # The notebook reads from a local cache of bucket files. To populate it:
    #
    #   hf buckets sync hf://buckets/McClain/PlasmidLMEval/runs/grpo_plannotate_full_20260408 \
    #       /tmp/plasmidlm_eval_cache/runs__grpo_plannotate_full_20260408
    #   hf buckets sync hf://buckets/McClain/PlasmidLMEval/runs/base_kmer6_matched_20260420_1303 \
    #       /tmp/plasmidlm_eval_cache/runs__base_kmer6_matched_20260420_1303
    #   (same pattern for the three sweep cells)
    #
    # The cell below tries `hf buckets cp` for any missing file and raises a clear
    # error if the CLI isn't available. We only shell out if files are missing.
    def _ensure_local(run_key: str, wanted: list[str]):
        local_run = CACHE / run_key.replace("/", "__")
        local_run.mkdir(exist_ok=True, parents=True)
        missing = [f for f in wanted if not (local_run / f).exists()]
        if missing:
            # Best-effort: try `hf buckets sync` to populate. If not installed, error clearly.
            src = f"hf://buckets/{BUCKET}/{run_key}"
            proc = subprocess.run(["hf", "buckets", "sync", src, str(local_run)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Missing local files under {local_run} and `hf buckets sync` failed:\n"
                    f"{proc.stderr}\n"
                    "Populate the cache manually:\n"
                    f"    hf buckets sync {src} {local_run}"
                )
        return local_run

    grpo_dir = _ensure_local(GRPO_RUN, WANTED_MAIN)
    base_dir = _ensure_local(BASE_RUN, WANTED_MAIN)
    sweep_dirs = {k: _ensure_local(v, WANTED_SWEEP) for k, v in SWEEP_CELLS.items()}
    return base_dir, grpo_dir, sweep_dirs


@app.cell
def _(base_dir, grpo_dir, pd, sweep_dirs):
    import json

    def _load(d):
        out = {
            "gens":  pd.read_parquet(d / "generations.parquet").reset_index(drop=True),
            "tier3": pd.read_parquet(d / "metrics/tier3_essentials.parquet").reset_index(drop=True),
        }
        fid = d / "metrics/prompt_fidelity_per_seq.parquet"
        if fid.exists():
            out["fidelity"] = pd.read_parquet(fid).reset_index(drop=True)
        tier1 = d / "metrics/tier1_distributional.parquet"
        if tier1.exists():
            out["tier1"] = pd.read_parquet(tier1).reset_index(drop=True)
        extra = d / "metrics/extra_metrics.json"
        if extra.exists():
            out["extra"] = json.loads(extra.read_text())
        disc = d / "metrics/discriminator.json"
        if disc.exists():
            out["disc"] = json.loads(disc.read_text())
        mem = d / "metrics/memorization.parquet"
        if mem.exists():
            out["memorization"] = pd.read_parquet(mem).reset_index(drop=True)
        return out

    grpo = _load(grpo_dir)
    base = _load(base_dir)
    sweep = {k: _load(v) for k, v in sweep_dirs.items()}

    assert len(grpo["gens"]) == len(base["gens"]) == 1000
    assert (grpo["gens"]["prompt"].values == base["gens"]["prompt"].values).all(), \
        "prompt order mismatch — runs are not paired"
    return base, grpo, sweep


@app.cell
def _(mo):
    mo.md("""
    ## 1. Wilson CIs on the two headline metrics
    """)
    return


@app.cell
def _(base, grpo, np, pd, proportion_confint):
    def _wilson(k: int, n: int) -> dict:
        _lo, _hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        return {"rate": k / n, "ci_low": float(_lo), "ci_high": float(_hi), "k": k, "n": n}

    _plaus_rows = []
    for _name, _r in [("base", base), ("grpo", grpo)]:
        _k = int(_r["tier3"]["plausibility_pass"].sum())
        _plaus_rows.append({"run": _name, "metric": "plausibility_pass",
                            **_wilson(_k, len(_r["tier3"]))})

    _fid_rows = []
    _rng_fid = np.random.default_rng(0)
    _B_fid = 20_000
    for _name, _r in [("base", base), ("grpo", grpo)]:
        _fid = _r["fidelity"]
        _with = _fid[_fid["n_requested"] > 0]
        _rates = (_with["n_found"] / _with["n_requested"]).to_numpy()
        _idx = _rng_fid.integers(0, len(_rates), size=(_B_fid, len(_rates)))
        _boot = _rates[_idx].mean(axis=1)
        _fid_rows.append({
            "run": _name, "metric": "prompt_fidelity_per_seq_mean",
            "rate": float(_rates.mean()),
            "ci_low": float(np.quantile(_boot, 0.025)),
            "ci_high": float(np.quantile(_boot, 0.975)),
            "k": None, "n": int(len(_rates)),
        })

    ci_table = pd.DataFrame(_plaus_rows + _fid_rows)
    ci_table[["run", "metric", "rate", "ci_low", "ci_high", "k", "n"]]
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Paired McNemar + paired bootstrap on the two deltas

    McNemar is the right test for matched-pair binary outcomes. Bootstrap gives us a
    CI on the delta that respects the pairing (resample prompt indices, recompute Δ).
    """)
    return


@app.cell
def _(base, grpo, mcnemar, np, pd):
    _rng = np.random.default_rng(0)
    _B = 20_000

    _bp = base["tier3"]["plausibility_pass"].astype(bool).to_numpy()
    _gp = grpo["tier3"]["plausibility_pass"].astype(bool).to_numpy()

    tab_plaus = pd.crosstab(pd.Series(_bp, name="base"), pd.Series(_gp, name="grpo"))
    mc_plaus = mcnemar(tab_plaus.values, exact=False, correction=True)

    _idx = _rng.integers(0, len(_bp), size=(_B, len(_bp)))
    _boot_plaus = _gp[_idx].mean(axis=1) - _bp[_idx].mean(axis=1)

    _mask = (base["fidelity"]["n_requested"] > 0).to_numpy() & \
            (grpo["fidelity"]["n_requested"] > 0).to_numpy()
    _bfid = (base["fidelity"]["n_found"] / base["fidelity"]["n_requested"]).to_numpy()[_mask]
    _gfid = (grpo["fidelity"]["n_found"] / grpo["fidelity"]["n_requested"]).to_numpy()[_mask]

    _idx2 = _rng.integers(0, len(_bfid), size=(_B, len(_bfid)))
    _boot_fid = _gfid[_idx2].mean(axis=1) - _bfid[_idx2].mean(axis=1)

    paired_summary = pd.DataFrame([
        {"metric": "plausibility_pass",
         "delta": float(_gp.mean() - _bp.mean()),
         "boot_ci_low":  float(np.quantile(_boot_plaus, 0.025)),
         "boot_ci_high": float(np.quantile(_boot_plaus, 0.975)),
         "mcnemar_p": float(mc_plaus.pvalue)},
        {"metric": "prompt_fidelity_per_seq",
         "delta": float(_gfid.mean() - _bfid.mean()),
         "boot_ci_low":  float(np.quantile(_boot_fid, 0.025)),
         "boot_ci_high": float(np.quantile(_boot_fid, 0.975)),
         "mcnemar_p": None},
    ])
    paired_summary
    return mc_plaus, tab_plaus


@app.cell
def _(mc_plaus, mo, tab_plaus):
    mo.md(f"""
    **Plausibility contingency (base × GRPO).** Note the near-symmetric off-diagonal:
    GRPO saved **{tab_plaus.loc[False, True]}** base failures but *broke*
    **{tab_plaus.loc[True, False]}** base passes. The net is small and the McNemar
    p is {mc_plaus.pvalue:.3f} — GRPO shuffles plausibility, it doesn't uniformly lift it.
    """)
    return


@app.cell
def _(tab_plaus):
    tab_plaus
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The paper figure: plausible **and** fidelity ≥ T

    A structurally plausible plasmid that lacks the requested features is not useful.
    The right metric is the *intersection*. Sweeping the fidelity threshold shows where
    GRPO's real gain lives.
    """)
    return


@app.cell
def _(base, grpo, mcnemar, np, pd, proportion_confint):
    thresholds = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]

    def _useful_mask(run, t):
        _plaus = run["tier3"]["plausibility_pass"].astype(bool).to_numpy()
        _fid = run["fidelity"]
        _has_tokens = (_fid["n_requested"] > 0).to_numpy()
        _rate = np.where(_has_tokens,
                         _fid["n_found"].to_numpy() / np.maximum(_fid["n_requested"].to_numpy(), 1),
                         np.nan)
        if t == 0.0:
            return _plaus
        return _plaus & (np.nan_to_num(_rate, nan=-1.0) >= t - 1e-9)

    _rng = np.random.default_rng(1)
    _B = 20_000
    _rows = []
    for _t in thresholds:
        _bm = _useful_mask(base, _t)
        _gm = _useful_mask(grpo, _t)
        _bn, _bk = len(_bm), int(_bm.sum())
        _gn, _gk = len(_gm), int(_gm.sum())
        _blo, _bhi = proportion_confint(_bk, _bn, alpha=0.05, method="wilson")
        _glo, _ghi = proportion_confint(_gk, _gn, alpha=0.05, method="wilson")

        _tab = pd.crosstab(pd.Series(_bm, name="base"), pd.Series(_gm, name="grpo"))
        _tab = _tab.reindex(index=[False, True], columns=[False, True], fill_value=0)
        _exact = _tab.values.min() < 25
        _mc = mcnemar(_tab.values, exact=_exact, correction=True)

        _idx = _rng.integers(0, _bn, size=(_B, _bn))
        _boot = _gm[_idx].mean(axis=1) - _bm[_idx].mean(axis=1)

        _rows.append({
            "T": _t,
            "base_rate": _bk / _bn, "base_ci_low": _blo, "base_ci_high": _bhi,
            "grpo_rate": _gk / _gn, "grpo_ci_low": _glo, "grpo_ci_high": _ghi,
            "delta": _gk / _gn - _bk / _bn,
            "boot_ci_low":  float(np.quantile(_boot, 0.025)),
            "boot_ci_high": float(np.quantile(_boot, 0.975)),
            "mcnemar_p": float(_mc.pvalue),
        })
    intersection_tbl = pd.DataFrame(_rows)
    intersection_tbl
    return (intersection_tbl,)


@app.cell
def _(intersection_tbl, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))

    _T = intersection_tbl["T"].to_numpy()
    _ax1.plot(_T, intersection_tbl["base_rate"], "o-", color="#6b7280", label="base")
    _ax1.fill_between(_T, intersection_tbl["base_ci_low"], intersection_tbl["base_ci_high"],
                      color="#6b7280", alpha=0.2)
    _ax1.plot(_T, intersection_tbl["grpo_rate"], "o-", color="#059669", label="grpo-plannotate")
    _ax1.fill_between(_T, intersection_tbl["grpo_ci_low"], intersection_tbl["grpo_ci_high"],
                      color="#059669", alpha=0.2)
    _ax1.set_xlabel("fidelity threshold T")
    _ax1.set_ylabel("fraction plausible AND fidelity ≥ T")
    _ax1.set_title("Useful plasmid rate (paired n=1000)")
    _ax1.legend()
    _ax1.grid(alpha=0.3)

    _ax2.plot(_T, intersection_tbl["delta"], "o-", color="#0369a1")
    _ax2.fill_between(_T, intersection_tbl["boot_ci_low"], intersection_tbl["boot_ci_high"],
                      color="#0369a1", alpha=0.2)
    _ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    _ax2.set_xlabel("fidelity threshold T")
    _ax2.set_ylabel("Δ (grpo − base), paired bootstrap 95% CI")
    _ax2.set_title("Effect size of GRPO vs T")
    _ax2.grid(alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Per-category fidelity deltas

    Decompose the +6.1 pp overall fidelity gain by feature category.
    """)
    return


@app.cell
def _(base, grpo, pd, proportion_confint):
    categories = ["ORI", "AMR", "PROM", "ELEM", "REPORTER", "TAG"]
    _cat_rows = []
    for _cat in categories:
        _col_req = f"n_requested_{_cat}"
        _col_fnd = f"n_found_{_cat}"
        if _col_req not in base["fidelity"].columns:
            continue
        for _name, _r in [("base", base), ("grpo", grpo)]:
            _k = int(_r["fidelity"][_col_fnd].sum())
            _n = int(_r["fidelity"][_col_req].sum())
            if _n == 0:
                continue
            _lo, _hi = proportion_confint(_k, _n, alpha=0.05, method="wilson")
            _cat_rows.append({"category": _cat, "run": _name,
                              "rate": _k / _n, "k": _k, "n": _n,
                              "ci_low": float(_lo), "ci_high": float(_hi)})
    category_tbl = pd.DataFrame(_cat_rows)
    category_tbl
    return categories, category_tbl


@app.cell
def _(categories, category_tbl, np, plt):
    _pivoted = category_tbl.pivot(index="category", columns="run", values="rate").reindex(categories)
    _lo = category_tbl.pivot(index="category", columns="run", values="ci_low").reindex(categories)
    _hi = category_tbl.pivot(index="category", columns="run", values="ci_high").reindex(categories)

    _x = np.arange(len(categories))
    _w = 0.35
    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _offset, _run, _color in [(-_w/2, "base", "#6b7280"), (+_w/2, "grpo", "#059669")]:
        _vals = _pivoted[_run].values
        _yerr = np.vstack([_vals - _lo[_run].values, _hi[_run].values - _vals])
        _ax.bar(_x + _offset, _vals, width=_w, label=_run, color=_color,
                yerr=_yerr, capsize=3, edgecolor="white")
    _ax.set_xticks(_x); _ax.set_xticklabels(categories)
    _ax.set_ylabel("fidelity (token-pooled)"); _ax.set_ylim(0, 0.7)
    _ax.set_title("Per-category fidelity, Wilson 95% CIs")
    _ax.legend(); _ax.grid(alpha=0.3, axis="y")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    **TAG regression note.** Base TAG hit rate > GRPO TAG hit rate — the only
    category where GRPO loses. TAG tokens are tiny peptide sequences (6–18 bp)
    that BLAST struggles to find, so the training scorer rarely rewards them.
    RL therefore learns to deprioritise them. Classic Goodhart's law. Worth
    flagging as a limitation in the paper.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Base temperature sweep

    Three additional 100-seq cells on the base model at t∈{0.3, 0.5, 1.0}. Context:
    is the matched (t=0.7) comparison unfair to the base?
    """)
    return


@app.cell
def _(base, grpo, pd, sweep):
    def _rates(r):
        _t3 = r["tier3"]
        _fid = r.get("fidelity")
        _plaus = float(_t3["plausibility_pass"].mean())
        if _fid is not None and (_fid["n_requested"] > 0).any():
            _sub = _fid[_fid["n_requested"] > 0]
            _fidrate = float((_sub["n_found"] / _sub["n_requested"]).mean())
        else:
            _fidrate = None
        _disc_auc = None
        if "disc" in r and isinstance(r["disc"], dict):
            _disc_auc = r["disc"].get("auc")
        return {"plausibility": _plaus, "fidelity_per_seq_mean": _fidrate,
                "discriminator_auc": _disc_auc, "n": len(_t3)}

    _sweep_rows = [
        {"run": "base t=0.3",            **_rates(sweep["t=0.3"])},
        {"run": "base t=0.5",            **_rates(sweep["t=0.5"])},
        {"run": "base t=0.7 (matched)",  **_rates(base)},
        {"run": "base t=1.0",            **_rates(sweep["t=1.0"])},
        {"run": "grpo t=0.7 (matched)",  **_rates(grpo)},
    ]
    sweep_tbl = pd.DataFrame(_sweep_rows)
    sweep_tbl
    return


@app.cell
def _(mo):
    mo.md("""
    Even at its best temperature (0.3), base fidelity (37.8%) is still below GRPO's
    41.6%. GRPO's advantage on fidelity is not explained away by sampling temperature.

    ## 6. Discriminator AUC

    A LightGBM classifier distinguishes real Addgene plasmids from generated ones.
    """)
    return


@app.cell
def _(base, grpo, pd):
    _disc_rows = []
    for _name, _r in [("base t=0.7 (matched)", base), ("grpo t=0.7 (matched)", grpo)]:
        _auc = _r.get("disc", {}).get("auc") if isinstance(_r.get("disc"), dict) else None
        _top = _r.get("disc", {}).get("top_features", []) if isinstance(_r.get("disc"), dict) else []
        _disc_rows.append({
            "run": _name,
            "auc": _auc,
            "top_features": ", ".join(f"{_f}({_v:.0f})" for _f, _v in (_top or [])[:5]) if _top else None,
        })
    pd.DataFrame(_disc_rows)
    return


@app.cell
def _(mo):
    mo.md("""
    Base AUC = 1.000 (trivially separable). GRPO AUC = 0.937. GRPO moves the
    distribution away from the easiest-to-detect pretrained signature. Next step
    for the paper: verify the top SHAP features to confirm this isn't a trivial
    length or GC artifact.

    ## 7. Memorization

    sourmash containment against `eval/reference/training_sigs.zip` at threshold 0.3.
    """)
    return


@app.cell
def _(base, grpo):
    def _memo_summary(r):
        _mem = r.get("memorization")
        if _mem is None or len(_mem) == 0:
            return {"n_hits_at_0.3": 0, "max_containment": None}
        return {"n_hits_at_0.3": int((_mem["containment"] >= 0.3).sum()),
                "max_containment": float(_mem["containment"].max())}

    {"base": _memo_summary(base), "grpo": _memo_summary(grpo)}
    return


@app.cell
def _(mo):
    mo.md("""
    Zero hits for both models at containment ≥ 0.3. No evidence of training-set
    memorization in either.

    ## 8. Distributional checks — length, GC, MFE

    Sanity-check that GRPO didn't collapse the distribution.
    """)
    return


@app.cell
def _(base, grpo, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.5))

    def _overlay(ax, col, title, bins=40, clip=None):
        _b = base["tier1"][col].values
        _g = grpo["tier1"][col].values
        _lo = min(_b.min(), _g.min()); _hi = max(_b.max(), _g.max())
        if clip is not None:
            _lo = max(_lo, clip[0]); _hi = min(_hi, clip[1])
        ax.hist(_b, bins=bins, range=(_lo, _hi), alpha=0.55, color="#6b7280", label="base")
        ax.hist(_g, bins=bins, range=(_lo, _hi), alpha=0.55, color="#059669", label="grpo")
        ax.set_title(title); ax.legend()

    _overlay(_axes[0], "length", "sequence length (bp)", clip=(0, 20_000))
    _overlay(_axes[1], "gc", "GC fraction")
    _remaining = [_c for _c in ["longest_orf", "n_features", "orf_density"]
                  if _c in base["tier1"].columns]
    if _remaining:
        _overlay(_axes[2], _remaining[0], _remaining[0])
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(base, grpo):
    def _mfe(r):
        _ex = r.get("extra") or {}
        return _ex.get("mfe_density", {})

    {"base": _mfe(base), "grpo": _mfe(grpo)}
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paper summary (the TL;DR cell)

    | Claim | Value | Evidence |
    |---|---|---|
    | Plausibility improvement from GRPO is not significant | +2.8 pp, 95% CI [−1.4, +7.1], McNemar p=0.21 | cells 1–2 |
    | Prompt-fidelity improvement is significant | +6.1 pp (per-seq mean), paired bootstrap 95% CI [+3.4, +8.8] | cell 2 |
    | **Composite "useful plasmid" improvement is large and highly significant** | **+9.8 pp at T=0.5 fidelity, McNemar p ≈ 6×10⁻⁶** | cell 3 |
    | Per-category gains: ORI, ELEM, PROM, AMR, REPORTER; regression on TAG | +7.8, +5.0, +4.2, +3.0, +2.0, −5.6 pp | cell 4 |
    | Base already performs best at t=0.3, not t=0.7 | fidelity 37.8% at 0.3 vs 35.5% at matched 0.7 | cell 5 |
    | GRPO reduces trivial separability from real Addgene | AUC 1.000 → 0.937 | cell 6 |
    | No training-set memorization (base or GRPO) | 0 hits at containment ≥ 0.3 | cell 7 |
    """)
    return


if __name__ == "__main__":
    app.run()
