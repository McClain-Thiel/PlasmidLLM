"""Base vs GRPO-plannotate comparison — bootstrap CIs, Wilson intervals, paired McNemar.

Compares two eval runs stored in the `McClain/PlasmidLMEval` HF bucket:

    GRPO run (n=1000)  :  runs/grpo_plannotate_full_20260408/
    Base run  (n=1000) :  runs/base_kmer6_matched_<STAMP>/      (produced 2026-04-20)

Both runs use the *same* 1000 paired prompts (from the GRPO run's `generations.parquet`),
the *same* sampling config (temp=0.7, top_p=0.95, top_k=50, seed=42, max_new_tokens=3000),
and the *same* downstream annotation pipeline (pLannotate + prodigal + dustmasker, identical
pins). Any difference in outcomes is attributable to the GRPO post-training step.

The pLannotate run for the base model was launched from
`/opt/dlami/nvme/eval_output/base_kmer6_matched_20260420_1303` on `g6-big` and synced
into the bucket once complete.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Base vs GRPO-plannotate

        **Paired design.** Both models generated one sequence per prompt for the same 1000
        prompts, at identical sampling config. The headline question is: does GRPO training
        against the pLannotate signal reliably improve the fraction of generated sequences
        that are biologically plausible, and if so by how much — with confidence?

        This notebook computes:

        1. Wilson intervals for overall `plausibility_pass` and each category hit rate.
        2. Paired McNemar test between base and GRPO on prompt-level pass / fail.
        3. Bootstrap CIs on the base–GRPO *difference* (aligned per prompt, so the bootstrap
           respects pairing).
        4. Per-category breakdown with CI bars.
        """
    )
    return


@app.cell
def _():
    # --- Bucket paths and local cache -----------------------------------------
    # We download metrics + per-sequence parquets once, then work off the local cache.
    from pathlib import Path

    BUCKET = "McClain/PlasmidLMEval"
    GRPO_RUN = "runs/grpo_plannotate_full_20260408"
    # Base run: produced on 2026-04-20 on g6-big. The timestamp suffix is the HHMM at launch
    # (stored in the run dir's config.json under `timestamp`). Update here if re-running.
    BASE_RUN = "runs/base_kmer6_matched_20260420_1303"

    CACHE = Path("/tmp/plasmidlm_eval_cache")
    CACHE.mkdir(exist_ok=True)
    return BASE_RUN, BUCKET, CACHE, GRPO_RUN, Path


@app.cell
def _(BASE_RUN, BUCKET, CACHE, GRPO_RUN):
    # Download only the parquets we need. Generations + tier3 (plausibility + category hits).
    import subprocess

    wanted = [
        "generations.parquet",
        "metrics/tier3_essentials.parquet",
        "metrics/tier5_architecture.parquet",
        "metrics/memorization.parquet",
        "config.json",
        "generation_summary.json",
    ]
    for run in (GRPO_RUN, BASE_RUN):
        local_run = CACHE / run.replace("/", "__")
        local_run.mkdir(exist_ok=True, parents=True)
        for f in wanted:
            dst = local_run / f.replace("/", "__")
            if dst.exists():
                continue
            src = f"hf://buckets/{BUCKET}/{run}/{f}"
            # `hf buckets cp` writes the file verbatim, no prefix munging.
            subprocess.run(["hf", "buckets", "cp", src, str(dst)], check=True)
    return


@app.cell
def _(BASE_RUN, CACHE, GRPO_RUN):
    import pandas as pd

    def load_run(run_key: str) -> dict:
        d = CACHE / run_key.replace("/", "__")
        gens = pd.read_parquet(d / "generations.parquet")
        tier3 = pd.read_parquet(d / "metrics__tier3_essentials.parquet")
        return {"gens": gens, "tier3": tier3, "dir": d}

    grpo = load_run(GRPO_RUN)
    base = load_run(BASE_RUN)

    # tier3 must contain plausibility_pass and the per-category has_* flags.
    assert "plausibility_pass" in grpo["tier3"].columns, "tier3.parquet missing plausibility_pass"
    assert len(grpo["gens"]) == len(base["gens"]) == 1000, "expected 1000 paired rows per run"
    return base, grpo, load_run, pd


@app.cell
def _(mo):
    mo.md("## 1. Wilson intervals (per-run, per-category)")
    return


@app.cell
def _(base, grpo, pd):
    from statsmodels.stats.proportion import proportion_confint

    def _wilson(df: pd.DataFrame, col: str) -> dict:
        k = int(df[col].sum())
        n = int(df[col].notna().sum())
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        return {"n": n, "k": k, "rate": k / n, "ci_low": lo, "ci_high": hi}

    cols = [c for c in ["plausibility_pass", "has_ori", "has_selection_marker",
                        "has_promoter", "has_terminator", "has_cds"]
            if c in grpo["tier3"].columns]

    rows = []
    for col in cols:
        rows.append({"run": "base",  "metric": col, **_wilson(base["tier3"], col)})
        rows.append({"run": "grpo",  "metric": col, **_wilson(grpo["tier3"], col)})
    wilson_tbl = pd.DataFrame(rows)
    wilson_tbl
    return cols, proportion_confint, wilson_tbl


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Paired McNemar on prompt-level viability

        Each row is the same prompt. We compare `plausibility_pass` between base and GRPO.
        The 2×2 table counts (pass_base, pass_grpo) ∈ {(0,0),(0,1),(1,0),(1,1)}. McNemar
        tests the off-diagonal asymmetry — i.e. whether GRPO flipping a fail→pass is more
        common than the reverse.
        """
    )
    return


@app.cell
def _(base, grpo, pd):
    from statsmodels.stats.contingency_tables import mcnemar

    # Align by prompt. The two runs were generated from the same paired_prompts file, so
    # both tier3 tables share the same row order (index 0..999 maps to the same prompt).
    # We sanity-check that ordering explicitly.
    b = base["gens"][["prompt"]].reset_index(drop=True)
    g = grpo["gens"][["prompt"]].reset_index(drop=True)
    assert (b["prompt"].values == g["prompt"].values).all(), \
        "prompt order mismatch — cannot do paired test without realignment"

    tb = base["tier3"]["plausibility_pass"].reset_index(drop=True).astype(bool)
    tg = grpo["tier3"]["plausibility_pass"].reset_index(drop=True).astype(bool)
    both = pd.crosstab(tb, tg, rownames=["base"], colnames=["grpo"])
    res = mcnemar(both.values, exact=False, correction=True)
    {"contingency_table": both.to_dict(),
     "mcnemar_statistic": float(res.statistic),
     "mcnemar_pvalue": float(res.pvalue)}
    return both, mcnemar, tb, tg


@app.cell
def _(mo):
    mo.md("## 3. Bootstrap CI on base–GRPO difference (paired)")
    return


@app.cell
def _(tb, tg):
    import numpy as np

    # Paired bootstrap: resample prompt indices with replacement, recompute the difference.
    rng = np.random.default_rng(0)
    B = 10_000
    diffs = np.empty(B)
    base_arr = tb.to_numpy().astype(float)
    grpo_arr = tg.to_numpy().astype(float)
    n = len(base_arr)
    idx = rng.integers(0, n, size=(B, n))
    diffs = grpo_arr[idx].mean(axis=1) - base_arr[idx].mean(axis=1)
    point = float(grpo_arr.mean() - base_arr.mean())
    ci = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))
    {"grpo_minus_base": point, "bootstrap_95ci": ci, "B": B}
    return B, ci, diffs, point, rng


@app.cell
def _(diffs, point):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(diffs, bins=60, edgecolor="white")
    ax.axvline(point, color="tab:red", linewidth=2, label=f"point estimate = {point:+.3f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", label="null (no effect)")
    ax.set_xlabel("plausibility_pass(GRPO) − plausibility_pass(base)")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title("Paired bootstrap (B = 10,000) over 1000 prompts")
    ax.legend()
    fig
    return ax, fig, plt


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Per-category hit rate with Wilson CIs

        One bar per (category, run) with the Wilson 95% interval.
        """
    )
    return


@app.cell
def _(cols, plt, wilson_tbl):
    import numpy as np

    wt = wilson_tbl.pivot(index="metric", columns="run", values="rate").loc[cols]
    lo = wilson_tbl.pivot(index="metric", columns="run", values="ci_low").loc[cols]
    hi = wilson_tbl.pivot(index="metric", columns="run", values="ci_high").loc[cols]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(cols))
    w = 0.35
    for offset, run, color in [(-w/2, "base", "#6b7280"), (+w/2, "grpo", "#059669")]:
        vals = wt[run].values
        yerr = np.vstack([vals - lo[run].values, hi[run].values - vals])
        ax.bar(x + offset, vals, width=w, label=run, color=color, yerr=yerr,
               capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-category hit rate (Wilson 95% CIs)")
    ax.legend()
    fig
    return ax, fig, hi, lo, wt, x


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Memorization check on the base run

        Uses sourmash containment against `eval/reference/training_sigs.zip`. GRPO's check
        (done in April) found zero hits at containment ≥ 0.3. The base model is the
        pretrained model, so any memorization would appear *here*, not in the GRPO run.
        """
    )
    return


@app.cell
def _(CACHE, pd):
    base_mem = pd.read_parquet(CACHE / "runs__base_kmer6_matched_20260420" / "metrics__memorization.parquet")
    grpo_mem = pd.read_parquet(CACHE / "runs__grpo_plannotate_full_20260408" / "metrics__memorization.parquet")
    summary = {
        "base_hits_>=0.3": int((base_mem["containment"] >= 0.3).sum()),
        "grpo_hits_>=0.3": int((grpo_mem["containment"] >= 0.3).sum()),
        "base_max_containment": float(base_mem["containment"].max()) if len(base_mem) else None,
        "grpo_max_containment": float(grpo_mem["containment"].max()) if len(grpo_mem) else None,
    }
    summary
    return base_mem, grpo_mem, summary


if __name__ == "__main__":
    app.run()
