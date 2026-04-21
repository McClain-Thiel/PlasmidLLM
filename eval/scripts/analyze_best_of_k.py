# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=14.0", "numpy>=1.26", "statsmodels>=0.14"]
# ///
"""Best-of-K analysis, paired, base vs GRPO.

Reward proxy: per-prompt pLannotate n_found (requested-features hit count).
Ties resolved by plausibility_pass then by higher fidelity rate.

Because the ranking uses ground-truth pLannotate output, this is an *oracle*
upper bound on what rejection sampling can achieve — a real sampler needs a
cheaper proxy and would do worse.

For K' ∈ {1, 2, 4} we take the best-of-K' sample per prompt:
  K=1: seed 42 alone (matched-config runs already in the paper).
  K=2: max over seeds 42, 43.
  K=4: max over seeds 42, 43, 44, 45.

Cache layout assumed: each seed's run directory sits under
`/tmp/plasmidlm_eval_cache/` and carries the standard
`metrics/tier3_essentials.parquet`, `metrics/prompt_fidelity_per_seq.parquet`,
and `generations.parquet` files (`hf buckets sync` from the PlasmidLMEval bucket
lays them out this way).

Usage:
    uv run --script eval/scripts/analyze_best_of_k.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

CACHE = Path("/tmp/plasmidlm_eval_cache")

SEEDS = {
    "base": {
        42: "runs__base_kmer6_matched_20260420_1303",
        43: "runs__bon_base_seed43_20260420_1303",
        44: "runs__bon_base_seed44_20260420_1303",
        45: "runs__bon_base_seed45_20260420_1303",
    },
    "grpo": {
        42: "runs__grpo_plannotate_full_20260408",
        43: "runs__bon_grpo_seed43_20260420_1303",
        44: "runs__bon_grpo_seed44_20260420_1303",
        45: "runs__bon_grpo_seed45_20260420_1303",
    },
}


def load_seed(model: str, seed: int) -> pd.DataFrame:
    d = CACHE / SEEDS[model][seed]
    t3 = pd.read_parquet(d / "metrics/tier3_essentials.parquet").reset_index(drop=True)
    fid = pd.read_parquet(d / "metrics/prompt_fidelity_per_seq.parquet").reset_index(drop=True)
    gen = pd.read_parquet(d / "generations.parquet").reset_index(drop=True)
    df = pd.DataFrame({
        "prompt": gen["prompt"].values,
        "plaus":  t3["plausibility_pass"].astype(bool).values,
        "n_req":  fid["n_requested"].values,
        "n_fnd":  fid["n_found"].values,
        "fid":    np.where(fid["n_requested"] > 0,
                           fid["n_found"] / np.maximum(fid["n_requested"], 1),
                           np.nan),
    })
    df["useful_T50"] = df["plaus"] & (df["fid"] >= 0.5)
    return df


def best_of_k(per_seed: list[pd.DataFrame], K: int,
              rank_by: str = "n_fnd") -> pd.DataFrame:
    """For each prompt, take the row with max rank_by across the first K seeds.

    Tie-break: prefer plaus=True, then higher fid.
    """
    assert len(per_seed) >= K
    subset = per_seed[:K]
    # Sanity: all share the same prompt order.
    for s in subset[1:]:
        assert (s["prompt"].values == subset[0]["prompt"].values).all()
    n = len(subset[0])
    # Stack values, take argmax with tie-breaking on (n_fnd, plaus, fid).
    stack = np.stack([s[rank_by].values for s in subset], axis=0)    # (K, n)
    plaus = np.stack([s["plaus"].values.astype(int) for s in subset], axis=0)
    fid = np.stack([np.nan_to_num(s["fid"].values, nan=-1) for s in subset], axis=0)
    # Composite key: (n_fnd, plaus, fid) descending.
    key = stack * 1e6 + plaus * 1e3 + fid
    best_idx = key.argmax(axis=0)  # (n,)
    rows = []
    for i in range(n):
        src = subset[best_idx[i]].iloc[i]
        rows.append({
            "prompt": src["prompt"],
            "plaus":  bool(src["plaus"]),
            "n_req":  int(src["n_req"]),
            "n_fnd":  int(src["n_fnd"]),
            "fid":    float(src["fid"]) if pd.notna(src["fid"]) else np.nan,
            "useful_T50": bool(src["useful_T50"]),
            "from_seed_idx": int(best_idx[i]),
        })
    return pd.DataFrame(rows)


def rate_ci(vec: np.ndarray) -> tuple[float, float, float]:
    k = int(vec.sum()); n = len(vec)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, float(lo), float(hi)


base_per_seed = [load_seed("base", s) for s in (42, 43, 44, 45)]
grpo_per_seed = [load_seed("grpo", s) for s in (42, 43, 44, 45)]

print("=== Per-seed baseline rates ===")
print("        base     GRPO     Δ")
for metric in ("plaus", "useful_T50"):
    for seed in (42, 43, 44, 45):
        b = base_per_seed[seed - 42][metric].values
        g = grpo_per_seed[seed - 42][metric].values
        print(f"  {metric:10s}  seed={seed}: "
              f"base {b.mean():.3%}  grpo {g.mean():.3%}  Δ {g.mean() - b.mean():+.3%}")

print()
print("=== best-of-K (oracle, ranked by pLannotate n_found) ===")
rng = np.random.default_rng(2026)
B = 10_000
for K in (1, 2, 4):
    base_k = best_of_k(base_per_seed, K)
    grpo_k = best_of_k(grpo_per_seed, K)
    for metric in ("plaus", "useful_T50"):
        bv = base_k[metric].values
        gv = grpo_k[metric].values
        br, blo, bhi = rate_ci(bv)
        gr, glo, ghi = rate_ci(gv)
        idx = rng.integers(0, len(bv), size=(B, len(bv)))
        boot = gv[idx].mean(axis=1) - bv[idx].mean(axis=1)
        # McNemar
        tab = pd.crosstab(pd.Series(bv, name="b"), pd.Series(gv, name="g"))
        tab = tab.reindex(index=[False, True], columns=[False, True], fill_value=0)
        mc = mcnemar(tab.values,
                     exact=(tab.values.min() < 25),
                     correction=True)
        print(f"  K={K}  {metric:10s}  base {br:.1%} [{blo:.1%},{bhi:.1%}]  "
              f"grpo {gr:.1%} [{glo:.1%},{ghi:.1%}]  "
              f"Δ {gr-br:+.1%}  boot [{np.quantile(boot,0.025):+.1%},{np.quantile(boot,0.975):+.1%}]  "
              f"McNemar p={mc.pvalue:.2g}")

print()
print("=== fidelity per-seq mean, best-of-K ===")
for K in (1, 2, 4):
    base_k = best_of_k(base_per_seed, K)
    grpo_k = best_of_k(grpo_per_seed, K)
    bf = base_k.loc[base_k["n_req"] > 0, "fid"].values
    gf = grpo_k.loc[grpo_k["n_req"] > 0, "fid"].values
    idx = rng.integers(0, len(bf), size=(B, len(bf)))
    boot = gf[idx].mean(axis=1) - bf[idx].mean(axis=1)
    print(f"  K={K}  base {bf.mean():.3%}   grpo {gf.mean():.3%}   "
          f"Δ {gf.mean()-bf.mean():+.3%}  boot [{np.quantile(boot,0.025):+.3%},{np.quantile(boot,0.975):+.3%}]")
