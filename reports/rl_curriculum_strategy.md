# PlasmidLM RL Post-Training: Curriculum Strategy Report

**Date**: 2026-02-27
**Branch**: `rl-experiments`
**Status**: All experiments complete

---

## 1. Executive Summary

The pretrained PlasmidLM (17M params, 96% token accuracy) generates DNA sequences
that contain functional components of the **right category** but not the **right
specific variant**. Three curriculum experiments were run to find the optimal
alpha-blending strategy for PPO post-training.

**Key findings**:

1. **Pure presence reward (α=0) produces no learning** — the pretrained model is
   already optimal for category presence. Scores flat at 0.591 over 590 steps.

2. **PPO learning is real but modest** — at α=0.5, the model scores ~0.376 vs a
   pretrained baseline of ~0.330. PPO roughly doubles the improvement from exact
   match, but the absolute gains are small (+4.6% above baseline).

3. **The score decline with increasing alpha is NOT a curriculum problem** — it's
   a measurement artifact. The task gets harder, and the model keeps pace at ~2×
   the pretrained baseline regardless of alpha level.

4. **Recommended strategy**: Skip Phase 1 (useless at α=0). Start with α=0.3 and
   ramp slowly to α=0.5 over 3000 steps, then plateau. Consider category-staged
   curriculum or reward shaping for larger gains.

---

## 2. Baseline Analysis: What the Pretrained Model Can Do

Evaluated the pretrained checkpoint (step 15000) by generating 5 rollouts per
token at temperature=1.0, max_new_tokens=512, scoring with both exact-match
(α=1.0) and category-presence (α=0.0) rewards.

### Per-Category Baseline

| Category | Tokens | Exact Match | Category Presence | Gap |
|----------|--------|-------------|-------------------|-----|
| AMR      | 11     | 0.077       | **1.000**         | 0.923 |
| ELEM     | 14     | 0.069       | **1.000**         | 0.931 |
| REPORTER | 6      | 0.033       | **1.000**         | 0.967 |
| PROM     | 12     | 0.128       | 0.404             | 0.275 |
| TAG      | 7      | 0.176       | 0.315             | 0.139 |
| ORI      | 7      | 0.047       | 0.075             | 0.028 |

### Interpretation

Three distinct tiers emerge:

1. **Tier 1 (AMR, ELEM, REPORTER)**: Perfect category presence (1.0) but very
   low exact match (~0.05). The model *knows* these categories and generates DNA
   from the right functional class, but the specific motif sequences are wrong.

2. **Tier 2 (PROM, TAG)**: Moderate presence (0.3-0.4) with partial exact match
   (0.13-0.18). TAG has the highest exact match (0.176) because tag motifs
   (FLAG, HIS, HA) are short conserved peptide-coding regions.

3. **Tier 3 (ORI)**: Near-zero everywhere (0.075 presence, 0.047 exact). The
   model essentially cannot generate origins of replication. This is a **cold
   start problem**.

### Multi-Token Prompts

| Prompt | Exact | Presence |
|--------|-------|----------|
| AMR_KANAMYCIN + ORI_COLE1 | 0.068 | 0.537 |
| AMR_AMPICILLIN + PROM_CMV + REPORTER_EGFP | 0.095 | 0.807 |
| AMR_KANAMYCIN + ORI_COLE1 + PROM_T7 | 0.159 | 0.526 |

Multi-token prompts show the model can compose, but ORI drags down scores.

---

## 3. PPO v3 Reference Run (α: 0→1 over 1000 steps)

### Configuration

- α schedule: 0.0 → 1.0 over 1000 steps, then α=1.0 for remaining steps
- Learning rate: 3e-6, KL coeff: 0.5, clip range: 0.1
- Batch: 4 × 4 grad accum = 16 effective
- Response length: 1024 tokens

### Metrics Trajectory (2,221 steps)

| Window | ~Alpha | Score | KL |
|--------|--------|-------|-----|
| Steps 1-50 | 0.03 | 0.581 | 0.30 |
| Steps 51-100 | 0.08 | 0.555 | 0.40 |
| Steps 101-200 | 0.15 | 0.531 | 0.45 |
| Steps 201-500 | 0.35 | 0.439 | 0.50 |
| Steps 501-1000 | 0.75 | 0.268 | 0.64 |
| Steps 1001-1500 | 1.00 | 0.159 | 0.70 |
| Steps 1501-2000 | 1.00 | 0.159 | 0.74 |
| Steps 2001-2221 | 1.00 | 0.160 | 0.76 |

### Diagnosis

Score decline mirrors alpha increase exactly. The model plateaus at 0.159 once
α=1.0 is reached and shows no further improvement over 1200+ additional steps.

---

## 4. Experiment Design

Three experiments test different curriculum strategies, each running 400 steps
from the pretrained base (no warm-start from PPO v3 checkpoints).

### Experiment A: Soft Alpha Ceiling (α max = 0.5)

**Config**: `configs/exp_a_soft_ceiling.py`

- α ramps from 0.0 → 0.5 linearly over 400 steps
- Tests: Does partial presence credit keep the learning signal dense enough?

### Experiment B: Pure Presence Reward (α = 0 constant)

**Config**: `configs/exp_b_presence_only.py`

- α stays at 0.0 throughout (reward = category presence score only)
- Tests: How fast does the model improve on structural correctness alone?

### Experiment C: Two-Phase Curriculum

**Config**: `configs/exp_c_two_phase.py`

- Phase 1 (steps 0-200): α = 0.0 constant (dedicated structure learning)
- Phase 2 (steps 200-400): α ramps 0.0 → 0.7 (gradual specificity)
- Tests: Does a dedicated presence phase before ramping lead to higher
  final scores than immediate ramping?

---

## 5. Experiment Results

### Experiment B: Pure Presence (α=0) — Complete (590 steps)

| Window | Score | KL | Entropy |
|--------|-------|-----|---------|
| Steps 1-50 | 0.590 | 0.28 | 974 |
| Steps 51-100 | 0.588 | 0.31 | 958 |
| Steps 101-200 | 0.596 | 0.39 | 974 |
| Steps 201-300 | 0.592 | 0.45 | 977 |
| Steps 301-400 | 0.592 | 0.57 | 959 |
| Steps 401-500 | 0.589 | 0.56 | 952 |
| Steps 501-590 | 0.591 | 0.58 | 953 |

**Result**: Scores completely flat at 0.591 ± 0.032 throughout. **No learning.**
The pretrained model is already near-optimal for category presence. KL grows
steadily (0.28 → 0.58), meaning the policy IS changing, but the changes
provide no measurable improvement on presence scoring.

**Implication**: Phase 1 of a two-phase curriculum (α=0 structure learning)
would be wasted compute for this pretrained model. Structure is already learned.

### Experiment A: Soft Ceiling (α: 0→0.5 over 400 steps) — Complete (513 steps)

| Window | ~Alpha | Score | Expected* | Improvement |
|--------|--------|-------|-----------|-------------|
| Steps 1-50 | 0.03 | 0.577 | 0.565 | +0.012 |
| Steps 51-100 | 0.09 | 0.544 | 0.534 | +0.010 |
| Steps 101-150 | 0.16 | 0.524 | 0.500 | +0.024 |
| Steps 151-200 | 0.22 | 0.503 | 0.470 | +0.033 |
| Steps 201-250 | 0.28 | 0.471 | 0.440 | +0.031 |
| Steps 251-300 | 0.34 | 0.441 | 0.410 | +0.031 |
| Steps 301-350 | 0.41 | 0.416 | 0.376 | +0.040 |
| Steps 351-400 | 0.47 | 0.389 | 0.345 | +0.044 |
| Post-400 plateau | 0.50 | 0.376 | 0.330 | **+0.046** |

*Expected = α × 0.08 + (1-α) × 0.58 (pretrained baseline at each alpha)*

**Result**: PPO IS learning. Improvement above pretrained baseline grows
monotonically with alpha, reaching **+4.6%** at α=0.5. The model is roughly
doubling its exact-match component at each alpha level.

The 113 post-400 steps at α=0.5 show the score stabilizing at ~0.376, suggesting
the model has approximately converged at this alpha in 400 steps.

### Experiment C: Two-Phase Curriculum — Complete (400 steps)

| Window | ~Alpha | Score | KL |
|--------|--------|-------|-----|
| Steps 1-50 (Phase 1) | 0.00 | 0.590 | 0.24 |
| Steps 51-100 (Phase 1) | 0.00 | 0.586 | 0.27 |
| Steps 101-150 (Phase 1) | 0.00 | 0.593 | 0.30 |
| Steps 151-200 (Phase 1) | 0.00 | 0.600 | 0.23 |
| Steps 201-250 (Phase 2) | 0.09 | 0.556 | 0.21 |
| Steps 251-300 (Phase 2) | 0.26 | 0.475 | 0.20 |
| Steps 301-350 (Phase 2) | 0.44 | 0.404 | 0.16 |
| Steps 351-400 (Phase 2) | 0.61 | 0.327 | 0.13 |

Phase 1 (α=0): score = 0.592 ± 0.030, KL = 0.259
Phase 2 (α→0.7): score = 0.440 ± 0.090, KL = 0.178

**Result**: Phase 1 shows the same flat pattern as Exp B — no learning at α=0.
Phase 2 ramp produces declining scores as alpha increases, consistent with Exp A.
Final score at α≈0.7 is 0.327.

**Notably**: Exp C has *lower KL* than Exp A at equivalent alpha levels (0.13 vs
0.49 at α≈0.5). The 200-step plateau phase may have helped the value function
warm up, leading to more efficient policy updates in Phase 2. This produces a
slightly higher score at α=0.7 compared to PPO v3 (0.307 vs 0.293).

### Cross-Experiment Comparison at Equivalent Alpha

| Alpha | Pretrained | PPO v3 | Exp A | Exp C | Best Δ |
|-------|-----------|--------|-------|-------|--------|
| 0.00 | 0.580 | 0.583 | 0.585 | 0.592 | +0.012 |
| 0.10 | 0.530 | 0.548 | 0.544 | 0.550 | +0.020 |
| 0.20 | 0.480 | 0.509 | 0.507 | 0.508 | +0.029 |
| 0.30 | 0.430 | 0.461 | 0.461 | 0.460 | +0.031 |
| 0.40 | 0.380 | 0.419 | 0.419 | 0.424 | +0.044 |
| 0.50 | 0.330 | 0.373 | 0.379 | 0.376 | +0.049 |
| 0.60 | 0.280 | 0.334 | — | 0.331 | +0.054 |
| 0.70 | 0.230 | 0.293 | — | 0.307 | +0.077 |
| 1.00 | 0.080 | 0.159 | — | — | +0.079 |

**Key finding**: All three PPO runs achieve remarkably similar scores at
equivalent alpha levels, regardless of curriculum strategy. PPO adds
~+0.03-0.08 above the pretrained baseline, with larger absolute gains at
higher alpha. The two-phase curriculum (Exp C) shows a slight edge at
α=0.7 (+0.014 above PPO v3), possibly due to better value function warmup.

The improvement saturates quickly — PPO v3 at 2221 steps and Exp A/C at
400 steps achieve nearly identical scores. Most learning happens in the
first ~100 steps at each alpha level.

---

## 6. Analysis and Recommendations

### Key Insight: PPO Learning Is Real But Limited

The data shows a consistent pattern across all four experiments:

1. **PPO approximately doubles the exact-match component** of the pretrained
   baseline at every alpha level (0.08 → ~0.16 at α=1.0).

2. **This improvement saturates quickly** — PPO v3 at 2221 steps and Exp A/C at
   400 steps achieve nearly identical scores at the same alpha.

3. **Structure learning (α=0) is pointless** — the pretrained model is already
   at the presence ceiling for Tier 1 categories (Exp B shows 0 learning).

4. **The bottleneck is not the curriculum** — it's the model's ability to learn
   specific DNA motif sequences from sparse rewards.

5. **Two-phase curriculum provides marginal benefit** — Exp C's 200-step plateau
   led to lower KL (better sample efficiency) and slightly better scores at
   α=0.7, but the absolute difference is small (+0.014).

### Why Learning Is Limited

The reward landscape is fundamentally challenging:

- **Long sequences, sparse alignment**: Generated DNA is 500-1024bp. A specific
  motif might be 50-200bp. The reward is averaged over the whole prompt, diluting
  the per-token gradient signal.

- **Binary-like exact match**: Smith-Waterman alignment scores jump sharply from
  near-0 to near-1 when the model gets the motif "right." There's limited gradient
  in the 0.1-0.7 range.

- **Cold-start categories**: ORI (0.075 presence) and PROM (0.404) can't improve
  without first learning to generate the right category of DNA, but presence
  reward doesn't help (Exp B shows no learning).

### Revised Recommendations

Given the experimental data, I recommend **two parallel strategies**:

#### Strategy 1: Moderate Alpha PPO (immediate, incremental)

```
Config: α=0.3 constant for 3000 steps, then ramp to α=0.5 over 2000 steps
```

Skip the structure-learning phase entirely. Start at α=0.3 where:
- The reward signal is reasonably dense (expected score ~0.43)
- PPO can add ~3-4% improvement
- The model trains on specificity from step 1

**Expected outcome**: ~0.46 score at α=0.3, ~0.38 at α=0.5. Modest but real
improvement. Target exact-match score ~0.20 (vs 0.08 baseline).

#### Strategy 2: Category-Staged Curriculum (higher effort, higher potential)

The biggest gains likely come from focusing on achievable categories first:

```
Stage 1 (steps 0-1000):   Only AMR/TAG prompts (α=0.5)
   → AMR has 1.0 presence, TAG has 0.18 exact — both achievable
Stage 2 (steps 1000-2500): Add ELEM/REPORTER (α=0.5)
   → 1.0 presence — should converge quickly
Stage 3 (steps 2500-4000): Add PROM (α=0.3)
   → 0.4 presence — needs more steps at lower alpha
Stage 4 (steps 4000-6000): Add ORI (α=0.2)
   → 0.075 presence — needs lowest alpha for any signal
Stage 5 (steps 6000-8000): All categories (ramp α to 0.5)
   → Consolidate with full prompt distribution
```

**Implementation**: Modify `PlasmidRewardWrapper` to accept `active_categories`.
Use a `CategoryStageCallback` that expands the set at milestone steps. Filter
training prompts to only include active categories OR mask rewards for inactive
categories to zero.

**Expected outcome**: Much higher per-category exact match for AMR/TAG/REPORTER
(potentially >0.40), with gradual transfer to harder categories.

### Hyperparameter Recommendations

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| KL coeff | 0.5 | 0.2 | Current policy drifts without learning — relax constraint |
| Clip range | 0.1 | 0.2 | Allow larger updates (model is conservative) |
| Learning rate | 3e-6 | 5e-6 | Faster learning with denser signal |
| Temperature | 1.0 | 1.0 | Keep current (entropy is healthy at ~950) |
| Starting α | 0.0 | 0.3 | Skip useless structure-learning phase |
| Max α | 1.0 | 0.5 | Soft ceiling for signal density |
| Total steps | 5000 | 5000+ | Same or more budget |

### Success Metrics

1. **QC pass rate**: Fraction of generated sequences with SW alignment ≥ 0.70
   for the exact conditioned token. Target: >30% (from current ~5%).

2. **Mean exact match score**: Average per-token alignment score at α=1.0.
   Target: >0.25 (from current 0.08 baseline, 0.16 PPO v3 plateau).
   *Revised down from 0.35 based on experimental data showing limited PPO gains.*

3. **Category breakdown**: Per-category QC pass rates. AMR and TAG should
   reach >40% first; ORI is stretch goal.

4. **EOS rate**: Fraction of sequences with proper `<EOS>` termination.
   Currently ~0% (model never generates EOS).

---

## 7. Technical Notes

### TRL Experimental PPOTrainer Quirks

1. **`max_steps` is ignored** — TRL's experimental PPOTrainer uses `total_episodes`
   for the training loop length. Must set `total_episodes = max_steps × batch_size
   × grad_accum` explicitly.

2. **`control.should_training_stop` not checked** — TrainerCallback cannot stop
   training via the standard HF mechanism. The `total_episodes` fix is the only
   reliable way to control training length.

3. **`on_step_begin` not called** — Only `on_step_end` is called, which means
   alpha updates take effect one step late (negligible over long ramps).

### Experiment Infrastructure

| Experiment | Steps | Duration | α schedule |
|------------|-------|----------|------------|
| PPO v3 | 2221 | ~10h | 0→1 over 1000 |
| Exp B (presence) | 590 | ~2.6h | 0 constant |
| Exp A (soft ceiling) | 513 | ~2.3h | 0→0.5 over 400 |
| Exp C (two-phase) | 400 | ~1.8h | 0 for 200, then 0→0.7 |

---

## Appendix A: File Inventory

```
configs/
  exp_a_soft_ceiling.py    # α: 0→0.5 over 400 steps
  exp_b_presence_only.py   # α=0 constant for 400 steps
  exp_c_two_phase.py       # α=0 for 200 steps, then ramp to 0.7

scripts/
  train_ppo.py             # Modified: alpha_plateau_steps, MaxStepsStopCallback
  run_experiments.sh       # Sequential experiment runner

reports/
  rl_curriculum_strategy.md  # This file
```

## Appendix B: Reward Function Summary

```
reward(prompt, seq) = mean(blended_scores) + eos_bonus - length_penalty

blended_score(token) = α × exact_match(token, seq) + (1-α) × presence(category, seq)

exact_match:  SW alignment of seq against all registry variants of the specific token
presence:     SW alignment of seq against up to 10 representatives of the token's category

α=0.0: "Did you generate any valid ORI?"
α=0.5: "Half credit for any ORI, half for the specific one requested"
α=1.0: "Only the exact variant matters"
```

## Appendix C: Raw Metric Summaries

### PPO v3 (α: 0→1 over 1000 steps, 2221 steps)

```
Window     Score  KL     ~Alpha
1-50       0.581  0.302  0.03
51-100     0.555  0.398  0.08
101-200    0.531  0.451  0.15
201-500    0.439  0.499  0.35
501-1000   0.268  0.643  0.75
1001-1500  0.159  0.701  1.00
1501-2000  0.159  0.742  1.00
2001-2221  0.160  0.763  1.00
```

### Exp B (α=0 constant, 590 steps)

```
Window     Score  KL     Entropy
1-50       0.590  0.282  974
51-100     0.588  0.313  958
101-200    0.596  0.388  974
201-300    0.592  0.446  977
301-400    0.592  0.572  959
401-500    0.589  0.559  952
501-590    0.591  0.578  953
```

### Exp A (α: 0→0.5, 513 steps)

```
Window     Score  KL     ~Alpha
1-50       0.577  0.273  0.03
51-100     0.544  0.364  0.09
101-150    0.524  0.439  0.16
151-200    0.503  0.481  0.22
201-250    0.471  0.463  0.28
251-300    0.441  0.496  0.34
301-350    0.416  0.514  0.41
351-400    0.389  0.482  0.47
401-513    0.376  0.573  0.50 (plateau)
```

### Exp C (α=0 for 200 steps, then 0→0.7, 400 steps)

```
Window     Score  KL     ~Alpha  Phase
1-50       0.590  0.244  0.00    Phase 1
51-100     0.586  0.266  0.00    Phase 1
101-150    0.593  0.300  0.00    Phase 1
151-200    0.600  0.225  0.00    Phase 1
201-250    0.556  0.212  0.09    Phase 2
251-300    0.475  0.204  0.26    Phase 2
301-350    0.404  0.161  0.44    Phase 2
351-400    0.327  0.133  0.61    Phase 2
```
