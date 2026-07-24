# NeurIPS 2026 Submission 25202 — Rebuttal Triage

**Paper:** PlasmidLM: A Promptable DNA Language Model via Verifiable-Reward Post-Training
**Forum:** https://openreview.net/forum?id=F5vPBPLwsK
**Reviews in:** 23 Jul 2026. Three reviewers, all `3: Borderline reject`. Confidences 3, 3, **5**.
**Meta-review verdict:** "heading toward rejection."
**Triage date:** 24 Jul 2026
**Branch:** `claude/neurips-rebuttal-triage-59f713` (worktree)

---

## 0. Executive read

> ## 🚨 STOP — read F0 first.
> **The 1,000-prompt evaluation benchmark was drawn from the pretraining corpus, not from
> the 22,400-pair holdout.** All 1,000 prompts appear verbatim in
> `training_pairs_v4.parquet`. This is verified in the eval code and in all three run
> configs behind the paper's numbers. The paper's Section 3.1 description of the benchmark
> does not match what the code did. **This is a factual error in the paper and it must be
> corrected regardless of what happens with this submission.** See §1/F0.

The meta-review is the decision function, and it is *not* a summary of the three reviews —
it is a sharper argument than any of them. It reduces to one claim:

> The gains establish better performance **against this scorer**, not model-independent
> controllability.

Everything else (splits, retrieval, scope) is supporting fire. The AC also explicitly warns
that "if these reservations are substantially correct, resolving them would likely require
more than rebuttal-stage clarification" — i.e. they have pre-committed to reject unless we
produce *evidence*, not prose.

**The reservations are substantially correct on the points that matter, and the split
concern in particular is materially worse than any reviewer realised.** F0 means the
rebuttal cannot be organised around defending the headline numbers. The realistic options
are (a) re-run the benchmark on genuinely held-out prompts inside the window and report
whatever it says, or (b) withdraw/major-revise. Both are legitimate; (a) is worth
attempting because the infrastructure is fast (~46 min for 1,000 generations per the eval
log) — but only if a genuinely disjoint prompt pool can be constructed (see C1).

There are still **four factual corrections** we can make with confidence, and they are
worth making — but they are now secondary to F0.

---

## 1. Verified findings (checked against code/artifacts today — not guesses)

Each of these was checked directly. Citations are to real files/lines.

### F0 — 🚨 The evaluation benchmark is drawn from the pretraining corpus. **CONFIRMED, HIGH CONFIDENCE.**

**What the paper says** (`main.tex:202-204`):
> "A separate holdout of 22,400 prompt–sequence pairs, **disjoint from pretraining**,
> provides the pool from which the 1,000-prompt evaluation benchmark is sampled."

**What the code did** (`eval/scripts/generate.py:86-95`):
```python
prompt_df = pd.read_parquet(args.prompts, columns=["prompt"])   # default: data/training_pairs_v4.parquet
...
unique_prompts = prompt_df["prompt"].unique()
sampled = np.random.choice(unique_prompts, size=min(args.n, len(unique_prompts)), replace=False, ...)
```
There is **no holdout filtering anywhere in the generation path.** `holdout_full_prompts.parquet`
is used only for periodic eval *during GRPO training*
(`grpo_plannotate_v3_anyscale.py:59`), never for the reported benchmark.

**Run configs confirm it.** `runs/grpo_plannotate_full_20260408/config.json`:
```json
{ "model": "McClain/PlasmidLM-kmer6-GRPO-plannotate",
  "prompts": "data/training_pairs_v4.parquet", "n": 1000, "seed": 42, ... }
```

**Measured directly** against `training_pairs_v4.parquet` (108,468 pairs / 20,644 plasmid
IDs / 5,124 unique prompts — matches the paper's corpus figures exactly):

| Run (all n=1,000) | distinct prompts | present in pretraining corpus | shared with GRPO run |
|---|---|---|---|
| `grpo_plannotate_full_20260408` (single-shot headline) | 1,000 | **1,000 / 1,000** | — |
| `base_kmer6_matched_20260420` (paired base) | 1,000 | **1,000 / 1,000** | 1,000 |
| `bon_grpo_seed43_20260420` (best-of-K) | 1,000 | **1,000 / 1,000** | 1,000 |

All three run families behind every headline number share one identical 1,000-prompt set,
and **100% of it was seen during pretraining.** Because GRPO also rolled out over the whole
of `training_pairs_v4.parquet` (F5), 100% was seen during post-training too.

Worse, the prompts are largely plasmid-specific:

| Training plasmids carrying that exact prompt | eval prompts |
|---|---|
| **exactly 1** | **530** |
| 2 | 173 |
| 3–5 | 162 |
| ≥6 | 135 |

For **530 of 1,000** benchmark prompts, pretraining contained exactly one plasmid with that
token set — i.e. a near-unique (prompt → sequence) mapping the model was directly trained on.

**What this does and does not mean.** It does *not* prove the outputs are memorised: the
model still emits a multi-kb sequence, and prompts are unordered token sets, not answers.
Testing memorisation is a separate measurement (M2.2). What it *does* mean is that the
paper's stated evaluation protocol is not the protocol that was run, so the phrase
"held-out 1,000-prompt benchmark" — which appears in the abstract, Section 4, and the
conclusion — is not currently supported. Meta-review point 2 and Dok5 W5 were right, and
neither reviewer knew how right.

**Implication:** every headline number (48.5% single-shot, 89.7% best-of-4, and the base-vs-GRPO
deltas) is measured on seen prompts and needs re-running on a genuinely disjoint pool before
it can be defended. This dominates the rest of the triage.

### F1 — Appendix A.1 is a literal `% TODO` block. **CONFIRMED.**
`main.tex:640-650` in `~/Projects/PhD/plasmidlm-paper` is a commented-out outline
("TODO: fill in the exact functional form of the recall penalty..."). Meta-review calls
this out by name ("The promised Appendix A.1 is also empty, leaving the reward
insufficiently specified"); SeFq also flags missing appendices A.1, B, D.

**This is free to fix.** The reward is fully specified in code:
`post_training/scorers/plannotate.py:456-520` plus the length penalty in
`post_training/configs/grpo_plannotate_v3_anyscale.py:44-47`.

### F2 — The GRPO reward is **recall-only**. There is no precision term. **CONFIRMED.**
From `post_training/scorers/plannotate.py:456-520`:

```
per-token quality  q_t   = w_id·min(pct_id/100,1) + w_cov·min(cov/100,1) + w_norm·min(bitscore/slen/1.5, 1)
aggregate quality  Q     = geo_mean({q_t : t found})^sharpness
recall             r     = n_found / n_expected
recall penalty     R     = recall_floor + (1 - recall_floor)·r      # recall_floor = 0.5
composite          reward= Q · R          (then length penalty: 0.02/kb above 6kb, cap 0.15)
```

Nothing in this expression looks at components the sequence contains but the prompt did
**not** request. pSXh W2 and meta-review point 1 ("Faithfulness rewards recall without
penalizing unrequested components") are **factually correct, not a misreading.** We cannot
argue this one; we can only measure it and report it.

### F3 — The 95/5 train/val split IS pair-level random, NOT plasmid-ID stratified. **CONFIRMED.**
`plasmid_llm/data.py:88-98` is a bare `torch.randperm` over pair indices. With 108,468 pairs
over 20,644 plasmid IDs (≈5.3 pairs/plasmid), the validation set almost certainly contains
sequences that also appear in training under a different prompt subset. Dok5 W5 is right.

There are *two different splits*, and Dok5 (confidence 5) blurred them. I had this filed as
our one factual correction against them — **F0 removes that.** For the record:
- The 95/5 train/val split is pair-level random. It taints the **val loss 0.129 / token acc
  97.4%** in Table `tab:hyperparams` — a diagnostic entry, not a headline claim.
- The headline rates were supposed to come from the separate 22,400-pair holdout. Per F0,
  **they did not.** They came from the pretraining corpus.

So Dok5's conflation turned out to be harmless: they attacked the wrong split and were
still right about the thing that matters. Do not attempt the "you conflated two splits"
correction — it would read as deflection and a reviewer who then checks the eval code finds
F0.

⚠️ Secondary problem, ours not theirs: per project memory, the released
`McClain/PlasmidLM-kmer6` base checkpoint's training logs were **not retained**, and the
0.129/97.4% figures are unverifiable against any on-disk artifact (the local
`checkpoint-15000` has a different SHA256 and reports 0.0926/96.1%). The paper already
half-admits this at `main.tex:942`. If a reviewer pulls this thread we have no defence.
Recommend we **drop the eval-loss row** rather than defend it.

### F4 — The 16,384 vs 4,096 discrepancy: both reviewers are right to ask, and the real answer is more restrictive than either guessed. **CONFIRMED.**
The tokenizer is **k=6 with stride 3** (`config.json`: `kmer_k: 6, kmer_stride: 3`;
`tokenization_kmer.py:129`). Each token advances **3 bp**, not 6. Therefore:

| Quantity | Tokens | DNA |
|---|---|---|
| `max_position_embeddings` (architectural RoPE capacity) | 16,384 | ~49 kb |
| `max_seq_len` — **actual pretraining window** (`plasmid_llm/config.py:30`, all `pretraining/configs/*.py`) | 4,096 | **~12.3 kb** |
| `max_new_tokens` — **actual GRPO rollout + eval generation cap** (`grpo_plannotate_v3_anyscale.py:48`) | 2,500 | **~7.5 kb** |

The codebase already states this: `pretraining/configs/p4_kmer6_s3.py:4` — "At
max_seq_len=4096, covers ~12K bp of DNA context."

This single table answers **pSXh W5/Q4, SeFq Q2, and SeFq Q3 at once**, and explains why
Figure 3 tops out near 8 kb — that is the 7.5 kb generation cap, not a plotting choice.
It is a clean clarification win. It also honestly concedes a real scope limit: the model
**cannot** emit the >12 kb Addgene tail, which we should state rather than let a reviewer
discover.

### F5 — GRPO rollout prompts are drawn from the **training** pairs. **CONFIRMED, needs quantifying.**
`grpo_plannotate_v3_anyscale.py:53` — `prompts_parquet=training_pairs_v4.parquet`.
Eval prompts come from `holdout_full_prompts.parquet`. With only **5,124 unique prompts**
across the whole corpus (`main.tex:199`), the eval prompts are very likely a subset of the
GRPO rollout prompt pool.

This is **not** sequence leakage — the reward is registry-BLAST and never sees the paired
reference sequence, so the model is not shown "the answer." But it is prompt-level overlap
between RL training and evaluation, no reviewer has spotted it, and it will look far worse
if a reviewer finds it than if we disclose it with a number. **Disclose proactively.**

### F6 — The artifacts needed for the highest-value new analyses already exist. **CONFIRMED.**
HF bucket `McClain/PlasmidLMEval` (public, 321 files, 256 MB) contains, per run
(base matched, GRPO, 4× best-of-N seeds, temperature sweep):
- `plannotate_results.json` — **full annotations, including components we did not request**
  → the precision/specificity metric (pSXh W2/Q2, meta M1) is computable **from existing
  data with zero regeneration.**
- `orfs/prodigal_proteins.faa` — **translated ORFs already extracted**
  → an orthogonal protein-level annotator (Pfam/HMMER, AMRFinderPlus — Dok5 Q3, meta M1)
  can run on existing artifacts. No GPU, no regeneration.
- `generations.parquet`, `metrics/prompt_fidelity_per_seq.parquet`, `reference/training_sigs.zip`.

This is the good news in this triage: **the two analyses the AC most wants are the two that
need no new generation.**

### F7 — Real PlasmidGPT (Shao 2024) is public and runnable. **CONFIRMED.**
`lingxusb/PlasmidGPT` on HF is Shao's model. Also `UCL-CSSB/PlasmidGPT` (our lab's) and
`UCL-CSSB/PlasmidGPT-SFT` / `-GRPO`.

🚨 **Name-collision trap:** `McClain/plasmid-gpt-244m` and `-319m` are *our own* GPT-2
conditional models that happen to be called "PlasmidGPT" in their model cards
(244M, 13,260 pairs, different conditioning vocabulary: `<HOST:ECOLI>` etc.).
**They are not Shao 2024.** Do not cite or present them as the baseline — a reviewer who
checks would treat that as misrepresentation. They are legitimate as an *additional internal
baseline* if we want one.

### F8 — Tooling gap on this machine.
`blastn` present. `mmseqs`, `cd-hit`, `hmmscan/hmmsearch`, `amrfinder` **absent**; no conda.
Clustering and orthogonal-annotator work needs g6-big / p4 (per CLAUDE.md, `plannotate`
conda env has blast/prodigal/sourmash) or a fresh install. **Check this before promising
any clustering number in the rebuttal.**

---

## 2. Open questions — must resolve before drafting the response

| # | Question | Why it's load-bearing | Status |
|---|---|---|---|
| ~~C1~~ | ~~Is the 22,400-pair holdout disjoint at plasmid level?~~ | Superseded by **F0** — the holdout was never used for the benchmark at all. | **RESOLVED (badly)** |
| **C1′** | **Can a genuinely disjoint prompt pool be built at all?** Requires locating `holdout_full_prompts.parquet` (Anyscale bucket, per `infra/anyscale/run_anyscale.py:102`) **and** verifying its plasmid IDs are disjoint from the 20,644 in `training_pairs_v4.parquet`. Fallback: construct one ourselves by holding out plasmid IDs and re-deriving prompts. | Decides whether the honest re-run is even available, and therefore whether we rebut or withdraw. **Now the single most load-bearing unknown.** | Not started. Needs Anyscale creds, or copies on g6-big/p4 at `/opt/dlami/nvme/data/` |
| ~~C2~~ | ~~Eval-prompt ∩ GRPO-rollout-prompt overlap?~~ | **Answered: 100%.** All 1,000 eval prompts are in `training_pairs_v4.parquet`, over which GRPO rolled out. | **RESOLVED** |
| **C3** | Are `hmmer` + Pfam, or AMRFinderPlus, installable/available on g6-big or p4 in-window? | Gates whether the orthogonal-annotator answer (Dok5 Q3, meta M1) is deliverable. | Not checked |
| **C4** | Which paper source is the NeurIPS submission? | `~/Projects/PhD/plasmidlm-paper` is on branch **`icml-workshop`** with 65 uncommitted insertions to `main.tex` and untracked `icml2026.sty`. The tree is being repurposed for an ICML workshop submission. Editing `main.tex` here risks clobbering that work. | **Needs your call** — see §5 |
| **C5** | **Given F0 — rebut or withdraw?** | F0 is not a presentation problem. If the re-run is not feasible in-window, the rebuttal cannot honestly defend the headline numbers, and a response that stays quiet about F0 would be a misrepresentation. | **Needs your call** — see §5 |

---

## 3. Triage by meta-review point

Severity: 🔴 decides the outcome · 🟠 materially moves a reviewer · 🟡 cheap credibility.
Feasibility: **NOW** (existing artifacts) · **WINDOW** (doable in rebuttal period) · **NO** (concede).

### M1 — Reward, metric, and best-of-K ranker are all the same registry + pLannotate. 🔴

The AC's primary objection. We cannot argue our way out (F2 confirms the recall-only
critique). Plan is to *measure* it.

| # | Action | Sev | Feas | Notes |
|---|---|---|---|---|
| M1.1 | **Precision / specificity metric.** For each generation, count detected components **not** requested by the prompt; report extra ORIs / AMRs / promoters / reporters per plasmid, base vs GRPO, single-shot and best-of-4. | 🔴 | **NOW** | Directly answers pSXh Q2 + meta M1. Recomputable from `plannotate_results.json` in the bucket. **Highest value-per-hour item in the whole rebuttal.** ⚠️ Run this *before* promising a number — if GRPO makes specificity *worse*, that is important for us to know first. |
| M1.2 | **Orthogonal annotator re-score.** Pfam/HMMER on the existing `prodigal_proteins.faa`; AMRFinderPlus for AMR calls. Report base-vs-GRPO delta under the independent annotator. | 🔴 | **WINDOW** (blocked on C3) | Dok5 Q3 verbatim; the AC's "independent annotator" ask. Proteins already extracted (F6) so no regeneration. If the GRPO delta survives, this substantially rescues M1. |
| M1.3 | **Held-out registry split.** Hold out a subset of the 660 registry entries from the reward, evaluate faithfulness only on held-out entries. | 🟠 | **NO** for the rebuttal | Requires GRPO retraining (~17 h + Ray setup) *and* re-eval. Propose as camera-ready/future work; do not promise. |
| M1.4 | **Fill Appendix A.1** from `plannotate.py:456-520`. | 🟡 | **NOW** | Named explicitly by the AC. Trivial. Do it first. |
| M1.5 | **Recalibrate biological claims.** Systematic pass replacing "viable" → "annotatable under our rule"; state plainly that viability = origin + marker + (promoter or CDS) **detected**, and does not check ORF integrity, reading-frame coherence, or origin completeness. | 🟠 | **NOW** | Dok5 W3 is correct and we partially concede it already. Owning it fully is cheaper than defending. |

### M2 — Independence from the training corpus / retrieval effects. 🔴

| # | Action | Sev | Feas | Notes |
|---|---|---|---|---|
| **M2.0** | 🚨 **Re-run the benchmark on a genuinely plasmid-ID-disjoint prompt pool.** Regenerate n=1,000 for base and GRPO, re-annotate, recompute useful-plasmid rate and best-of-K. Report the new numbers whatever they are. | 🔴🔴 | **WINDOW**, blocked on C1′ | Consequence of F0. Generation is ~46 min/1,000 per the eval log; pLannotate ~35 min at 8 workers. The compute is the easy part — **C1′ is the blocker.** Nothing else in M2 matters until this is settled. |
| M2.1 | **Correct the benchmark description in §3.1** and every "held-out" claim (abstract, §4, conclusion). Also state that the 95/5 train/val split is pair-level (F3). | 🔴 | **NOW** | Required regardless of rebuttal outcome. This is a correction, not a rebuttal tactic. |
| M2.2 | **BLASTN every generation against the 20,644 training plasmids**; report max identity × coverage distribution, and the fraction of each generation covered by its single best training match. | 🔴 | **WINDOW** | This is what the AC actually means by "quantify train–test similarity" and what beats the sourmash objection ("does not resolve near-duplicate or sub-plasmid reuse"). `blastn` is available locally; `full_sequences.parquet` (420 MB) is on S3. **Strongest single answer to M2.** |
| M2.3 | **Nearest-neighbour retrieval baseline.** For each eval prompt, return the training plasmid with max prompt-token-set Jaccard; score it through the identical viability/faithfulness pipeline. | 🔴 | **WINDOW** | Dok5 W2 + pSXh W4 + meta M3 in one number. No model inference needed — pure retrieval + existing annotation pipeline. If PlasmidLM beats retrieval, that is the paper's missing keystone. **If it doesn't, we need to know now, not in the camera-ready.** |
| M2.4 | **Backbone/cluster analysis.** MMseqs2 or CD-HIT at 90% nt identity over the 20,644; report cluster-size distribution and BB_* class distribution. | 🟠 | **WINDOW**, blocked on C3/F8 | Dok5 Q1/Q2. Needs a tool install. |
| M2.5 | Disclose eval-prompt ∩ GRPO-rollout-prompt overlap. | 🟠 | **NOW** | **Answered: 100%** (F0/F5). Number in hand; needs stating, not computing. |

### M3 — Demonstrated scope is narrower than "specification following." 🟠

Mostly framing plus one baseline. The AC's language ("identify reliably supported prompt
classes") is an invitation to narrow the claim rather than defend it.

| # | Action | Sev | Feas | Notes |
|---|---|---|---|---|
| M3.1 | **Retitle / reframe hard-token annotation matching vs functional construct design** throughout. Say plainly: we demonstrate *hard-token annotation matching*; full specification following is not established. | 🟠 | **NOW** | Costs nothing, addresses M3 + pSXh W3 + a chunk of the AC's closing paragraph. |
| M3.2 | **"Reliably supported prompt classes" table.** We already have the data: define an operational support threshold (e.g. ≥100 pretraining occurrences) and publish the in-scope vocabulary explicitly. | 🟠 | **NOW** | Per-token data exists (`per_token_hit_rates.pdf`, treemap). Turns a weakness into a deliverable. |
| M3.3 | **PlasmidGPT (Shao 2024) comparison.** Download `lingxusb/PlasmidGPT`, generate n=1,000, push through the identical viability + distributional pipeline. Faithfulness is not directly comparable (different conditioning vocabulary) — report viability + distributional honestly and contrast conditioning interfaces in prose. | 🟠 | **WINDOW** | SeFq's *single biggest* objection and pSXh Q3. F7 says it's public and runnable. Scope it to viability/distributional; do **not** overclaim a faithfulness head-to-head. |
| M3.4 | Cunningham et al. 2025 — contrast in related work. | 🟡 | **NOW** | SeFq [2]. Prose only; wet-lab-validated work, we cannot match it empirically. |
| M3.5 | Answer SeFq Q4 (why not fine-tune Evo2 with an extended vocabulary). | 🟡 | **NOW** | We have `eval/scripts/evo2_score.py` and `McClain/Evo2_1b_base` — we already use Evo2 as a scorer, so we can answer from experience rather than speculation. Frame as complementary, note the 19.3M-vs-1B accessibility argument the reviewers already credited. |

### Not addressable — concede explicitly

| # | Item | Notes |
|---|---|---|
| X1 | **Wet-lab validation** (Dok5 W4, Q4). | Cannot be done in a rebuttal window under any circumstances. Dok5's "shouldn't be expensive" underrates turnaround, not cost. Concede directly and state it as the honest boundary of the claim. |
| X2 | **Dok5's framing that from-scratch plasmid design is not a real need.** | They explicitly said they did *not* score us on this. Do not spend rebuttal space relitigating it; a brief, non-defensive acknowledgement at most. |
| X3 | **Retraining with a plasmid-ID-stratified split.** | Pretraining + GRPO re-run. Not in-window. Concede + commit for camera-ready. |

---

## 4. Suggested execution order

Ordered by (value × feasibility) ÷ time, and by what unblocks decisions.

**Phase 0 — resolve the unknowns (do first, today).**
1. **C1′** — locate `holdout_full_prompts.parquet` and verify plasmid-ID disjointness, or
   build a disjoint pool ourselves. *This gates M2.0, and M2.0 gates whether a rebuttal is
   possible at all.* Everything else is secondary.
2. **C5** — decide rebut vs withdraw once C1′ is known.
3. **C3** — check hmmer/Pfam/AMRFinderPlus availability on g6-big/p4.
4. **C4** — decide the paper-source question (§5).

**Phase 1 — the honest re-run (blocks everything downstream).**
5. **M2.0** regenerate + re-annotate + rescore on the disjoint pool. Until this lands, every
   number in §5 of the paper is provisional and no rebuttal language should be drafted
   around it.

**Phase 2 — free wins, no new compute** (safe to do in parallel with Phase 1; none depend on M2.0).
6. M1.4 fill Appendix A.1 · 7. F4 seq-length clarification table · 8. M2.1 benchmark-description
correction · 9. M2.5 prompt-overlap disclosure · 10. M3.1 reframing pass ·
11. M1.5 viability recalibration · 12. M3.2 supported-vocabulary table · 13. M3.4/M3.5 prose answers.

**Phase 3 — new analyses on existing artifacts.**
14. **M1.1 precision/specificity** (highest value-per-hour) · 15. M2.2 BLASTN train-similarity ·
16. M2.3 retrieval baseline.
⚠️ M1.1 and M2.2 should be **re-run on the M2.0 outputs** once those exist; running them on the
current generations characterises a benchmark we are about to retire.

**Phase 4 — needs setup.**
17. M1.2 orthogonal annotator · 18. M3.3 PlasmidGPT comparison · 19. M2.4 clustering.

**Phase 5 — assemble.** Draft per-reviewer responses; global revision; PDF diff.

> **Run before you promise.** M2.0, M1.1, M2.2, and M2.3 can each come back *against* us —
> M2.0 most of all. Get the numbers before committing to any language in the response.

---

## 5. Needs your decision

1. 🚨 **Rebut or withdraw (C5).** F0 means the headline numbers as published are measured on
   prompts the model saw in pretraining. My read: attempt the M2.0 re-run first, then decide
   with the new numbers in hand — the compute is roughly a day, and if the rates hold up on a
   disjoint pool the rebuttal becomes genuinely strong (it would answer meta M2 head-on).
   If they collapse, or if C1′ shows no disjoint pool can be built in time, the honest move is
   to withdraw rather than file a response that leaves F0 unstated. **Your call — but a
   rebuttal that does not disclose F0 is not one I can help draft.**

2. **Which paper source is the NeurIPS submission (C4)?** `~/Projects/PhD/plasmidlm-paper` is
   on branch `icml-workshop` with 65 uncommitted `main.tex` insertions and untracked
   `icml2026.sty` — the tree looks repurposed for an ICML workshop version. Editing `main.tex`
   in place would tangle the two. Options: branch off `main` for the rebuttal, or commit/stash
   the ICML work first. **I have not touched `main.tex`.**
   ⚠️ Separately: F0 affects the ICML workshop version too if it reuses these numbers.

3. **What is the actual rebuttal deadline?** Reviews posted 23 Jul; today is 24 Jul. Drives
   whether M2.0 is feasible and therefore drives decision 1.

4. **Anyscale S3 credentials (C1′)** — still valid for `holdout_full_prompts.parquet`? If not,
   fallbacks are g6-big/p4 `/opt/dlami/nvme/data/`, or constructing a disjoint pool ourselves
   from `training_pairs_v4.parquet` (it has `plasmid_id`, so a clean ID-level holdout is
   straightforward to build).

5. **Notion research-log entry?** Standing convention is to log meaningful findings. F0 clearly
   qualifies. Say the word and I'll search the Research DB for an existing PlasmidLM entry to
   append to rather than duplicating. **Not logging anything without your go-ahead.**

---

## 6. Reviewer-by-reviewer crosswalk

| Reviewer | Conf | Concern | Maps to | Our read |
|---|---|---|---|---|
| **Dok5** | **5** | W1 backbone clustering absent | M2.4 | Correct |
| Dok5 | 5 | W2 no retrieval null | M2.3 | Correct — **highest-risk open question** |
| Dok5 | 5 | W3 "viable" = annotatable, not functional | M1.5 | Correct; concede |
| Dok5 | 5 | W4 no wet lab | X1 | Correct; cannot fix |
| Dok5 | 5 | W5 split leakage | **F0**, F3, M2.0 | **Correct, and understated.** They conflated the two splits but were right about the one that matters. Do not attempt the "you conflated two splits" correction |
| Dok5 | 5 | Q3 orthogonal annotator | M1.2 | Runnable (F6) |
| **pSXh** | 3 | W1 reward/metric/ranker coupled | M1.1, M1.2 | Correct |
| pSXh | 3 | W2 recall-only faithfulness | F2, M1.1 | **Correct — verified in code** |
| pSXh | 3 | W3 soft tokens uneven | M3.1, M3.2 | Correct; already partly conceded |
| pSXh | 3 | W4 no retrieval/template baseline | M2.3 | Correct |
| pSXh | 3 | W5 16,384 vs 4,096 | **F4** | Answerable cleanly — **clarification win** |
| pSXh | 3 | W6 circularity / linearization | — | Minor; prose |
| **SeFq** | 3 | No head-to-head vs PlasmidGPT | M3.3 | Addressable (F7) |
| SeFq | 3 | Missing appendices A.1, B, D | **F1** | **Correct — verified.** Free fix |
| SeFq | 3 | Q2/Q3 context length | **F4** | Same clean answer |
| SeFq | 3 | Q4 why not fine-tune Evo2 | M3.5 | Answerable from experience |
| SeFq | 3 | "Failure analysis is the more interesting headline" | M3.1 | **Take the advice** — it aligns with the AC's "identify reliably supported prompt classes" |

> SeFq's closing suggestion and the AC's closing instruction point the same direction:
> reframe around the failure analysis and the honest scope of control, rather than defending
> the headline rate. That is probably the highest-leverage *free* move available.

---

## 7. Artifact inventory (verified present)

**HF bucket `McClain/PlasmidLMEval`** (public, 321 files, 256 MB):
- `runs/grpo_plannotate_full_20260408/` — GRPO n=1,000: `generations.parquet`,
  `plannotate_results.json`, `orfs/prodigal_proteins.faa`, `metrics/*`
- `runs/base_kmer6_matched_20260420_1303/` — matched base n=1,000, same artifacts
- `runs/bon_{base,grpo}_seed{43,44,45}_*/` — best-of-N seeds
- `runs/base_kmer6_sweep_*/t{03,05,10}/` — temperature sweep
- `runs/forced_sp_{base,grpo}_20260427/` — soft-token forcing (CAI analysis)
- `reference/` — `addgene_500.fasta`, `addgene_reference_metrics.csv`, `training_sigs.zip`

**S3 `s3://phd-research-storage-1758274488/databricks_export/`:**
`training_pairs_v4.parquet` (723 MB) · `full_sequences.parquet` (420 MB) ·
`metadata.parquet` (26 MB) · `annotations.parquet` (244 MB) ·
`motif_registry_combined.parquet` (141 KB)

**Models:** `McClain/PlasmidLM-kmer6` (base) · `McClain/PlasmidLM-kmer6-GRPO-plannotate` ·
`lingxusb/PlasmidGPT` (Shao 2024) · `UCL-CSSB/PlasmidGPT{,-SFT,-GRPO}` ·
`McClain/Evo2_1b_base`

**Not yet located:** `holdout_full_prompts.parquet` — referenced at
`infra/anyscale/run_anyscale.py:102` under the Anyscale bucket. **Blocks C1′, which blocks
M2.0, which blocks the rebuttal.** Highest-priority retrieval.

Note `training_pairs_v4.parquet` carries a `plasmid_id` column and a `splits` column — but
`splits` is a **domain tag** (`["combined","mammalian"]`, `["combined","crispr"]`, …), *not*
a train/val/test assignment. Every one of the 20,644 plasmid IDs sits in exactly one domain
split. There is no ID-level train/test partition recorded in the corpus; one would have to
be constructed.

---

## 8. Reproducing the F0 check

```bash
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet .
curl -sLO "https://huggingface.co/buckets/McClain/PlasmidLMEval/resolve/runs/grpo_plannotate_full_20260408/generations.parquet"
```
```python
import duckdb
T = "'training_pairs_v4.parquet'"; G = "'generations.parquet'"
duckdb.sql(f"select count(*) from (select distinct prompt from {G}) g "
           f"where g.prompt in (select prompt from {T})").show()   # -> 1000
```
Run configs: `https://huggingface.co/buckets/McClain/PlasmidLMEval/resolve/runs/<run>/config.json`
Generation path: `eval/scripts/generate.py:86-95`.
