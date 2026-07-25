# NeurIPS 2026 Submission 25202 — Rebuttal Triage

**Paper:** PlasmidLM: A Promptable DNA Language Model via Verifiable-Reward Post-Training
**Forum:** https://openreview.net/forum?id=F5vPBPLwsK
**Reviews in:** 23 Jul 2026. Three reviewers, all `3: Borderline reject`. Confidences 3, 3, **5**.
**Meta-review verdict:** "heading toward rejection."
**Triage date:** 24 Jul 2026
**Branch:** `claude/neurips-rebuttal-triage-59f713` (worktree)

---

## 0. Executive read

> ## Read F0 and F9 together — one bad, one good.
> **F0:** the 1,000-prompt evaluation benchmark was drawn from the pretraining corpus, not
> from the 22,400-pair holdout the paper describes. Verified in `eval/scripts/generate.py`
> and in all three run configs behind the paper's numbers. **This is a factual error in the
> Methods and must be corrected regardless of what happens with this submission.**
>
> **F0b:** and the designated 22,400-pair holdout **would not have helped** — it is *also*
> not plasmid-disjoint. All 22,392 of its distinct sequences appear in
> `training_pairs_v4.parquet`; 20,562 of the 20,588 plasmids behind it were in pretraining.
> It is a pair-level reshuffle of the same plasmids. **No plasmid-level holdout ever existed
> in the pipeline.**
>
> **F9:** but the model is *not* regurgitating. Pooling *all* 20,644 training plasmids and
> allowing arbitrary fragment stitching, the median generation has only **43%** of its length
> explained by the training corpus — versus **81%** for a genuinely held-out real Addgene
> plasmid. The gap holds inside every length bin. F0 is a Methods-description and re-run
> problem, **not** evidence that the headline numbers are memorisation-inflated, and F9
> answers the memorisation half of meta-review point 2 outright.
> ⚠️ The *retrieval* half (M2.3) is still untested and is the bigger risk — run it early.

### Findings at a glance

Measured today, against code and artifacts. ✅ helps us · ⚠️ hurts us · 🚨 must disclose.

| # | Finding | Verdict |
|---|---|---|
| 🚨 **F0** | The 1,000-prompt benchmark was sampled from the **pretraining corpus**, not a holdout. `generate.py` does no filtering; all three run configs confirm. | Methods error. Must correct. |
| 🚨 **F0b** | The designated 22,400-pair holdout is **also not plasmid-disjoint** — 22,392/22,392 sequences and 20,562/20,588 plasmids are in pretraining. | No plasmid-level holdout ever existed. Answers Dok5 Q1 with the worst case. |
| ✅ **F9** | Model is **not regurgitating**: median generation is 43% explained by the *entire* training corpus vs **81%** for a real held-out Addgene deposit. Holds in every length bin. | Rejects the mosaic hypothesis. Best asset. |
| 🚨 **F13** | **Retrieval saturates the metric**: nearest-neighbour lookup scores ~**95%** useful rate through the same detector. 760/1000 prompts have a training plasmid with an identical component set. | The useful-plasmid rate cannot distinguish design from lookup. |
| ⚠️ **F11** | GRPO **worsens specificity**: unrequested components 1.96 → 2.92/seq; extra origins 18.3% → 29.8%, extra AMR 20.1% → 30.6%. | Predicted by pSXh; follows from the recall-only reward (F2). |
| ⚠️ **F12** | Under **AMRFinderPlus** (independent), absolute AMR recovery halves (26.1% → 13.5%) and the GRPO delta is **not significant** (+1.0 pp, 95% CI [−1.17, +3.27]). | Directionally consistent but underpowered. Not "shown to be zero". |
| ✅ **F4** | 16,384 vs 4,096 resolves: k-mer **stride 3** → 16,384 = architectural (~49 kb); training window 4,096 tok ≈ **12.3 kb**; generation cap 2,500 tok ≈ **7.5 kb**. | Clean answer to pSXh Q4, SeFq Q2/Q3. |
| ✅ **F1** | Appendix A.1 is a literal `% TODO`; reward fully recoverable from `plannotate.py:456-520`. | Free fix (camera-ready — response is text-only). |
| ⚠️ **F2** | Reward is **recall-only**: `geo_mean(found quality) × (0.5 + 0.5·recall)`. No precision term. | pSXh W2 correct, not arguable. |
| ⚠️ **F3** | 95/5 train/val split is pair-level random. | Taints only the val-loss table entry. |
| ✅ **F10** | A disjoint benchmark **is** constructible: 120,449 unseen plasmids; built and running. | Enables M2.0. |
| ✅ **F14** | **GRPO effect replicates exactly on the disjoint benchmark: +9.8 pp** (30.0% → 39.8%), p=2e-06. Paper reported +9.8 pp. | Contamination did **not** manufacture the gain. |
| ✅ **F15** | Best-of-4 replicates too: 67.0% → 80.5% (paper 78.9% → 89.7%). | Both headline numbers survive an honest benchmark. |
| ⚠️ **F16** | **Model does not compose.** On novel component combinations, full-spec satisfaction is **0%** (as is retrieval); substituted-in component produced 4.6% of the time, reverts to the removed one 40.2%. GRPO gain still replicates (+9.8pp @ T=0.5). | Cannot claim compositional/de-novo design. Narrows the contribution. |

**The spine of the response**, if the pending number supports it:

> Concede F0/F0b/F11 without hedging. Then lead with **F14**: rebuilt the benchmark from
> plasmids the model has never seen, and the GRPO effect is *identical* (+9.8 pp, p=2e-06) —
> the contamination did not manufacture the result. Support with **F9** (outputs are 43%
> training-explained vs 81% for a real novel deposit, so not retrieval in disguise). Then
> make **F13** the contribution rather than the wound: *the useful-plasmid rate is saturated
> by a trivial retrieval baseline (~95%), so it needs a novelty axis* — retrieval wins on
> component satisfaction and returns nothing new.

**Verdict: reframe, do not withdraw.** F14 is the reason. The central GRPO claim survives an
honest benchmark; what fails is the framing of the headline metric as evidence of useful
*design*.

---

The meta-review is the decision function, and it is *not* a summary of the three reviews —
it is a sharper argument than any of them. It reduces to one claim:

> The gains establish better performance **against this scorer**, not model-independent
> controllability.

Everything else (splits, retrieval, scope) is supporting fire. The AC also explicitly warns
that "if these reservations are substantially correct, resolving them would likely require
more than rebuttal-stage clarification" — i.e. they have pre-committed to reject unless we
produce *evidence*, not prose.

**The reservations are substantially correct on the points that matter** — the reward really
is recall-only, the appendix really is empty, and the benchmark really was drawn from the
training corpus. But the *inference* the AC and Dok5 draw from the split problem — that the
headline rates are retrieval/memorisation artefacts — **is measurably false** (F9). That
asymmetry is the rebuttal's spine:

> Concede the process errors completely and without hedging. Contest the memorisation
> conclusion with the BLAST control, which is stronger evidence than anything currently in
> the paper.

Plan: (1) fix the Methods description and re-run the benchmark on a genuinely disjoint pool
(fast — ~46 min for 1,000 generations per the eval log, once a pool exists); (2) lead the
rebuttal with F9 plus the precision metric (M1.1); (3) bank the four free factual
corrections. Withdrawal is **not** indicated on current evidence.

Residual risk: the M2.0 re-run could still move the rates. F9 makes a collapse unlikely —
the model is not leaning on seen sequences — but the numbers should be in hand before any
response language is finalised.

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

⚠️ **The overlap table above is corroborating, not diagnostic — do not lead with it.** There
are only 5,124 unique prompts in the entire corpus, so *any* 1,000-prompt sample, including
one drawn from a legitimately plasmid-disjoint holdout, would also show ~100% verbatim
overlap. A sharp AC will make exactly that point. **The dispositive evidence is the run
config and `generate.py`, not the overlap count.**

Context on prompt specificity (same caveat applies):

| Training plasmids carrying that exact prompt | eval prompts |
|---|---|
| exactly 1 | 530 |
| 2 | 173 |
| 3–5 | 162 |
| ≥6 | 135 |

**What this does and does not mean.** It does *not* show the outputs are memorised — the
model still emits a multi-kb sequence, and prompts are unordered token sets, not answers.
That is a separate measurement, and **F9 ran it: the answer is no.** What F0 *does* mean is
that the paper's stated evaluation protocol is not the protocol that was run, so
"held-out 1,000-prompt benchmark" — in the abstract, Section 4, and the conclusion — is not
currently supported.

**Implication:** the Methods description must be corrected and the benchmark re-run on a
disjoint pool (M2.0) for the claim to be literally true. Per F9, expect the rates to move
little.

### F0b — 🚨 The 22,400-pair holdout is not plasmid-disjoint either. **CONFIRMED. C1′ RESOLVED.**

Retrieved `holdout_full_prompts.parquet` from the Anyscale bucket
(`s3://anyscale-production-data-vm-us-east-1-f7164253/org_.../plasmid_llm/data/`; our AWS
creds do reach it). 22,400 rows — matches the paper. Schema is
`prompt, sequence, has_hard_tokens, reward_motifs, n_tokens`; **no `plasmid_id`**, so
disjointness was tested on sequence identity and via a join back to `full_sequences.parquet`.

| check | result |
|---|---|
| distinct sequences in holdout | 22,392 |
| …that also appear in `training_pairs_v4.parquet` | **22,392 (100%)** |
| distinct plasmid IDs behind the holdout | 20,588 |
| …that were also in pretraining | **20,562 (99.9%)** |

The "holdout" is the **same plasmid set with different annotation-subset prompts** — a
pair-level reshuffle. So the paper's "disjoint from pretraining" (`main.tex:202`) is wrong
about the holdout file itself, independently of F0.

**This answers Dok5's Q1 exactly** ("Is the 22,400-pair held-out pool disjoint at the
plasmid-ID level, Addgene-deposit level, or only the pair index?"). Answer: **only the pair
index.** They guessed the worst case and the worst case is what happened.

**Consequence:** no plasmid-level holdout has ever existed in this project, so M2.0 means
*building* one, not *finding* one. Which is feasible — see F10.

### F10 — ✅ A genuinely disjoint benchmark is constructible. **CONFIRMED.**

`full_sequences.parquet` holds **141,093 distinct plasmids**; pretraining used **20,644**.
That leaves **120,449 plasmids the model has never seen**, 123,967 rows of them in the
2–20 kb range (median 7.0 kb). Source split: 115,120 addgene, 38,546 user.

So M2.0 is a construction job, and the compute exists: **`g6-big2` (g6.4xlarge, L4 23 GB) is
already running** and configured — `plannotate` conda env present, and
`/opt/dlami/nvme/plasmid_venv/bin/python` has torch 2.9.1+cu128 / transformers 4.57.6 with
CUDA live. Reachable as `ssh g6-big` (52.54.249.131). Marginal cost of using it ≈ £0.

⚠️ **Prompt derivation needs a fresh pLannotate run, not `annotations.parquet`.** I tried
deriving prompts directly from the existing annotation dump using the repo's own
`feature_to_category_token()` (`scripts/build_motif_registry.py:136`) with its documented
filters (`percmatch ≥ 95`, exclude fragments). Validated against training plasmids whose
prompts we know:

| filter | n | exact set match | recall | precision |
|---|---|---|---|---|
| percmatch ≥95, excl. fragments | 14,593 | 1.4% | **76.4%** | **99.9%** |

Precision 99.9% means the mapping is *sound* — every derived token is real. But recall 76.4%
means `annotations.parquet` is an incomplete dump (it covers only 14,606 of the 20,644
training plasmids). Deriving prompts from it would systematically **under-request** tokens,
making the new benchmark easier and not comparable to the old one. So the honest path is to
run pLannotate fresh on the sampled held-out plasmids — same pipeline as the paper's
detection-ceiling measurement.

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

### F9 — ✅ The model is **not** regurgitating training plasmids. **MEASURED — best rebuttal asset we have.**

Ran the analysis the AC asked for ("quantify train–test similarity") and the one that beats
the sourmash objection. BLASTN of every generation against all 20,644 training plasmids.

**Primary statistic — mosaic coverage.** For each query, the union of *all* HSPs against
*any* training plasmid: what fraction of this sequence can the training corpus explain, if
you are allowed to stitch fragments from arbitrarily many training plasmids? This is
deliberately the most generous possible test of Dok5 W2, whose hypothesis is explicitly a
multi-plasmid "BLAST-stitched mosaic of training fragments."

**The control is what makes it interpretable.** Plasmids are built from a shared parts
catalogue, so raw coverage means nothing alone. Real Addgene plasmids **not** in training
(418 of the 500-plasmid panel; the other 82 are in training and were excluded) set the bar
for how similar a genuinely novel plasmid looks.

| Panel | n | median mosaic coverage | mean | ≥90% covered |
|---|---|---|---|---|
| **Generated** | 1,000 | **0.429** | 0.418 | 7.3% |
| **Real held-out Addgene** | 418 | **0.809** | 0.774 | 25.4% |

**The median generation has only ~43% of its length explained by the entire training corpus,
versus ~81% for a genuine held-out Addgene deposit.** If generations were stitched from
training fragments, this number would approach 1.0. It is less than half the value for real
novel plasmids. **The mosaic hypothesis is rejected.**

Length-stratified, since generations skew longer (median 9.0 kb vs 7.2 kb real) — the gap
holds inside every bin, so it is not a length artefact:

| length bin | gen n | gen median | real n | real median |
|---|---|---|---|---|
| 2–4 kb | 52 | 0.206 | 41 | 0.866 |
| 4–6 kb | 139 | 0.721 | 87 | 0.818 |
| 6–8 kb | 152 | 0.598 | 123 | 0.804 |
| 8–10 kb | 584 | 0.350 | 88 | 0.821 |

<details><summary>Secondary: best-single-subject stats (⚠️ length-confounded — do not lead with these)</summary>

| Panel | n | median best-match cov | near-dup (cov≥90%, id≥99%) | derivative (cov≥75%, id≥95%) |
|---|---|---|---|---|
| Generated (all) | 997 | 0.523 | 7.0% | 19.5% |
| Generated (viable only) | 596 | 0.637 | 10.9% | 31.2% |
| Real held-out Addgene | 418 | 0.790 | 21.1% | 56.5% |

**Do not put the 7.0% / 21.1% near-duplicate contrast in the response.** near-dup is gated at
≥90% coverage *by a single subject*, and generations are longer (9.0 kb) than typical training
plasmids (~7.5 kb) — so a long generation is *mechanically* capped below 90% single-subject
coverage however derivative it is. That biases the gap in our favour for a reason unrelated to
novelty, and Dok5 ("checked the math carefully", confidence 5) is exactly the reader who would
catch it. The mosaic table above is the length-robust version and says the same thing more
strongly.

Viable-only subset uses `tier3_essentials.parquet`; index alignment (`gen_i` ↔ generations row
*i*) **verified by sequence join, 1000/1000**, not assumed.
</details>

Script: [`eval/scripts/train_similarity_blast.py`](../../eval/scripts/train_similarity_blast.py)
— reproduces every table above end-to-end; `blastn` only, ~2 min on 8 threads. Numbers here are
that script's output. BLAST's `max_target_seqs` tie-breaking makes the third decimal vary
slightly between runs; nothing load-bearing moves.

**Caveats to state honestly in the rebuttal:**
- Nucleotide-identity novelty, **not** functional novelty. Shows the model is not copying;
  does not show the constructs work.
- **F9 answers only the memorisation half of meta-review point 2.** The retrieval half
  (M2.3) is untested and is the one that can actually sink the contribution — a model that
  isn't copying can still be matched by simple prompt-Jaccard retrieval, which would make
  "de novo generation" hollow regardless of F9. **Run M2.3 before drafting.**
- Run on the current (F0-contaminated) benchmark; re-run on M2.0 outputs, though a disjoint
  pool should if anything push generated similarity lower.

### F11 — ⚠️ GRPO **worsens** specificity: it adds unrequested components. **MEASURED — goes against us.**

pSXh W2/Q2 and meta-review point 1 asked for a precision-aware metric. Computed it from the
existing `plannotate_results.json` for the paired base and post-GRPO runs (n=996 prompts with
≥1 hard token, identical procedure both sides). Detected components are mapped to the same
token vocabulary as requested ones via the repo's `feature_to_category_token`.

| metric | base | post-GRPO | Δ |
|---|---|---|---|
| mean recall (= paper's faithfulness) | 22.6% | **26.8%** | **+4.2 pp** ✅ |
| mean precision (never reported) | 50.6% | **47.8%** | **−2.8 pp** ❌ |
| mean unrequested components / sequence | 1.96 | **2.92** | **+0.96** ❌ |
| generations with ≥1 unrequested component | 62.4% | **70.8%** | **+8.4 pp** ❌ |

Per category, share of generations carrying an **unrequested** component:

| | ORI | AMR | PROM | REPORTER | ELEM | TAG |
|---|---|---|---|---|---|---|
| base | 18.3% | 20.1% | 38.2% | 21.3% | 36.7% | 5.7% |
| post-GRPO | **29.8%** | **30.6%** | **49.3%** | **29.8%** | **46.6%** | 6.3% |

**Unrequested origins and resistance markers each rise by ~10 pp** — precisely the components
pSXh singled out as "undesirable even if requested components are present."

Robustness: without the quality filter (`percmatch ≥95`, non-fragment) the same direction
holds — extras rise 3.85 → 4.91 per sequence and 86.1% → 87.3% of generations — though
precision is flat (36.1% → 36.4%) rather than declining. **The defensible claim is therefore:
GRPO raises recall and raises unrequested-component count; precision is flat-to-slightly-worse.**
Do not claim a precision collapse; the strict-filter number alone does not support it.

**This is textbook reward misspecification and follows directly from F2** — the reward is
`geo_mean(quality of found tokens) × recall_penalty` with no precision term, so adding
components is never punished and sometimes helps. It is the same mechanism as the epitope-tag
regression already in the paper (§5.3), and it is the honest answer to meta point 1.

⚠️ **Absolute values are not comparable to the paper's per-category hit rates** (e.g. ORI
44.1%/51.9%). The paper detects via BLAST against the 660-entry registry; this uses
pLannotate's own Feature calls plus the repo token map. Different detectors, different
absolute levels. The **base-vs-GRPO contrast is apples-to-apples** because both sides use one
identical procedure — report the contrast, not the levels.

Script: [`eval/scripts/prompt_precision.py`](../../eval/scripts/prompt_precision.py).

**Framing recommendation.** Don't bury this. Reporting it unprompted — with the mechanism —
converts the AC's "you never measured specificity" into "we measured it, found the predicted
failure, and can explain it from the reward's functional form." That is a much better position
than being shown it in a later review round, and it pairs naturally with the reframing SeFq
recommended (lead with the failure analysis).

### F12 — ⚠️ Under an independent annotator the GRPO gain is **not statistically significant**. **MEASURED.**

pSXh Q1 asks directly: "How much of the GRPO improvement remains when evaluated with an
independent annotation pipeline?" Dok5 Q3 asks for AMRFinderPlus specifically. Ran it.

NCBI **AMRFinderPlus 4.2.5** (DB 2025-12-03.1) on the already-archived
`prodigal_proteins.faa` for the paired base and post-GRPO runs. It shares nothing with the
reward path — not the 660-entry registry, not pLannotate, not the databases, not the method
(curated HMMs + reference gene catalogue over translated ORFs, vs BLAST over nucleotide
motifs).

| AMR recovery, token-pooled | base | post-GRPO | Δ |
|---|---|---|---|
| paper's registry/pLannotate (Table 2) | 26.1% | 29.1% | **+3.0 pp** |
| **AMRFinderPlus (independent)** | **13.5%** | **14.6%** | **+1.0 pp** |

Paired bootstrap over 959 prompts / 1,373 requested AMR tokens, 10k resamples:
**95% CI on the delta = [−1.17, +3.27] pp**, p(Δ≤0) = 0.19. McNemar on "≥1 requested AMR
recovered": discordant 98 base-only vs 114 GRPO-only, exact two-sided **p = 0.30**.

**Two distinct findings, and the second is easy to overstate:**

1. **Absolute recovery roughly halves** under the independent annotator (26.1% → 13.5%).
   AMRFinderPlus requires an intact translatable ORF matching a curated reference;
   registry BLAST credits nucleotide similarity even for truncated or frame-broken copies.
   This *quantifies* the annotatable-vs-functional gap that Dok5 W3 asserts qualitatively —
   about half of "recovered" resistance markers do not survive a functional-grade check.

2. **The GRPO delta has the same sign but is not statistically significant.** ⚠️ **This is
   "not independently confirmed", NOT "shown to be zero."** The CI's upper bound (+3.27 pp)
   comfortably contains the registry-measured +3.0 pp, so the data are entirely consistent
   with the paper's effect — the test simply lacks power. AMR is the *worst* category for
   this: it has the smallest registry-measured gain (+3.0 pp) and few tokens. Do not write
   "GRPO does not generalise"; write that the independent check is directionally consistent
   but underpowered.

**Structural limit worth stating plainly in the response.** An orthogonal annotator can only
independently verify *protein-coding* categories — AMR (1,373 requested tokens) and REPORTER
(350). PROM (2,828) and ELEM (3,991), the two largest, are non-coding and have no
independent protein-level checker. So **~70% of requested hard tokens cannot be verified by
any annotator independent of the registry**, with the tools available. That is a real
limitation of the evaluation design, it is the honest core of meta-review point 1, and
conceding it is stronger than pretending AMRFinderPlus settles the question.

Script: [`eval/scripts/orthogonal_amr.py`](../../eval/scripts/orthogonal_amr.py).

### F13 — 🚨 A retrieval baseline **saturates** the useful-plasmid rate. **MEASURED — the most serious finding in this triage.**

Dok5 W2 and pSXh W4 both demand the null the paper never ran. Ran it on the new
plasmid-disjoint benchmark (F10): index each of the 20,583 training plasmids by the union of
hard tokens across its prompts, retrieve the plasmid maximising Jaccard against the
requested set, and score **that retrieved plasmid** against the prompt.

Scored two ways, and the fair one is the second:

| retrieval scored by… | mean faithfulness | faithful @ T=0.5 | implied useful rate |
|---|---|---|---|
| the plasmid's recorded annotation | 96.2% | 100.0% | 95.0% |
| **the same pLannotate detector the model faces** | **95.7%** | **100.0%** [99.6, 100.0] | **95.0%** |

I expected the second to be much lower — the model is scored by a lossy detector, so scoring
retrieval off ground-truth labels would have been an unfair advantage. It isn't: pLannotate
finds the components reliably in real plasmids, so the fair comparison gives the same answer.
Useful rate assumes the paper's 95.0% real-panel viability, which is generous by design —
a null should be steelmanned.

**Nearest-neighbour lookup achieves a ~95% useful-plasmid rate.** The paper reports 48.5%
single-shot and 89.7% best-of-4. On the paper's own primary metric, retrieving an existing
Addgene plasmid beats the model.

⚠️ **Do not compare 95.0% against the paper's 48.5% directly.** Prompts in this benchmark
were derived through the same pLannotate + `percmatch≥95` filter used to score, so tokens the
detector cannot re-find were never requested (mean 8.64 hard tokens/prompt vs 10.42 in the
paper's). The detection ceiling here is ~100%, against 71.8% in the paper. The valid
comparison is **model vs retrieval on this benchmark**, both measured identically — which is
exactly what M2.0 will produce.

**What this does and does not show.** It does not show the model is worthless. It shows the
**useful-plasmid rate cannot distinguish design from lookup**, because it contains no novelty
term. That is a critique of the metric, and it is correct.

**The defensible position combines F13 with F9**, and it is the strongest framing available:

> Retrieval attains ~95% useful rate but returns an existing plasmid — by construction 100%
> of it is training sequence. PlasmidLM's outputs have a median of 43% explained by the whole
> training corpus, against 81% for a genuinely novel Addgene deposit (F9). Retrieval wins on
> component satisfaction and scores zero on novelty; the model produces novel constructs at a
> faithfulness cost. **Reported alone, the useful-plasmid rate is saturated by a trivial
> baseline — it needs a novelty axis to mean anything.**

That is a real scientific contribution (the metric, honestly characterised) and it aligns
with SeFq's advice to lead with the failure analysis.

⚠️ **Still unmeasured: the model's own number on this benchmark.** Generation is running.
Until it lands, "retrieval beats the model" is an inference from a comparison this finding
itself says is invalid — treat it as pending, not established.

**When it lands, decompose `useful = viable × faithful` rather than reporting one number.**
Retrieval's 95% is ~100% faithful × 95% viable — it inherits real-plasmid viability free.
Generated sequences ran ~59.6% viable in the original eval, and possibly lower here since
these prompts carry no `VEC_*`/soft tokens. If so the model is capped near ~60% useful
*however good faithfulness gets*, which would mean **the gap to retrieval is mostly a
viability gap, not a faithfulness gap** — a sharper and more defensible diagnosis than a raw
delta, and the true content of the saturation story. Score the model through the identical
path used for retrieval (pLannotate Feature + `percmatch≥95` + repo token map + the
origin/marker/(promoter|CDS) rule); do not cross detectors or splice in the paper's numbers.

**Structural caveat on the eventual gap.** This benchmark's ~100% detection ceiling favours
retrieval: real components clear `percmatch≥95` trivially, generated ones may not. So
whatever gap emerges is an **upper bound** on retrieval's true advantage.

Scripts: [`eval/scripts/retrieval_baseline.py`](../../eval/scripts/retrieval_baseline.py)
(fast, annotation-free) plus the same-detector rescoring described above.

### F14 — ✅ **The GRPO effect replicates exactly on genuinely held-out plasmids.** **MEASURED — M2.0 COMPLETE.**

The honest re-run. 1,000 prompts from Addgene plasmids never seen in pretraining (F10),
both checkpoints, the paper's exact sampling config, everything scored through one identical
path (pLannotate Feature → repo token map → `percmatch ≥ 95`, non-fragment; viability =
origin AND selection marker AND (promoter OR CDS)).

| panel | n | viable | mean F | faithful @0.5 | **USEFUL** | Wilson 95% |
|---|---|---|---|---|---|---|
| supervised base | 1000 | 48.0% | 29.2% | 31.1% | **30.0%** | [27.2, 32.9] |
| post-GRPO | 1000 | 61.6% | 37.4% | 40.5% | **39.8%** | [36.8, 42.9] |
| retrieval null (F13) | 1000 | ~95% | 95.7% | 100.0% | **~95.0%** | [99.6, 100.0]† |

† retrieval's Wilson interval is on the faithfulness pass rate; useful applies the 95.0%
real-panel viability constant.

**Δ(GRPO − base) = +9.8 pp.** McNemar exact **p = 2×10⁻⁶** (discordant 161 base-only vs 259
GRPO-only); paired bootstrap 95% CI **[+5.8, +13.8] pp**.

**The paper reported +9.8 pp on the contaminated benchmark (38.7% → 48.5%). The disjoint
benchmark gives +9.8 pp (30.0% → 39.8%). The effect size is identical.**

This is the single best result of the triage. The contamination in F0/F0b **did not
manufacture the GRPO improvement** — the central claim survives a benchmark built from
plasmids the model has never seen. It is the direct, empirical answer to the meta-review's
"substantial overlap would undermine both the held-out rates and the claim of de novo
generation": the *rates* shift down, the *claim about GRPO* does not.

**Absolute rates fall ~9 pp** (38.7/48.5 → 30.0/39.8). Three contributors, not separable
here: prompts carry no `VEC_*`/soft tokens and are mildly OOD (F10); the scoring path is
stricter than the registry-BLAST used in the paper; and unseen plasmids are genuinely harder.
Report the delta as the robust quantity and the levels as not directly comparable.

**Decomposition — and it corrects the prediction I went in with.** I expected the gap to
retrieval to be mostly viability. It is *both*, with faithfulness the larger:

| axis | GRPO | retrieval | ratio |
|---|---|---|---|
| viable | 61.6% | ~95% | 0.65 |
| faithful @0.5 | 40.5% | 100% | 0.41 |

So the model loses about a third of the gap on viability and closer to two-thirds on
faithfulness. Both need saying; "it's a viability gap" would have been wrong.

Scripts: [`eval/scripts/build_disjoint_benchmark.py`](../../eval/scripts/build_disjoint_benchmark.py),
[`eval/scripts/score_disjoint.py`](../../eval/scripts/score_disjoint.py).

### F15 — ✅ Best-of-4 replicates on the disjoint benchmark too. **MEASURED.**

The abstract's second headline (89.7% best-of-4) had no honest version — the disjoint run
was single-shot only. Generated 3 more seeds for each checkpoint (seeds 43/44/45; the
single-shot seed 42 is the 4th draw), oracle ranker over K draws, same scoring path.

| K | base | post-GRPO |
|---|---|---|
| 1 | 30.0% | 39.8% |
| 2 | 47.7% | 61.6% |
| 4 | **67.0%** [64.0, 69.8] | **80.5%** [77.9, 82.8] |

Paper reported 78.9% → 89.7% best-of-4 (base → GRPO). Disjoint gives 67.0% → 80.5%. Same
shape: GRPO strictly above base at every K (+9.8 to +13.9 pp), absolute levels ~9-12 pp
lower — consistent with F14. The best-of-K story survives the honest benchmark.

⚠️ Still the same ranker caveat as the paper (pSXh W6): best-of-K evaluates a
model-plus-annotation-ranker system, not one-pass reliability. And the ~95% retrieval
baseline (F13) sits above even best-of-4, so best-of-K does not change the metric-saturation
picture.

### F16 — ⚠️ The model does **not** compose: novel-combination benchmark. **MEASURED — negative, and important.**

The experiment McClain most wanted, and the honest answer is not the one we hoped for.
Built 500 prompts by substituting one well-supported component into a real held-out
plasmid's set, verified absent from every training plasmid ("pUC19 backbone but
chloramphenicol instead of ampicillin"). By construction retrieval is capped at F=1.0 → 0%.

| panel | viable | F≥0.5 | **F=1.0 (full spec)** |
|---|---|---|---|
| retrieval (ceiling) | — | 100.0% | **0.0%** (by construction) |
| PlasmidLM base | 44.6% | 16.6% | **0.0%** [0.0, 0.8] |
| PlasmidLM GRPO | 61.0% | 26.4% | **0.0%** [0.0, 0.8] |

**The capability claim I set this up to make does not hold.** It is not "only the generative
model satisfies novel specs" — *neither* does, at full satisfaction. The model scores 0%.

The mechanism, and it is the sharpest diagnostic in the whole triage: when asked for the
substituted-**in** component, the model produces it **4.6%** of the time (base 3.4%), and
**reverts to the substituted-out component 40.2%** of the time (base 28.8%). It defaults to
the familiar training co-occurrence instead of following the instruction. That is the same
long-tail / co-occurrence failure the paper already documents (§5.3), now shown directly for
composition — the model has learned plasmid *templates*, not a compositional grammar over parts.

**What survives:** the GRPO gain replicates even here. useful@T=0.5: 16.6% → 26.4%, +9.8 pp,
CI [+4.8, +14.8], McNemar p=2e-4 — identical in size to F14/F15. GRPO robustly improves
partial satisfaction on novel specs; it does not buy compositional generalization.

**Consequence for the rebuttal — this is load-bearing.** F16 means we **cannot** claim
compositional / de-novo design as a demonstrated capability. The honest contribution narrows
to: *promptable generation of novel sequence realising component specifications that resemble
training combinations*, with GRPO improving partial faithfulness. Retrieval beats us on
in-distribution specs (F13); we beat retrieval on nothing measured so far at full
satisfaction. The paper's designer-facing framing ("hand the model a spec, get the
construct") is not supported for novel specs. Better we found this now.

Scripts: `experiments/neurips_rebuttal/novel_combination.py`, `score_novel.py`.

### F17 — ⚠️ Unconditional generation matches PlasmidLM on the useful rate. **MEASURED.**

Completion-only comparison (SeFq's #1 ask). PlasmidGPT/Shao and PlasmidGPT-SFT/Cunningham have
no interface to request components; scoring their *unconditional* samples against the specs
gives the base rate an unsteerable generator hits. Shao (median 15 kb, so generous — longer
sequences contain more components by chance) scored through the same pLannotate path:

| panel | benchmark | viable | F≥0.5 | F=1.0 | useful |
|---|---|---|---|---|---|
| Shao (unconditional) | disjoint | 64.2% | 40.3% | 8.6% | **39.9%** |
| PlasmidLM GRPO | disjoint | 61.6% | 40.5% | 5.3% | **39.8%** |
| Shao (unconditional) | novel | 64.2% | 30.6% | 0.4% | **30.2%** |
| PlasmidLM GRPO | novel | 61.0% | 26.4% | 0.0% | **26.4%** |

**Unconditional generation matches (or slightly beats) PlasmidLM on the useful-plasmid rate on
both benchmarks.** Same lesson as retrieval (F13): the useful rate is dominated by common
components that any plausible plasmid contains, so it does not measure prompt control. And on
novel combinations, PlasmidLM produces the requested substitute **4.6%** of the time — *below*
Shao's unconditional base rate of **6.2%**. By the useful rate, prompting buys nothing.

### F18 — ✅ …but prompt-conditional lift shows PlasmidLM has real control for ~44% of components. **MEASURED — the metric the paper should have led with.**

The useful rate can't separate control from base rate. This can: for each component, does
requesting it make it appear more than when it is *not* requested?

    lift(t) = P(t detected | t requested) − P(t detected | t NOT requested)

PlasmidLM GRPO, disjoint benchmark, 41 components with enough support:

| component | requested→present | not-requested→present | lift |
|---|---|---|---|
| AMR_ZEOCIN | 38% | 4% | **+34** |
| PROM_CMV | 31% | 11% | **+21** |
| ORI_F1 | 58% | 38% | **+20** |
| ORI_COLE1 | 63% | 45% | **+18** |
| ELEM_CMV_ENHANCER | 33% | 16% | **+17** |
| AMR_NEOMYCIN | 18% | 2% | **+16** |
| … | | | |
| AMR_CHLORAMPHENICOL | 0% | 0% | −0 |
| ELEM_MCS | 0% | 8% | −8 |

**Mean lift +7.8 pp; 18 of 41 components show lift > +10 pp**, up to +34. The model genuinely
responds to the prompt for a substantial fraction of the vocabulary; the near-zero-lift
components are exactly the known rare tail (§5.3). Prompting works, measurably — it is just
invisible in the useful rate because common components are common regardless.

**This is the honest, defensible spine for the paper.** Replace the saturated useful rate with
prompt-conditional lift as the primary controllability metric. It: (a) is not matched by
retrieval or unconditional generation (both have lift ≈ 0 by construction — their output is
independent of the prompt); (b) shows real control for ~44% of components; (c) maps exactly
where control fails (the tail, and novel combinations, F16). It converts the reviewers' central
objection — "the useful rate doesn't establish control" — into a metric that does, and it is
strictly more informative than what was submitted.

**The head-to-head that proves lift is the right metric** (disjoint benchmark):

| model | useful rate | **mean lift** | components lift>+10pp |
|---|---|---|---|
| PlasmidLM GRPO | 39.8% | **+7.8 pp** | 18/41 |
| Shao (unconditional) | 39.9% | **−0.4 pp** | 1/41 |
| Cunningham (unconditional) | **49.2%** | **−0.3 pp** | 0/41 |

Cunningham (22 kb sequences) *beats* PlasmidLM on the useful rate (49 vs 40) yet has **zero
lift** — its output is independent of the prompt. This is the sharpest statement of the whole
rebuttal: the useful rate rewards long unconditional generation, lift does not. Lift separates
PlasmidLM from both baselines by ~8 pp while the useful rate cannot separate them at all.

**Nuance — the control is a pretraining property, not a GRPO gain.** Base and GRPO have
nearly equal lift (base +8.6 pp, 19/41 above +10; GRPO +7.8 pp, 18/41), and GRPO has *more*
zero-lift components (8 vs 3) — consistent with F11 (GRPO adds unrequested components, raising
P(present|not requested) and eroding lift). So: prompting control comes from supervised
pretraining; GRPO improves the useful *rate* (recall, +9.8 pp, F14) but does not add control
and slightly reduces its precision. All three statements are true and should be stated
together — it is a more accurate account of what GRPO does than the submission's.

Scripts: `experiments/neurips_rebuttal/control_lift.py`, `baselines.py completion`.

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
| ~~C1′~~ | ~~Can a genuinely disjoint prompt pool be built?~~ | **RESOLVED — yes, but it must be built.** The existing holdout is not disjoint (F0b); 120,449 unseen plasmids are available (F10); compute is already running. | **RESOLVED** |
| **C6** | Deadline is **27 Jul** (3 days), response is **text-only — no paper revisions, no external links** (supervision meeting, 24 Jul). Does M2.0 fit? | Generation ~90 min + pLannotate ~30 min on g6-big2. Fits comfortably *if* started now. | Feasible; needs your go-ahead |
| ~~C2~~ | ~~Eval-prompt ∩ GRPO-rollout-prompt overlap?~~ | **Answered: 100%.** All 1,000 eval prompts are in `training_pairs_v4.parquet`, over which GRPO rolled out. | **RESOLVED** |
| **C3** | Are `hmmer` + Pfam, or AMRFinderPlus, installable/available on g6-big or p4 in-window? | Gates whether the orthogonal-annotator answer (Dok5 Q3, meta M1) is deliverable. | Not checked |
| **C4** | Which paper source is the NeurIPS submission? | `~/Projects/PhD/plasmidlm-paper` is on branch **`icml-workshop`** with 65 uncommitted insertions to `main.tex` and untracked `icml2026.sty`. The tree is being repurposed for an ICML workshop submission. Editing `main.tex` here risks clobbering that work. | **Needs your call** — see §5 |
| **C5** | Given F0+F9, confirm the rebuttal strategy: concede process errors fully, contest the memorisation inference with F9. | F0 must be disclosed either way; F9 means disclosure is survivable rather than fatal. | **Needs your sign-off** — see §5 |

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
| M1.2 | ✅ **DONE — see F12.** AMRFinderPlus on archived prodigal proteins. Absolute recovery halves (26.1%→13.5%); GRPO delta +1.0 pp, **not significant** (95% CI [−1.17,+3.27], McNemar p=0.30). | 🔴 | **COMPLETE** | Answers pSXh Q1 / Dok5 Q3. Does **not** rescue M1 — underpowered, and only ~30% of requested tokens are protein-coding and thus independently checkable at all. |
| M1.3 | **Held-out registry split.** Hold out a subset of the 660 registry entries from the reward, evaluate faithfulness only on held-out entries. | 🟠 | **NO** for the rebuttal | Requires GRPO retraining (~17 h + Ray setup) *and* re-eval. Propose as camera-ready/future work; do not promise. |
| M1.4 | **Fill Appendix A.1** from `plannotate.py:456-520`. | 🟡 | **NOW** | Named explicitly by the AC. Trivial. Do it first. |
| M1.5 | **Recalibrate biological claims.** Systematic pass replacing "viable" → "annotatable under our rule"; state plainly that viability = origin + marker + (promoter or CDS) **detected**, and does not check ORF integrity, reading-frame coherence, or origin completeness. | 🟠 | **NOW** | Dok5 W3 is correct and we partially concede it already. Owning it fully is cheaper than defending. |

### M2 — Independence from the training corpus / retrieval effects. 🔴

| # | Action | Sev | Feas | Notes |
|---|---|---|---|---|
| **M2.0** | 🚨 **Re-run the benchmark on a genuinely plasmid-ID-disjoint prompt pool.** Regenerate n=1,000 for base and GRPO, re-annotate, recompute useful-plasmid rate and best-of-K. Report the new numbers whatever they are. | 🔴🔴 | **WINDOW**, blocked on C1′ | Consequence of F0. Generation is ~46 min/1,000 per the eval log; pLannotate ~35 min at 8 workers. The compute is the easy part — **C1′ is the blocker.** Nothing else in M2 matters until this is settled. |
| M2.1 | **Correct the benchmark description in §3.1** and every "held-out" claim (abstract, §4, conclusion). Also state that the 95/5 train/val split is pair-level (F3). | 🔴 | **NOW** | Required regardless of rebuttal outcome. This is a correction, not a rebuttal tactic. |
| M2.2 | ✅ **DONE — see F9.** BLASTN of all generations vs the 20,644 training plasmids, mosaic (multi-subject) coverage, real held-out Addgene control, length-stratified. Median generation 43% explained vs 81% for real held-out. | 🔴 | **COMPLETE** | `eval/scripts/train_similarity_blast.py`. **Strongest single answer to the memorisation half of M2** and the centrepiece of the response. Re-run on M2.0 outputs when they exist. |
| M2.3 | ✅ **DONE.** Retrieval 95.0% vs post-GRPO **39.8%** on the same benchmark, same detector. **Retrieval wins by ~55 pp.** | 🔴🔴 | **COMPLETE** | Confirms the metric is saturated by lookup. But F14 shows the GRPO effect itself is real and replicates — so this is a metric problem, not a "the model does nothing" problem. Reframe, don't withdraw. |
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
2. **C5** — sign off on the concede-process / contest-memorisation strategy.
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
14. **M2.3 retrieval baseline** — promoted; the untested half of M2 and the single item most
able to sink the contribution · 15. **M1.1 precision/specificity** · (M2.2 ✅ done — F9.)
⚠️ M1.1, M2.2 and M2.2b should be **re-run on the M2.0 outputs** once those exist; the current
runs characterise a benchmark we are about to retire.

**Phase 4 — needs setup.**
17. M1.2 orthogonal annotator · 18. M3.3 PlasmidGPT comparison · 19. M2.4 clustering.

**Phase 5 — assemble.** Draft per-reviewer responses; global revision; PDF diff.

> **Run before you promise.** M2.0, M1.1, M2.2b and M2.3 can each come back *against* us. Get
> the numbers before committing to any language in the response. (This rule already paid for
> itself once: it caught that F0's severity was overstated before the withdraw framing reached you.)

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
