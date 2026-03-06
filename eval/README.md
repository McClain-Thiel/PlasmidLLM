# eval

Evaluation for PlasmidLLM. Two independent axes — **model quality** and **scorer validity** — plus a visualization layer.

```
eval/
├── data/               # Fixture parquets (tracked in git)
│   └── training_pairs_sample.parquet   # Small sample of real (prompt, sequence) pairs
├── models/             # Does the model produce good plasmids?
├── scorers/            # Do the scorers separate good from bad?
└── plasmidspace/       # UI for exploring generated plasmids (TBD)
```

---

## data/ — Shared Fixtures

All evaluation data lives in `eval/data/` as parquet files. Parquet over FASTA because we need the prompt (with expected tokens) alongside the sequence, and parquet compresses well for git.

**Source file:** `training_pairs_sample.parquet` — a small sample (~50-100 rows) from the training set. Schema matches `PlasmidDataset` expectations:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Token prompt, e.g. `<BOS><VEC_MAMMALIAN><AMR_KANAMYCIN><ORI_PBR322><PROM_CMV><SEQ>` |
| `sequence` | str | Full plasmid DNA sequence |
| `sorted_tokens` | list[str] | Individual tokens in canonical order |
| `plasmid_id` | int | Addgene plasmid ID |
| `sequence_length` | int | Length of DNA sequence |

Corrupted and random negatives are derived programmatically from this file (no separate files needed):

| Negative class | How to generate |
|----------------|-----------------|
| **Mismatched** | Shuffle the prompt-sequence pairing: score sequence_i against prompt_j |
| **Random DNA** | `''.join(random.choices('ATGC', k=len(seq)))` matched to each prompt |
| **Shuffled** | Dinucleotide-shuffle each real sequence (preserves GC%, destroys features) |
| **Truncated** | Slice real sequences to 25%, 50%, 75% of original length |
| **Partial** | Remove specific tokens from the prompt, keep the same sequence |

---

## models/ — Model Evaluation

**Question:** Given a prompt requesting specific plasmid components, how well does the model's output match the request?

### What we measure

For each (prompt, generated_sequence) pair, annotate the generated sequence against the motif registry and compute:

| Metric | Description |
|--------|-------------|
| **True Positives (TP)** | Requested components found in the output |
| **False Negatives (FN)** | Requested components missing from the output |
| **False Positives (FP)** | Components present in the output that were not requested |
| **Recall** | TP / (TP + FN) — fraction of requested features successfully placed |
| **Precision** | TP / (TP + FP) — fraction of detected features that were actually requested |
| **Per-component quality** | For each TP: percent identity, coverage, and normalized alignment score |
| **QC pass rate** | Fraction of found components exceeding quality thresholds (e.g. ≥70% score ratio) |
| **EOS rate** | Whether the model terminates cleanly with an EOS token |
| **Sequence length** | Total bp generated |

### Per-component quality detail

Each found component gets an alignment-level breakdown:

- **pct_id** — percent identity from CIGAR alignment
- **coverage** — fraction of the reference motif covered
- **norm_score** — alignment score normalized by motif length
- **seq_type** — whether the best hit was DNA or protein (6-frame translation)
- **location** — (start, end) in the generated sequence

This lets us distinguish "the model placed something vaguely AMR-like" from "the model placed a near-perfect copy of the requested resistance gene."

### Aggregation levels

1. **Per-sequence** — full annotation card for one generation
2. **Per-category** — hit rate / quality by component type (ORI, AMR, PROM, REPORTER, TAG, ELEM)
3. **Per-model** — aggregate across a validation set, comparable across checkpoints
4. **Per-temperature** — sweep to find optimal decoding settings

### Baselines

- **Random DNA** — uniform random ATGC of matched length (should score ~0)
- **Shuffled real plasmids** — dinucleotide-shuffled (preserves GC%, destroys features)
- **Real plasmids** — ground-truth sequences scored against their own prompts (ceiling)

---

## scorers/ — Scorer Evaluation

**Question:** Does a scorer reliably separate good plasmids from bad ones, making it a useful RL training signal?

Scorers implement the `Scorer` ABC from `post_training/scorers/base.py`:

```python
class Scorer(ABC):
    def score_sequence(self, prompt: str, sequence: str, **kwargs) -> float: ...
    def score_batch(self, prompts: list[str], sequences: list[str], **kwargs) -> list[float]: ...
    def score_sequence_detailed(self, prompt: str, sequence: str, **kwargs) -> dict[str, Any]: ...
```

### Approach

Load `training_pairs_sample.parquet`, then for each scorer run it against multiple input classes and measure separation.

**Input classes** (all derived from the same parquet):

| Class | Construction | Expected score |
|-------|-------------|----------------|
| **Real matched** | `scorer.score_batch(prompts, sequences)` — original pairs | High |
| **Mismatched** | `scorer.score_batch(prompts, sequences[shuffled_idx])` — wrong pairings | Low-moderate |
| **Random DNA** | `scorer.score_batch(prompts, [random_atgc(len(s)) for s in sequences])` | Near zero |
| **Shuffled** | `scorer.score_batch(prompts, [dinuc_shuffle(s) for s in sequences])` | Near zero |
| **Truncated 50%** | `scorer.score_batch(prompts, [s[:len(s)//2] for s in sequences])` | Reduced |
| **Partial prompt** | `scorer.score_batch(drop_tokens(prompts), sequences)` — fewer expected tokens | Higher (fewer to miss) |

### Metrics

| Metric | Description |
|--------|-------------|
| **Separation** | mean(real_matched) − mean(negative_class) for each negative |
| **AUROC** | Real-matched vs each negative class as binary classification |
| **Score distributions** | Histograms per class — looking for non-overlapping IQRs |
| **Rank correlation** | Spearman between scorers on the same sequences (do they agree?) |
| **Graceful degradation** | Score vs truncation fraction — should be monotonically decreasing |
| **Per-component attribution** | Via `score_sequence_detailed` — does score come from the right components? |

### Scorer-specific checks

**AlignmentScorer** (`alignment.py`):
- Score-ratio saturates at 1.0 for perfect matches
- EOS bonus only fires when EOS present
- Protein alignment (6-frame) catches CDS features that DNA-only misses

**MotifScorer** (`motif.py`):
- Three-pass filter doesn't discard true positives (compare all-filters vs no-filters)
- Composite score is monotonic with recall
- K-mer and score-only pre-filters affect speed only, not final scores

### Pass criteria

A scorer is ready for RL when:

1. **Clean separation** — mean(real) > 3× mean(random) with non-overlapping IQRs
2. **Graceful degradation** — truncated sequences score proportionally to completeness
3. **No false ceilings** — real plasmids with all components score near the scorer's max
4. **Correct attribution** — `score_sequence_detailed` attributes signal to the right components

---

## plasmidspace/ — Visualization UI

TBD. Interactive explorer for generated plasmids — plasmid maps, annotation overlays, score breakdowns, comparative views across models/temperatures.
