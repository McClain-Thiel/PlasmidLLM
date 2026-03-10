# Databricks notebook source
# MAGIC %md
# MAGIC # Reward Function for PlasmidGPT Post-Training
# MAGIC
# MAGIC **Goal:** Given a prompt (set of special tokens) and a generated DNA sequence, compute a scalar reward ∈ [0, 1]
# MAGIC reflecting how well the sequence contains the expected functional motifs.
# MAGIC
# MAGIC **Approach:**
# MAGIC 1. Parse hard tokens from the prompt
# MAGIC 2. Look up canonical motif sequences from the motif registry
# MAGIC 3. For each expected motif:
# MAGIC    - **Non-coding** (ORI, PROM, ELEM): DNA-level Smith-Waterman (both strands)
# MAGIC    - **CDS** (AMR, REPORTER, TAG): DNA + protein-level alignment (6-frame translation), take the max
# MAGIC 4. Score = `alignment_score / max_possible_score` (precomputed via self-alignment)
# MAGIC 5. Aggregate per-motif scores → mean = final reward
# MAGIC
# MAGIC **Scoring metric:** `score_ratio = sw_score / self_alignment_score`. No CIGAR parsing needed.
# MAGIC This naturally captures both identity and coverage in a single metric.
# MAGIC
# MAGIC **Performance:** `parasail` (SIMD-accelerated Smith-Waterman) — ~57ms per example.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

# !pip install parasail biopython

import pandas as pd
import numpy as np
from Bio.Seq import Seq
from typing import Dict, List, Tuple, Optional
import parasail
import json
import time
import re

try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, size, rand
except ImportError:
    pass


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Motif Registry
# MAGIC
# MAGIC Build lookup: `token → {dna_seq, protein_seq, dna_max_score, protein_max_score}`.
# MAGIC
# MAGIC **Critical fix:** The registry has `seq_type ∈ {dna, protein, None}`.
# MAGIC Entries with `seq_type='protein'` already have protein sequences in the `sequence` field — 
# MAGIC we must NOT try to translate those as DNA.
# MAGIC

# COMMAND ----------

# ── Scoring config ──
DNA_MATRIX = parasail.matrix_create("ACGT", 1, -1)
PROTEIN_MATRIX = parasail.blosum62

DNA_OPEN = 5
DNA_EXTEND = 1
PROTEIN_OPEN = 10
PROTEIN_EXTEND = 1

CDS_CATEGORIES = {"AMR", "REPORTER", "TAG"}
DNA_CATEGORIES = {"ORI", "PROM", "ELEM"}


# COMMAND ----------

# ── Load from Databricks ──
motif_registry_df = spark.read.table("addgene.default.motif_registry")

motif_lookup_rows = (
    motif_registry_df
    .select("token", "sseqid", "sequence", "seq_len", "category", "seq_type")
    .distinct()
    .toPandas()
)

print(f"Loaded {len(motif_lookup_rows)} motif rows")
print(f"Categories: {sorted(motif_lookup_rows['category'].unique())}")
print(f"Seq types:  {motif_lookup_rows['seq_type'].unique()}")
print(f"seq_type counts:\n{motif_lookup_rows['seq_type'].value_counts(dropna=False)}")


# COMMAND ----------

def safe_translate(dna_seq: str) -> Optional[str]:
    """Translate DNA to protein. Returns None if invalid."""
    try:
        trimmed = dna_seq[:len(dna_seq) - (len(dna_seq) % 3)]
        if len(trimmed) < 3:
            return None
        return str(Seq(trimmed).translate())
    except:
        return None

def compute_max_score_dna(seq: str) -> int:
    """Self-alignment score for a DNA sequence = max possible SW score."""
    if not seq or len(seq) == 0:
        return 1  # avoid division by zero
    return parasail.sw_striped_16(seq, seq, DNA_OPEN, DNA_EXTEND, DNA_MATRIX).score

def compute_max_score_protein(seq: str) -> int:
    """Self-alignment score for a protein sequence."""
    if not seq or len(seq) == 0:
        return 1
    return parasail.sw_striped_16(seq, seq, PROTEIN_OPEN, PROTEIN_EXTEND, PROTEIN_MATRIX).score


# COMMAND ----------

# ── Build the lookup with proper seq_type handling ──
MOTIF_LOOKUP = {}

for _, row in motif_lookup_rows.iterrows():
    token = row["token"]
    seq = row["sequence"]
    seq_type = row["seq_type"]
    category = row["category"]
    is_cds = category in CDS_CATEGORIES
    
    if not seq or len(seq) == 0:
        continue
    
    if token not in MOTIF_LOOKUP:
        MOTIF_LOOKUP[token] = {
            "category": category,
            "is_cds": is_cds,
            "variants": [],
        }
    
    variant = {"sseqid": row["sseqid"]}
    
    if seq_type == "protein":
        # Sequence is already protein — do NOT treat as DNA
        variant["dna_seq"] = None
        variant["dna_len"] = 0
        variant["dna_max_score"] = 1
        variant["protein_seq"] = seq
        variant["protein_len"] = len(seq)
        variant["protein_max_score"] = compute_max_score_protein(seq)
    else:
        # Sequence is DNA (seq_type = 'dna' or None)
        variant["dna_seq"] = seq
        variant["dna_len"] = len(seq)
        variant["dna_max_score"] = compute_max_score_dna(seq)
        
        # For CDS categories, also pre-translate to protein
        if is_cds:
            protein = safe_translate(seq)
            variant["protein_seq"] = protein
            variant["protein_len"] = len(protein) if protein else 0
            variant["protein_max_score"] = compute_max_score_protein(protein) if protein else 1
        else:
            variant["protein_seq"] = None
            variant["protein_len"] = 0
            variant["protein_max_score"] = 1
    
    MOTIF_LOOKUP[token]["variants"].append(variant)

# Build fast lookup (longest variant per token)
MOTIF_LOOKUP_FAST = {}
for token, entry in MOTIF_LOOKUP.items():
    # Prefer DNA length, fall back to protein length
    best = max(entry["variants"], key=lambda v: v["dna_len"] or v.get("protein_len", 0))
    MOTIF_LOOKUP_FAST[token] = {
        "category": entry["category"],
        "is_cds": entry["is_cds"],
        "variants": [best],
    }

print(f"\nMotif lookup: {len(MOTIF_LOOKUP)} tokens")
print(f"  CDS (protein+DNA scoring): {sum(1 for v in MOTIF_LOOKUP.values() if v['is_cds'])}")
print(f"  Non-coding (DNA scoring):  {sum(1 for v in MOTIF_LOOKUP.values() if not v['is_cds'])}")

# Sanity check: show a few entries
for tok in list(MOTIF_LOOKUP_FAST.keys())[:3]:
    v = MOTIF_LOOKUP_FAST[tok]["variants"][0]
    print(f"  {tok}: dna_len={v['dna_len']}, dna_max={v['dna_max_score']}, "
          f"prot_len={v.get('protein_len',0)}, prot_max={v.get('protein_max_score',1)}, "
          f"is_cds={MOTIF_LOOKUP_FAST[tok]['is_cds']}")


# COMMAND ----------

# ── Option B: Save/load as JSON for non-Databricks use ──
# Save (run once in Databricks):
# with open("/dbfs/FileStore/plasmidgpt/motif_lookup.json", "w") as f:
#     json.dump(MOTIF_LOOKUP, f)
# with open("/dbfs/FileStore/plasmidgpt/motif_lookup_fast.json", "w") as f:
#     json.dump(MOTIF_LOOKUP_FAST, f)

# Load (run anywhere):
# with open("motif_lookup_fast.json") as f:
#     MOTIF_LOOKUP_FAST = json.load(f)
# NOTE: After loading from JSON, you must recompute max_scores since parasail
# objects can't be serialized. The max_scores are ints so they survive JSON fine.


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Alignment Functions (Score-Ratio)
# MAGIC
# MAGIC No CIGAR parsing. Just: `score_ratio = sw_score / self_alignment_score`.
# MAGIC
# MAGIC This naturally captures both identity and coverage:
# MAGIC - Perfect match → 1.0
# MAGIC - 90% identity over full length → ~0.8
# MAGIC - Perfect identity over 50% of motif → 0.5  
# MAGIC - Random sequence → ~0.0
# MAGIC
# MAGIC For CDS motifs, we try **both** DNA and protein alignment and take the max.
# MAGIC This means ground truth (exact DNA) always scores 1.0, while RL-generated 
# MAGIC sequences with synonymous codon changes still get credit via protein scoring.
# MAGIC

# COMMAND ----------

def align_dna_score(motif_dna: str, candidate_seq: str, max_score: int) -> dict:
    """
    DNA Smith-Waterman, both strands. Returns score_ratio ∈ [0, 1].
    """
    if not motif_dna or len(motif_dna) == 0 or len(candidate_seq) == 0:
        return {"score_ratio": 0.0, "raw_score": 0, "strand": 0}
    
    motif_rev = str(Seq(motif_dna).reverse_complement())
    
    score_fwd = parasail.sw_striped_16(
        candidate_seq, motif_dna, DNA_OPEN, DNA_EXTEND, DNA_MATRIX
    ).score
    
    score_rev = parasail.sw_striped_16(
        candidate_seq, motif_rev, DNA_OPEN, DNA_EXTEND, DNA_MATRIX
    ).score
    
    if score_fwd >= score_rev:
        best_score = score_fwd
        strand = 1
    else:
        best_score = score_rev
        strand = -1
    
    score_ratio = min(best_score / max(max_score, 1), 1.0)
    
    return {
        "score_ratio": round(score_ratio, 4),
        "raw_score": best_score,
        "strand": strand,
        "level": "dna",
    }


# COMMAND ----------

def align_protein_score(motif_protein: str, candidate_seq: str, max_score: int) -> dict:
    """
    Protein Smith-Waterman. Translates candidate in 6 frames, aligns each.
    Returns score_ratio ∈ [0, 1].
    """
    if not motif_protein or len(motif_protein) == 0 or len(candidate_seq) < 3:
        return {"score_ratio": 0.0, "raw_score": 0, "frame": None}
    
    fwd = candidate_seq.upper()
    rev = str(Seq(fwd).reverse_complement())
    
    best_score = 0
    best_frame = None
    
    for frame_offset in range(3):
        for seq, sign in [(fwd, "+"), (rev, "-")]:
            subseq = seq[frame_offset:]
            subseq = subseq[:len(subseq) - (len(subseq) % 3)]
            if len(subseq) < 3:
                continue
            try:
                prot = str(Seq(subseq).translate())
            except:
                continue
            if len(prot) == 0:
                continue
            
            score = parasail.sw_striped_16(
                prot, motif_protein, PROTEIN_OPEN, PROTEIN_EXTEND, PROTEIN_MATRIX
            ).score
            
            if score > best_score:
                best_score = score
                best_frame = f"{sign}{frame_offset}"
    
    score_ratio = min(best_score / max(max_score, 1), 1.0)
    
    return {
        "score_ratio": round(score_ratio, 4),
        "raw_score": best_score,
        "frame": best_frame,
        "level": "protein",
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reward Function
# MAGIC

# COMMAND ----------

HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}

def parse_hard_tokens(prompt: str, motif_lookup: dict = None) -> List[str]:
    """Extract hard tokens from prompt string."""
    tokens = re.findall(r'<[^>]+>', prompt)
    hard_tokens = []
    for t in tokens:
        inner = t.strip("<>")
        if any(inner.startswith(p) for p in HARD_PREFIXES):
            if motif_lookup is None or t in motif_lookup:
                hard_tokens.append(t)
    return hard_tokens


# COMMAND ----------

def score_motif(token: str, candidate_seq: str, motif_lookup: dict) -> dict:
    """
    Score a single motif against a candidate sequence.
    
    For CDS tokens: tries both DNA and protein alignment, takes the max.
    For non-coding tokens: DNA alignment only.
    """
    if token not in motif_lookup:
        return {"token": token, "score_ratio": 0.0, "found": False, "error": "not_in_registry"}
    
    entry = motif_lookup[token]
    is_cds = entry["is_cds"]
    
    best_score_ratio = 0.0
    best_detail = None
    best_sseqid = None
    
    for variant in entry["variants"]:
        results_for_variant = []
        
        # DNA alignment (if DNA sequence available)
        if variant.get("dna_seq"):
            dna_result = align_dna_score(
                variant["dna_seq"], candidate_seq, variant["dna_max_score"]
            )
            results_for_variant.append(dna_result)
        
        # Protein alignment (if CDS and protein available)
        if is_cds and variant.get("protein_seq"):
            prot_result = align_protein_score(
                variant["protein_seq"], candidate_seq, variant["protein_max_score"]
            )
            results_for_variant.append(prot_result)
        
        # Take best across DNA/protein for this variant
        for r in results_for_variant:
            if r["score_ratio"] > best_score_ratio:
                best_score_ratio = r["score_ratio"]
                best_detail = r
                best_sseqid = variant["sseqid"]
    
    if best_detail is None:
        return {"token": token, "score_ratio": 0.0, "found": False, "error": "no_alignable_seq"}
    
    # Binary QC: is this motif "found" with a strong match?
    found = best_score_ratio >= 0.70  # 70% of max score
    
    return {
        "token": token,
        "best_sseqid": best_sseqid,
        "score_ratio": best_score_ratio,
        "found": found,
        "raw_score": best_detail.get("raw_score"),
        "level": best_detail.get("level", "dna"),
        "strand": best_detail.get("strand"),
        "frame": best_detail.get("frame"),
    }


# COMMAND ----------

def compute_reward(
    prompt: str,
    candidate_seq: str,
    motif_lookup: dict,
    return_details: bool = False,
) -> float | Tuple[float, dict]:
    """
    Compute reward for a generated sequence given a prompt.
    Reward = mean of per-motif score_ratios for all hard tokens.
    """
    candidate_seq = candidate_seq.upper().strip()
    hard_tokens = parse_hard_tokens(prompt, motif_lookup)
    
    if not hard_tokens:
        details = {"reward": 0.0, "n_hard_tokens": 0, "n_found": 0, "qc_pass_rate": 0.0, "per_motif": []}
        return (0.0, details) if return_details else 0.0
    
    per_motif = [score_motif(t, candidate_seq, motif_lookup) for t in hard_tokens]
    
    scores = [m["score_ratio"] for m in per_motif]
    n_found = sum(1 for m in per_motif if m["found"])
    reward = float(np.mean(scores))
    qc_pass_rate = n_found / len(per_motif)
    
    details = {
        "reward": round(reward, 4),
        "n_hard_tokens": len(hard_tokens),
        "n_found": n_found,
        "qc_pass_rate": round(qc_pass_rate, 4),
        "per_motif": per_motif,
    }
    
    return (reward, details) if return_details else reward


# COMMAND ----------

def plasmid_reward_fn(
    prompts: List[str],
    completions: List[str],
    motif_lookup: dict,
) -> List[float]:
    """Batch reward function for RL training (GRPO/PPO compatible)."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        seq = completion.upper()
        for tag in ["<SEQ>", "<EOS>", "<BOS>", "<PAD>", "<UNK>"]:
            seq = seq.replace(tag, "")
        seq = re.sub(r'<[^>]+>', '', seq)
        seq = re.sub(r'[^ATGCN]', '', seq)
        
        if len(seq) < 100:
            rewards.append(0.0)
            continue
        rewards.append(compute_reward(prompt, seq, motif_lookup))
    
    return rewards


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test on Training Data
# MAGIC
# MAGIC Ground-truth sequences should score > 0.9 since the motifs are known to be present 
# MAGIC (verified by BLAST at 95%+ identity, 80%+ coverage).
# MAGIC

# COMMAND ----------

training_pairs = spark.read.table("addgene.default.training_pairs")

sample = (
    training_pairs
    .filter(col("reward_motifs").isNotNull() & (size(col("reward_motifs")) >= 1))
    .orderBy(rand(seed=42))
    .limit(20)
    .toPandas()
)

print(f"Loaded {len(sample)} training pairs")
sample[["plasmid_id", "prompt", "n_tokens", "sequence_length"]].head(10)


# COMMAND ----------

# Score each example
results = []
for idx, row in sample.iterrows():
    t0 = time.time()
    reward, details = compute_reward(
        row["prompt"], row["sequence"], MOTIF_LOOKUP_FAST, return_details=True,
    )
    elapsed = time.time() - t0
    
    results.append({
        "plasmid_id": row["plasmid_id"],
        "n_hard_tokens": details["n_hard_tokens"],
        "n_found": details["n_found"],
        "qc_pass_rate": details["qc_pass_rate"],
        "reward": details["reward"],
        "seq_len": row["sequence_length"],
        "time_sec": round(elapsed, 3),
    })

results_df = pd.DataFrame(results)
print("=== Ground Truth Scoring (parasail, score-ratio) ===")
print(f"Mean reward:       {results_df['reward'].mean():.4f}")
print(f"Mean QC pass rate: {results_df['qc_pass_rate'].mean():.4f}")
print(f"Mean time (sec):   {results_df['time_sec'].mean():.3f}")
print(f"Total time:        {results_df['time_sec'].sum():.1f}s for {len(results_df)} examples")
print()
results_df


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Inspect one example in detail

# COMMAND ----------

example_row = sample.iloc[0]
reward, details = compute_reward(
    example_row["prompt"], example_row["sequence"],
    MOTIF_LOOKUP_FAST, return_details=True,
)

print(f"Plasmid: {example_row['plasmid_id']}")
print(f"Prompt:  {example_row['prompt']}")
print(f"Seq len: {example_row['sequence_length']}")
print(f"Reward:  {details['reward']}")
print(f"QC pass: {details['qc_pass_rate']} ({details['n_found']}/{details['n_hard_tokens']})")
print()
for m in details["per_motif"]:
    status = "PASS" if m["found"] else "FAIL"
    level = m.get("level", "dna")
    frame = f" frame={m['frame']}" if m.get("frame") else ""
    strand = f" strand={'+'if m.get('strand',1)==1 else '-'}" if m.get("strand") else ""
    print(f"  [{status}] {m['token']:30s}  ratio={m['score_ratio']:.4f}  raw={m.get('raw_score','?')}  ({level}{frame}{strand})")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Inspect any low-scoring ground truth examples

# COMMAND ----------

# Find examples that score below 0.8 — these need investigation
low_scorers = results_df[results_df["reward"] < 0.8]
if len(low_scorers) > 0:
    print(f"Found {len(low_scorers)} examples with reward < 0.8:")
    for _, r in low_scorers.iterrows():
        print(f"  plasmid {int(r['plasmid_id'])}: reward={r['reward']:.4f}, qc={r['qc_pass_rate']:.4f}, "
              f"found={int(r['n_found'])}/{int(r['n_hard_tokens'])}")
    
    # Show per-motif detail for the worst one
    worst_idx = low_scorers["reward"].idxmin()
    worst_row = sample.iloc[worst_idx]
    _, worst_details = compute_reward(
        worst_row["prompt"], worst_row["sequence"],
        MOTIF_LOOKUP_FAST, return_details=True,
    )
    print(f"\n  Worst example (plasmid {worst_row['plasmid_id']}):")
    print(f"  Prompt: {worst_row['prompt']}")
    for m in worst_details["per_motif"]:
        status = "PASS" if m["found"] else "FAIL"
        print(f"    [{status}] {m['token']:30s}  ratio={m['score_ratio']:.4f}  ({m.get('level','?')})")
else:
    print("All examples scored >= 0.8 — reward function is working well!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Negative Control: Scrambled Sequences
# MAGIC
# MAGIC Scrambled sequences preserve nucleotide composition but destroy all motif structure.
# MAGIC Should score near 0.
# MAGIC

# COMMAND ----------

import random
random.seed(42)

scrambled_results = []
for idx, row in sample.iterrows():
    seq_list = list(row["sequence"])
    random.shuffle(seq_list)
    scrambled_seq = "".join(seq_list)
    
    reward, details = compute_reward(
        row["prompt"], scrambled_seq, MOTIF_LOOKUP_FAST, return_details=True,
    )
    scrambled_results.append({
        "plasmid_id": row["plasmid_id"],
        "reward": details["reward"],
        "qc_pass_rate": details["qc_pass_rate"],
    })

scrambled_df = pd.DataFrame(scrambled_results)
print("=== Negative Control (Scrambled) ===")
print(f"Mean reward:       {scrambled_df['reward'].mean():.4f}")
print(f"Mean QC pass rate: {scrambled_df['qc_pass_rate'].mean():.4f}")
print()
print("=== Positive Control (Ground Truth) ===")
print(f"Mean reward:       {results_df['reward'].mean():.4f}")
print(f"Mean QC pass rate: {results_df['qc_pass_rate'].mean():.4f}")
print()
gap = results_df['reward'].mean() - scrambled_df['reward'].mean()
print(f"Discrimination gap: {gap:.4f}")
if gap > 0.6:
    print("  -> Strong discrimination. Reward function working well.")
elif gap > 0.4:
    print("  -> Good discrimination. Should work for RL training.")
else:
    print("  -> Weak discrimination. Investigate the motif lookup / alignment.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Benchmark: Speed at Scale
# MAGIC

# COMMAND ----------

print("=== Per-Example Timing ===")
print(f"{'plasmid':>10s}  {'tokens':>6s}  {'seq_len':>8s}  {'time':>8s}  {'ms/tok':>8s}")
print("-" * 48)
for _, r in results_df.iterrows():
    ms_per_tok = r['time_sec'] * 1000 / max(r['n_hard_tokens'], 1)
    print(f"{int(r['plasmid_id']):>10d}  {int(r['n_hard_tokens']):>6d}  {int(r['seq_len']):>8d}  {r['time_sec']:>7.3f}s  {ms_per_tok:>6.1f}ms")

print(f"\nAvg per example: {results_df['time_sec'].mean()*1000:.0f}ms")
print(f"Avg per motif:   {results_df['time_sec'].sum()*1000 / max(results_df['n_hard_tokens'].sum(), 1):.0f}ms")

avg_time = results_df['time_sec'].mean()
for batch_size in [8, 16, 32]:
    for n_comp in [4, 8]:
        t = avg_time * batch_size * n_comp
        print(f"  batch={batch_size:>2d} x completions={n_comp}: {t:.1f}s/step, {t*1000/3600:.1f}h for 1k steps")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Motif Lookups
# MAGIC

# COMMAND ----------

# Save full lookup
motif_path = "/dbfs/FileStore/plasmidgpt/motif_lookup.json"
with open(motif_path, "w") as f:
    json.dump(MOTIF_LOOKUP, f, indent=2)
print(f"Saved MOTIF_LOOKUP ({len(MOTIF_LOOKUP)} tokens) to {motif_path}")

# Save fast lookup
fast_path = "/dbfs/FileStore/plasmidgpt/motif_lookup_fast.json"
with open(fast_path, "w") as f:
    json.dump(MOTIF_LOOKUP_FAST, f, indent=2)
print(f"Saved MOTIF_LOOKUP_FAST ({len(MOTIF_LOOKUP_FAST)} tokens) to {fast_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Scoring:** `score_ratio = sw_score / self_alignment_score` per motif, averaged across all hard tokens.
# MAGIC
# MAGIC | Category | Alignment | Why |
# MAGIC |----------|-----------|-----|
# MAGIC | AMR, REPORTER, TAG | DNA + Protein (max) | CDS features — synonymous mutations preserve function |
# MAGIC | ORI, PROM, ELEM | DNA only | Non-coding — exact DNA sequence matters |
# MAGIC
# MAGIC **API:**
# MAGIC ```python
# MAGIC # Single example
# MAGIC reward, details = compute_reward(prompt, sequence, MOTIF_LOOKUP_FAST, return_details=True)
# MAGIC
# MAGIC # Batch for RL
# MAGIC rewards = plasmid_reward_fn(prompts, completions, MOTIF_LOOKUP_FAST)
# MAGIC ```
# MAGIC
# MAGIC **Key fixes over v1:**
# MAGIC - Properly handles `seq_type='protein'` entries in motif registry (no longer tries to translate proteins as DNA)
# MAGIC - Score-ratio metric instead of CIGAR parsing (simpler, more reliable)
# MAGIC - DNA+protein dual scoring for CDS (ground truth always scores high via DNA; RL sequences get credit for synonymous changes via protein)
# MAGIC

# COMMAND ----------

# Re-score with FULL lookup (all variants) instead of FAST (longest only)
results_full = []
for idx, row in sample.iterrows():
    t0 = time.time()
    reward, details = compute_reward(
        row["prompt"], row["sequence"], MOTIF_LOOKUP, return_details=True,
    )
    elapsed = time.time() - t0
    
    results_full.append({
        "plasmid_id": row["plasmid_id"],
        "n_hard_tokens": details["n_hard_tokens"],
        "n_found": details["n_found"],
        "qc_pass_rate": details["qc_pass_rate"],
        "reward": details["reward"],
        "seq_len": row["sequence_length"],
        "time_sec": round(elapsed, 3),
    })

full_df = pd.DataFrame(results_full)
print("=== FAST lookup (longest variant only) ===")
print(f"Mean reward:       {results_df['reward'].mean():.4f}")
print(f"Mean QC pass rate: {results_df['qc_pass_rate'].mean():.4f}")
print(f"Mean time:         {results_df['time_sec'].mean()*1000:.0f}ms")
print()
print("=== FULL lookup (all variants, pick best) ===")
print(f"Mean reward:       {full_df['reward'].mean():.4f}")
print(f"Mean QC pass rate: {full_df['qc_pass_rate'].mean():.4f}")
print(f"Mean time:         {full_df['time_sec'].mean()*1000:.0f}ms")
print()

# Show per-example comparison
print(f"{'plasmid':>10s}  {'FAST':>8s}  {'FULL':>8s}  {'delta':>8s}")
print("-" * 40)
for (_, rf), (_, rfu) in zip(results_df.iterrows(), full_df.iterrows()):
    delta = rfu['reward'] - rf['reward']
    flag = " <<<" if delta > 0.1 else ""
    print(f"{int(rf['plasmid_id']):>10d}  {rf['reward']:>8.4f}  {rfu['reward']:>8.4f}  {delta:>+8.4f}{flag}")

# COMMAND ----------

# Aggregate per-TOKEN pass/fail rates across all 20 examples
from collections import defaultdict

token_stats = defaultdict(lambda: {"attempts": 0, "passes": 0, "total_ratio": 0.0, "failures": []})

for idx, row in sample.iterrows():
    _, details = compute_reward(
        row["prompt"], row["sequence"], MOTIF_LOOKUP, return_details=True,
    )
    for m in details["per_motif"]:
        tok = m["token"]
        token_stats[tok]["attempts"] += 1
        token_stats[tok]["total_ratio"] += m["score_ratio"]
        if m["found"]:
            token_stats[tok]["passes"] += 1
        else:
            token_stats[tok]["failures"].append({
                "plasmid": row["plasmid_id"],
                "ratio": m["score_ratio"],
                "level": m.get("level", "?"),
            })

print(f"{'token':35s}  {'attempts':>4s}  {'pass%':>6s}  {'avg_ratio':>9s}  {'n_fail':>6s}")
print("-" * 70)
for tok, s in sorted(token_stats.items(), key=lambda x: x[1]["total_ratio"]/max(x[1]["attempts"],1)):
    avg = s["total_ratio"] / max(s["attempts"], 1)
    pct = s["passes"] / max(s["attempts"], 1) * 100
    print(f"{tok:35s}  {s['attempts']:>4d}  {pct:>5.0f}%  {avg:>9.4f}  {len(s['failures']):>6d}")

# Show the worst tokens in detail
print("\n=== Tokens with avg ratio < 0.5 ===")
for tok, s in sorted(token_stats.items(), key=lambda x: x[1]["total_ratio"]/max(x[1]["attempts"],1)):
    avg = s["total_ratio"] / max(s["attempts"], 1)
    if avg >= 0.5:
        continue
    n_variants = len(MOTIF_LOOKUP[tok]["variants"]) if tok in MOTIF_LOOKUP else 0
    print(f"\n{tok} (avg_ratio={avg:.4f}, {n_variants} variants in registry):")
    for f in s["failures"][:3]:
        print(f"  plasmid {f['plasmid']}: ratio={f['ratio']:.4f} ({f['level']})")

# COMMAND ----------

