#!/usr/bin/env python3
"""Evaluate PlasmidLLM models using pLannotate biological annotation.

Generates sequences from validation prompts using vLLM, annotates them with
pLannotate (the biological ground truth), and scores how well requested
functional elements appear in the generated sequences.

Usage:
    python scripts/eval_plannotate.py \
        --models McClain/PlasmidLM-kmer6-MoE McClain/PlasmidLM-kmer6 \
        --n 100 --best-of 3 --output-dir eval_output
"""

from __future__ import annotations

import argparse
import gc
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_motif_registry import ALL_TOKEN_MAPS, feature_to_category_token

# Inline from post_training/reward.py to avoid parasail dependency
HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}
EXCLUDE_TOKENS = {"<ELEM_IRES>", "<ELEM_TRACRRNA>"}

# ── Constants ────────────────────────────────────────────────────────────────

PLANNOTATE_BIN = "/opt/dlami/nvme/miniconda3/envs/plannotate/bin/plannotate"
DEFAULT_PARQUET = "/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"
DEFAULT_MOTIF_REGISTRY = "data/motif_registry.parquet"

MIN_PERCMATCH = 95.0


# ── Inlined from reward.py (avoids parasail dependency) ─────────────────────

def load_motif_lookup(path: str) -> pd.DataFrame:
    """Load motif registry and index by token (lightweight, no alignment)."""
    df = pd.read_parquet(path)
    df = df.set_index("token", drop=False)
    df.index.name = "token_idx"
    return df


def parse_hard_tokens(prompt: str, lookup_df: pd.DataFrame) -> list[str]:
    """Extract hard tokens from prompt that exist in the lookup."""
    all_tokens = re.findall(r"<[^>]+>", prompt)
    known = set(lookup_df.index.unique())
    hard = []
    for t in all_tokens:
        inner = t.strip("<>")
        if any(inner.startswith(p) for p in HARD_PREFIXES):
            if t in known and t not in EXCLUDE_TOKENS:
                hard.append(t)
    return hard


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_dna(text: str) -> str:
    """Strip special tokens and non-DNA characters from generated text."""
    seq = re.sub(r"<[^>]+>", "", text.upper())
    seq = re.sub(r"[^ATGCN]", "", seq)
    return seq


def has_eos(text: str) -> bool:
    return "<EOS>" in text or "</s>" in text


def build_sseqid_to_token(registry_df: pd.DataFrame) -> dict[str, str]:
    """Build sseqid -> token lookup from motif registry for Tier 1 matching."""
    lookup = {}
    for _, row in registry_df.iterrows():
        sseqid = row.get("sseqid")
        token = row.get("token")
        if pd.notna(sseqid) and pd.notna(token):
            lookup[str(sseqid)] = str(token)
    return lookup


def map_annotation_to_token(
    feature: str,
    sseqid: str,
    sseqid_lookup: dict[str, str],
) -> str | None:
    """Two-tier mapping: sseqid (Tier 1) then feature name (Tier 2)."""
    # Tier 1: direct sseqid match from registry
    if sseqid in sseqid_lookup:
        return sseqid_lookup[sseqid]

    # Tier 2: substring matching on feature name
    result = feature_to_category_token(feature)
    if result is not None:
        return result[0]  # token string

    return None


def load_val_prompts(
    parquet_path: str,
    n: int,
    val_split: float = 0.05,
    seed: int = 42,
) -> list[str]:
    """Load and sample validation prompts, reproducing the val split."""
    table = pq.read_table(parquet_path)
    col = "prompt" if "prompt" in table.column_names else "token_prompt"
    all_prompts = table.column(col).to_pylist()

    n_total = len(all_prompts)
    n_val = int(n_total * val_split)

    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    val_indices = indices[:n_val]

    # Sample n prompts from val set
    torch.manual_seed(seed)
    sample_idx = torch.randperm(len(val_indices))[:n].tolist()
    selected = [val_indices[i] for i in sample_idx]

    return [all_prompts[i] for i in selected]


# ── MoE dense patch ──────────────────────────────────────────────────────────

def patch_moe_dense(model):
    """Monkey-patch MoE layers for dense batched-matmul inference.

    Runs all experts on all tokens via bmm, then zeros out unused experts
    with the router mask. Faster than the sparse dispatch loop.
    """
    patched = 0
    for layer in model.model.layers:
        if not hasattr(layer, "moe"):
            continue
        moe = layer.moe

        up_w = torch.stack([e.up_proj.weight.T for e in moe.experts]).contiguous()
        down_w = torch.stack([e.down_proj.weight.T for e in moe.experts]).contiguous()
        moe.register_buffer("_up_w", up_w, persistent=False)
        moe.register_buffer("_down_w", down_w, persistent=False)

        def _make_dense_forward(_moe):
            def forward_dense(hidden_states):
                batch, seq_len, hidden = hidden_states.shape
                flat = hidden_states.view(-1, hidden)
                num_tokens = flat.shape[0]

                router_logits = _moe.router(flat)
                router_probs = F.softmax(router_logits, dim=-1)
                top_weights, top_indices = torch.topk(router_probs, _moe.top_k, dim=-1)
                top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

                expert_mask = torch.zeros(
                    num_tokens, _moe.num_experts,
                    device=flat.device, dtype=flat.dtype,
                )
                expert_mask.scatter_(1, top_indices, top_weights)

                x = flat.unsqueeze(0).expand(_moe.num_experts, -1, -1)
                h = torch.bmm(x, _moe._up_w)
                h = F.gelu(h)
                expert_out = torch.bmm(h, _moe._down_w)

                output = torch.einsum("enh,ne->nh", expert_out, expert_mask)
                output = output.view(batch, seq_len, hidden)

                with torch.no_grad():
                    one_hot = torch.zeros(num_tokens, _moe.num_experts, device=flat.device)
                    one_hot.scatter_(1, top_indices, 1.0)
                    f_i = one_hot.sum(dim=0) / (num_tokens * _moe.top_k)
                P = router_probs.mean(dim=0)
                aux_loss = _moe.num_experts * (f_i * P).sum()

                return output, aux_loss
            return forward_dense

        moe._original_forward = moe.forward
        moe.forward = _make_dense_forward(moe)
        patched += 1

    print(f"  Patched {patched} MoE layers for dense dispatch")
    return model


def is_moe_model(model_id: str) -> bool:
    """Check if a model is MoE by inspecting its config."""
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return getattr(config, "use_moe", False)
    except Exception:
        return "moe" in model_id.lower()


def generate_with_hf(
    model_id: str,
    prompts: list[str],
    best_of: int = 3,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    seed: int = 42,
    batch_size: int = 16,
) -> tuple[list[list[str]], float]:
    """Generate sequences using HF generate() with batching. Applies MoE dense patch if needed."""
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    moe = is_moe_model(model_id)
    label = "MoE dense" if moe else "dense"
    print(f"  Loading HF model ({label}): {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if moe:
        patch_moe_dense(model)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"

    torch.manual_seed(seed)

    # Build flat list: each prompt repeated best_of times
    formatted = [p + "<SEP>" for p in prompts]
    all_prompts = []
    for p in formatted:
        all_prompts.extend([p] * best_of)

    total_seqs = len(all_prompts)
    all_texts = []
    total_tokens = 0
    t0 = time.time()

    for start in tqdm(range(0, total_seqs, batch_size), desc="  Generating", unit="batch"):
        batch_prompts = all_prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        for seq_ids in output_ids:
            gen_ids = seq_ids[prompt_len:]
            # Strip padding from generated portion
            gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
            total_tokens += len(gen_ids)
            all_texts.append(tokenizer.decode(gen_ids.tolist()))

    elapsed = time.time() - t0
    tps = total_tokens / elapsed if elapsed > 0 else 0
    print(f"  Done: {total_tokens} tokens in {elapsed:.1f}s ({tps:.0f} tok/s)")

    # Group back into candidates_per_prompt
    candidates_per_prompt = []
    for i in range(len(prompts)):
        candidates_per_prompt.append(all_texts[i * best_of : (i + 1) * best_of])

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return candidates_per_prompt, tps


# ── pLannotate ───────────────────────────────────────────────────────────────

def write_fasta(sequences: dict[str, str], path: Path) -> None:
    """Write {name: sequence} dict as FASTA."""
    with open(path, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")


def _annotate_one(args_tuple) -> pd.DataFrame | None:
    """Annotate a single sequence with pLannotate. Used by thread pool."""
    name, seq, output_dir, plannotate_bin = args_tuple
    seq_fasta = output_dir / f"{name}.fasta"
    seq_out = output_dir / name
    seq_out.mkdir(parents=True, exist_ok=True)

    with open(seq_fasta, "w") as f:
        f.write(f">{name}\n{seq}\n")

    cmd = [plannotate_bin, "batch", "-i", str(seq_fasta), "-o", str(seq_out), "-c", "-x"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    csv_files = list(seq_out.glob("*.csv"))
    if not csv_files:
        return None

    csv_path = max(csv_files, key=lambda p: p.stat().st_size)
    try:
        df = pd.read_csv(csv_path)
        if not df.empty:
            df["sample_id"] = name
            return df
    except Exception:
        pass
    return None


def run_plannotate(
    fasta_seqs: dict[str, str],
    output_dir: Path,
    plannotate_bin: str = PLANNOTATE_BIN,
    n_workers: int = 8,
) -> pd.DataFrame | None:
    """Run pLannotate on each sequence in parallel and combine results."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(name, seq, output_dir, plannotate_bin) for name, seq in fasta_seqs.items()]

    all_frames = []
    n_fail = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_annotate_one, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  pLannotate", unit="seq"):
            result = fut.result()
            if result is not None:
                all_frames.append(result)
            else:
                n_fail += 1

    print(f"  pLannotate: {len(all_frames)} succeeded, {n_fail} failed out of {len(fasta_seqs)}")

    if not all_frames:
        return None

    return pd.concat(all_frames, ignore_index=True)


def parse_plannotate_results(
    ann_df: pd.DataFrame,
    sseqid_lookup: dict[str, str],
) -> dict[str, list[str]]:
    """Parse pLannotate CSV into {sample_id: [matched_tokens]}.

    Applies quality filters and two-tier token mapping.
    """
    results: dict[str, list[str]] = defaultdict(list)

    # Identify column names (pLannotate output varies)
    id_col = None
    for candidate in ["sample_id", "qseqid", "plasmid_id", "Plasmid"]:
        if candidate in ann_df.columns:
            id_col = candidate
            break
    if id_col is None:
        print(f"  Warning: no ID column found in pLannotate output. Columns: {list(ann_df.columns)}")
        return results

    feat_col = "Feature" if "Feature" in ann_df.columns else None
    sseq_col = "sseqid" if "sseqid" in ann_df.columns else None
    perc_col = "percmatch" if "percmatch" in ann_df.columns else None
    frag_col = "fragment" if "fragment" in ann_df.columns else None

    if feat_col is None:
        print(f"  Warning: no Feature column. Columns: {list(ann_df.columns)}")
        return results

    for _, row in ann_df.iterrows():
        # Quality filter
        if perc_col and pd.notna(row.get(perc_col)) and row[perc_col] < MIN_PERCMATCH:
            continue
        if frag_col and pd.notna(row.get(frag_col)) and bool(row[frag_col]):
            continue

        sample_id = str(row[id_col])
        feature = str(row[feat_col]) if pd.notna(row.get(feat_col)) else ""
        sseqid = str(row[sseq_col]) if sseq_col and pd.notna(row.get(sseq_col)) else ""

        token = map_annotation_to_token(feature, sseqid, sseqid_lookup)
        if token is not None:
            results[sample_id].append(token)

    return dict(results)


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_prompts(
    prompts: list[str],
    candidates_per_prompt: list[list[str]],
    plannotate_tokens: dict[str, list[str]],
    lookup_df: pd.DataFrame,
) -> list[dict]:
    """Score each prompt using pLannotate annotations. Best-of selection by hit count."""
    results = []

    for i, (prompt, candidates) in enumerate(zip(prompts, candidates_per_prompt)):
        hard_tokens = parse_hard_tokens(prompt + "<SEP>", lookup_df)
        requested = set(hard_tokens)

        best_result = None
        best_hits = -1
        best_total_annotations = -1

        for j, cand_text in enumerate(candidates):
            sample_key = f"sample_{i}_cand_{j}"
            dna = extract_dna(cand_text)

            found_tokens = set(plannotate_tokens.get(sample_key, []))
            hits = len(requested & found_tokens)
            total_ann = len(plannotate_tokens.get(sample_key, []))

            # Best-of: highest hit count, then most total annotations
            if hits > best_hits or (hits == best_hits and total_ann > best_total_annotations):
                best_hits = hits
                best_total_annotations = total_ann
                best_result = {
                    "prompt_idx": i,
                    "prompt": prompt,
                    "candidate_idx": j,
                    "dna_len": len(dna),
                    "has_eos": has_eos(cand_text),
                    "requested_tokens": sorted(requested),
                    "n_requested": len(requested),
                    "found_tokens": sorted(requested & found_tokens),
                    "n_found": hits,
                    "extra_tokens": sorted(found_tokens - requested),
                    "hit_rate": hits / len(requested) if requested else 0.0,
                    "all_annotations": sorted(found_tokens),
                    "n_total_annotations": total_ann,
                }

        if best_result is not None:
            results.append(best_result)

    return results


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(
    model_results: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate markdown evaluation report."""
    lines = []
    lines.append("# PlasmidLLM pLannotate Evaluation Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Hit Rate | Avg Seq Len | EOS Rate | TPS | N Prompts |")
    lines.append("|-------|----------|-------------|----------|-----|-----------|")

    for model_id, data in model_results.items():
        scores = data["scores"]
        tps = data["tps"]
        name = model_id.split("/")[-1]

        hit_rates = [s["hit_rate"] for s in scores]
        seq_lens = [s["dna_len"] for s in scores]
        eos_rates = [s["has_eos"] for s in scores]

        avg_hr = np.mean(hit_rates) if hit_rates else 0
        avg_len = np.mean(seq_lens) if seq_lens else 0
        eos_pct = np.mean(eos_rates) if eos_rates else 0

        lines.append(
            f"| {name} | {avg_hr:.1%} | {avg_len:.0f} | {eos_pct:.1%} | {tps:.0f} | {len(scores)} |"
        )

    # Per-category breakdown
    lines.append("\n## Per-Category Breakdown\n")

    for model_id, data in model_results.items():
        name = model_id.split("/")[-1]
        lines.append(f"### {name}\n")
        lines.append("| Category | Requested | Found | Hit Rate |")
        lines.append("|----------|-----------|-------|----------|")

        cat_stats: dict[str, dict] = defaultdict(lambda: {"requested": 0, "found": 0})
        for s in data["scores"]:
            for token in s["requested_tokens"]:
                inner = token.strip("<>")
                cat = inner.split("_")[0] if "_" in inner else "OTHER"
                cat_stats[cat]["requested"] += 1
                if token in s["found_tokens"]:
                    cat_stats[cat]["found"] += 1

        total_req = 0
        total_found = 0
        for cat in sorted(cat_stats.keys()):
            cs = cat_stats[cat]
            total_req += cs["requested"]
            total_found += cs["found"]
            rate = cs["found"] / cs["requested"] if cs["requested"] > 0 else 0
            lines.append(f"| {cat} | {cs['requested']} | {cs['found']} | {rate:.1%} |")

        total_rate = total_found / total_req if total_req > 0 else 0
        lines.append(f"| **TOTAL** | **{total_req}** | **{total_found}** | **{total_rate:.1%}** |")
        lines.append("")

    # Per-prompt details (first 20)
    lines.append("## Per-Prompt Details (first 20)\n")

    for model_id, data in model_results.items():
        name = model_id.split("/")[-1]
        lines.append(f"### {name}\n")

        for s in data["scores"][:20]:
            status = "PERFECT" if s["hit_rate"] == 1.0 else f"{s['hit_rate']:.0%}"
            lines.append(
                f"- **Prompt {s['prompt_idx']}** [{status}]: "
                f"{s['n_found']}/{s['n_requested']} hits, "
                f"len={s['dna_len']}, eos={s['has_eos']}, "
                f"cand={s['candidate_idx']}"
            )
            if s["requested_tokens"]:
                found_set = set(s["found_tokens"])
                tokens_display = []
                for t in s["requested_tokens"]:
                    mark = "+" if t in found_set else "-"
                    tokens_display.append(f"{mark}{t}")
                lines.append(f"  - {' '.join(tokens_display)}")
            if s["extra_tokens"]:
                lines.append(f"  - Extra: {' '.join(s['extra_tokens'])}")

        lines.append("")

    # Probe test results
    has_probes = any("probe_results" in data for data in model_results.values())
    if has_probes:
        lines.append("## Probe Tests (Single-Token)\n")

        for model_id, data in model_results.items():
            probes = data.get("probe_results", [])
            if not probes:
                continue
            name = model_id.split("/")[-1]
            lines.append(f"### {name}\n")
            lines.append("| Token | Seq Len | Found | Annotations |")
            lines.append("|-------|---------|-------|-------------|")

            for p in probes:
                status = "YES" if p["hit"] else "NO"
                ann_str = ", ".join(p["found_tokens"]) if p["found_tokens"] else "(none)"
                lines.append(f"| {p['token']} | {p['dna_len']} | {status} | {ann_str} |")

            n_hit = sum(1 for p in probes if p["hit"])
            lines.append(f"\n**Probe hit rate: {n_hit}/{len(probes)} ({n_hit/max(len(probes),1):.0%})**\n")

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")


def print_summary(model_results: dict[str, dict]) -> None:
    """Print condensed summary to stdout."""
    print("\n" + "=" * 70)
    print("PLANNOTATE EVALUATION SUMMARY")
    print("=" * 70)

    for model_id, data in model_results.items():
        name = model_id.split("/")[-1]
        scores = data["scores"]
        tps = data["tps"]

        if not scores:
            print(f"\n  {name}: no scores (pLannotate may have failed)")
            print(f"    Speed:     {tps:.0f} tok/s")
            continue

        hit_rates = [s["hit_rate"] for s in scores]
        seq_lens = [s["dna_len"] for s in scores]
        eos_count = sum(1 for s in scores if s["has_eos"])
        total_req = sum(s["n_requested"] for s in scores)
        total_found = sum(s["n_found"] for s in scores)
        perfect = sum(1 for s in scores if s["hit_rate"] == 1.0)

        print(f"\n  {name}")
        print(f"    Hit rate:  {np.mean(hit_rates):.1%} (mean)  |  {total_found}/{total_req} components")
        print(f"    Perfect:   {perfect}/{len(scores)} prompts ({perfect/len(scores):.0%})")
        print(f"    Seq len:   {np.mean(seq_lens):.0f} (mean)  {np.median(seq_lens):.0f} (median)")
        print(f"    EOS rate:  {eos_count}/{len(scores)} ({eos_count/len(scores):.0%})")
        print(f"    Speed:     {tps:.0f} tok/s")

    print("\n" + "=" * 70)


# ── Probe tests ──────────────────────────────────────────────────────────────

DEFAULT_PROBE_TOKENS = [
    "AMR_AMPICILLIN", "AMR_KANAMYCIN", "AMR_CHLORAMPHENICOL",
    "ORI_COLE1", "ORI_F1",
    "PROM_CMV", "PROM_T7", "PROM_LAC",
    "REPORTER_EGFP", "REPORTER_MCHERRY",
    "ELEM_WPRE", "ELEM_POLYA_BGH",
    "TAG_HIS", "TAG_FLAG",
]


def run_probe_tests(
    models: list[str],
    probe_tokens: list[str],
    sseqid_lookup: dict[str, str],
    output_dir: Path,
    plannotate_bin: str = PLANNOTATE_BIN,
    max_tokens: int = 8000,
    temperature: float = 0.7,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Run targeted single-token tests: generate with one token, check if pLannotate finds it."""
    if not probe_tokens:
        probe_tokens = DEFAULT_PROBE_TOKENS

    # Normalize: strip <> if provided
    probe_tokens = [t.strip("<>") for t in probe_tokens]
    prompts = [f"<BOS> <{tok}>" for tok in probe_tokens]

    print(f"\n{'='*70}")
    print(f"PROBE TESTS: {len(probe_tokens)} single-token prompts")
    print(f"{'='*70}")

    results: dict[str, list[dict]] = {}

    for model_id in models:
        model_name = model_id.split("/")[-1]
        print(f"\n  Model: {model_name}")

        probe_dir = output_dir / model_name / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)

        # Generate one candidate per probe prompt
        candidates_per_prompt, _ = generate_with_hf(
            model_id=model_id,
            prompts=prompts,
            best_of=1,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )

        # Build sequences
        fasta_seqs = {}
        for i, candidates in enumerate(candidates_per_prompt):
            dna = extract_dna(candidates[0])
            if len(dna) >= 100:
                fasta_seqs[f"probe_{probe_tokens[i]}"] = dna

        # Run pLannotate
        ann_df = run_plannotate(fasta_seqs, probe_dir, plannotate_bin)

        plannotate_tokens = {}
        if ann_df is not None and not ann_df.empty:
            plannotate_tokens = parse_plannotate_results(ann_df, sseqid_lookup)

        # Check each probe
        model_probe_results = []
        print(f"\n  {'Token':<30s} {'SeqLen':>7s} {'Found?':>7s} {'Annotations'}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*30}")

        for i, tok in enumerate(probe_tokens):
            target_token = f"<{tok}>"
            sample_key = f"probe_{tok}"
            dna = extract_dna(candidates_per_prompt[i][0]) if i < len(candidates_per_prompt) else ""
            found_tokens = set(plannotate_tokens.get(sample_key, []))
            hit = target_token in found_tokens
            all_ann = sorted(found_tokens)

            result = {
                "token": tok,
                "target": target_token,
                "dna_len": len(dna),
                "hit": hit,
                "found_tokens": all_ann,
            }
            model_probe_results.append(result)

            status = "YES" if hit else "NO"
            print(f"  {tok:<30s} {len(dna):>7d} {status:>7s} {', '.join(all_ann) if all_ann else '(none)'}")

        n_hit = sum(1 for r in model_probe_results if r["hit"])
        print(f"\n  Probe hit rate: {n_hit}/{len(model_probe_results)} ({n_hit/max(len(model_probe_results),1):.0%})")
        results[model_id] = model_probe_results

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PlasmidLLM models with pLannotate annotation scoring"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="HuggingFace model IDs (e.g. McClain/PlasmidLM-kmer6)",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of val prompts")
    parser.add_argument("--best-of", type=int, default=3, help="Candidates per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max kmer tokens (~9kb DNA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="eval_output")
    parser.add_argument(
        "--parquet", type=str, default=DEFAULT_PARQUET,
        help="Training data parquet for val split",
    )
    parser.add_argument(
        "--motif-registry", type=str, default=DEFAULT_MOTIF_REGISTRY,
        help="Path to motif_registry.parquet",
    )
    parser.add_argument(
        "--plannotate-bin", type=str, default=PLANNOTATE_BIN,
        help="Path to plannotate binary in conda env",
    )
    parser.add_argument(
        "--probe-tokens", nargs="*", default=None,
        help="Run targeted single-token probe tests (e.g. AMR_AMPICILLIN ORI_COLE1). "
             "If flag given with no args, uses a default set.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load motif registry ──
    print(f"Loading motif registry: {args.motif_registry}")
    registry_df = pd.read_parquet(args.motif_registry)
    sseqid_lookup = build_sseqid_to_token(registry_df)
    print(f"  {len(registry_df)} entries, {len(sseqid_lookup)} unique sseqids")

    # Also load for parse_hard_tokens (needs the indexed format)
    lookup_df = load_motif_lookup(args.motif_registry)

    # ── Load validation prompts ──
    print(f"Loading validation prompts: {args.parquet}")
    prompts = load_val_prompts(args.parquet, args.n, seed=args.seed)
    print(f"  Selected {len(prompts)} prompts")

    # ── Per-model evaluation ──
    model_results: dict[str, dict] = {}

    for model_id in args.models:
        model_name = model_id.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate sequences (HF generate, with MoE dense patch if needed)
        candidates_per_prompt, tps = generate_with_hf(
            model_id=model_id,
            prompts=prompts,
            best_of=args.best_of,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

        # Step 2: Build FASTA sequences
        fasta_seqs = {}
        for i, candidates in enumerate(candidates_per_prompt):
            for j, cand_text in enumerate(candidates):
                dna = extract_dna(cand_text)
                if len(dna) >= 100:
                    fasta_seqs[f"sample_{i}_cand_{j}"] = dna

        # Also write combined FASTA for reference
        fasta_path = model_dir / "generated.fasta"
        write_fasta(fasta_seqs, fasta_path)
        print(f"  {len(fasta_seqs)} sequences to annotate")

        # Step 3: Run pLannotate (one sequence at a time)
        plannotate_dir = model_dir / "plannotate"
        ann_df = run_plannotate(fasta_seqs, plannotate_dir, args.plannotate_bin)

        if ann_df is None or ann_df.empty:
            print("  WARNING: No pLannotate annotations — skipping scoring")
            model_results[model_id] = {"scores": [], "tps": tps}
            continue

        print(f"  pLannotate found {len(ann_df)} annotations")

        # Step 4: Map annotations to tokens
        plannotate_tokens = parse_plannotate_results(ann_df, sseqid_lookup)
        n_mapped = sum(len(v) for v in plannotate_tokens.values())
        print(f"  Mapped {n_mapped} annotations to tokens across {len(plannotate_tokens)} sequences")

        # Step 5: Score
        scores = score_prompts(prompts, candidates_per_prompt, plannotate_tokens, lookup_df)
        model_results[model_id] = {"scores": scores, "tps": tps}

    # ── Targeted probe tests ──
    if args.probe_tokens is not None:
        probe_results = run_probe_tests(
            models=args.models,
            probe_tokens=args.probe_tokens,
            sseqid_lookup=sseqid_lookup,
            output_dir=output_dir,
            plannotate_bin=args.plannotate_bin,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        # Attach probe results to model_results for report
        for model_id in args.models:
            if model_id in probe_results:
                model_results[model_id]["probe_results"] = probe_results[model_id]

    # ── Generate report ──
    report_path = output_dir / "eval_report.md"
    generate_report(model_results, report_path)
    print_summary(model_results)


if __name__ == "__main__":
    main()
