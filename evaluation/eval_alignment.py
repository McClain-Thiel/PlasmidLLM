#!/usr/bin/env python3
"""Evaluate PlasmidLLM models using Smith-Waterman alignment against motif registry.

Uses parasail (SIMD-accelerated) to align generated sequences against all reference
motifs in the registry. Reports true positives, false positives, false negatives,
and sweeps across temperatures.

Usage:
    python scripts/eval_alignment.py \
        --models McClain/PlasmidLM-kmer6 McClain/PlasmidLM-kmer6-MoE \
        --n 50 --temps 0.3 0.5 0.7 0.9 1.0 --output-dir eval_alignment
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import parasail
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from Bio.Seq import Seq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ────────────────────────────────────────────────────────────────

HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}
EXCLUDE_TOKENS = {"<ELEM_IRES>", "<ELEM_TRACRRNA>"}

DEFAULT_PARQUET = "/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"
DEFAULT_REGISTRY = "data/motif_registry_combined.parquet"

# Alignment params (same as reward.py)
DNA_MATRIX = parasail.matrix_create("ACGT", 1, -1)
PROTEIN_MATRIX = parasail.blosum62
DNA_OPEN, DNA_EXTEND = 5, 1
PROTEIN_OPEN, PROTEIN_EXTEND = 10, 1

CDS_PREFIXES = {"AMR", "REPORTER", "TAG"}
MIN_SCORE_RATIO = 0.70  # default cutoff for "found"


# ── Registry loading ─────────────────────────────────────────────────────────

def load_registry(path: str) -> pd.DataFrame:
    """Load motif registry with precomputed max scores for normalization."""
    df = pd.read_parquet(path)

    # Only keep rows with sequences
    df = df[df["sequence"].notna()].copy()

    # Derive is_cds: true if category is "CDS" or token prefix indicates protein-coding
    if "is_cds" not in df.columns:
        df["is_cds"] = df["category"].eq("CDS") | df["token"].apply(
            lambda t: t.strip("<>").split("_")[0] in CDS_PREFIXES
        )

    # Split sequence into dna_seq / protein_seq
    df["dna_seq"] = df.apply(
        lambda r: r["sequence"] if r.get("seq_type") != "protein" else None, axis=1
    )
    df["protein_seq"] = df.apply(
        lambda r: r["sequence"] if r.get("seq_type") == "protein" else None, axis=1
    )

    # For CDS with DNA, also compute protein translation
    mask = df["is_cds"] & df["dna_seq"].notna() & (df["seq_type"] != "protein")
    df.loc[mask, "protein_seq"] = df.loc[mask, "dna_seq"].apply(_safe_translate)

    # Precompute self-alignment scores
    df["dna_max_score"] = df["dna_seq"].apply(
        lambda s: _self_score(s, protein=False) if pd.notna(s) else 1
    )
    df["protein_max_score"] = df["protein_seq"].apply(
        lambda s: _self_score(s, protein=True) if pd.notna(s) else 1
    )

    # Index by token
    df = df.set_index("token", drop=False)
    df.index.name = "token_idx"

    return df


def _safe_translate(dna: str) -> str | None:
    if not dna or len(dna) < 3:
        return None
    try:
        trimmed = dna[: len(dna) - (len(dna) % 3)]
        return str(Seq(trimmed).translate())
    except Exception:
        return None


def _self_score(seq: str, protein: bool = False) -> int:
    if not seq:
        return 1
    if protein:
        return parasail.sw_striped_16(seq, seq, PROTEIN_OPEN, PROTEIN_EXTEND, PROTEIN_MATRIX).score
    return parasail.sw_striped_16(seq, seq, DNA_OPEN, DNA_EXTEND, DNA_MATRIX).score


# ── Alignment ────────────────────────────────────────────────────────────────

def align_dna(motif_dna: str, candidate: str, max_score: int) -> float:
    """Smith-Waterman on both strands, returns score_ratio in [0,1]."""
    if not motif_dna or not candidate:
        return 0.0
    rev = str(Seq(motif_dna).reverse_complement())
    fwd = parasail.sw_striped_16(candidate, motif_dna, DNA_OPEN, DNA_EXTEND, DNA_MATRIX).score
    rev_s = parasail.sw_striped_16(candidate, rev, DNA_OPEN, DNA_EXTEND, DNA_MATRIX).score
    return round(min(max(fwd, rev_s) / max(max_score, 1), 1.0), 4)


def align_protein(motif_prot: str, candidate: str, max_score: int) -> float:
    """6-frame protein Smith-Waterman, returns score_ratio in [0,1]."""
    if not motif_prot or len(candidate) < 3:
        return 0.0
    fwd = candidate.upper()
    rev = str(Seq(fwd).reverse_complement())
    best = 0
    for offset in range(3):
        for seq in [fwd, rev]:
            sub = seq[offset:]
            sub = sub[: len(sub) - (len(sub) % 3)]
            if len(sub) < 3:
                continue
            try:
                prot = str(Seq(sub).translate())
            except Exception:
                continue
            s = parasail.sw_striped_16(prot, motif_prot, PROTEIN_OPEN, PROTEIN_EXTEND, PROTEIN_MATRIX).score
            best = max(best, s)
    return round(min(best / max(max_score, 1), 1.0), 4)


def score_token(token: str, candidate: str, registry_df: pd.DataFrame) -> float:
    """Best alignment score across all registry entries for a token."""
    if token not in registry_df.index:
        return 0.0
    rows = registry_df.loc[[token]]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T

    best = 0.0
    for _, row in rows.iterrows():
        if pd.notna(row.get("dna_seq")):
            best = max(best, align_dna(row["dna_seq"], candidate, int(row["dna_max_score"])))
        if row.get("is_cds") and pd.notna(row.get("protein_seq")):
            best = max(best, align_protein(row["protein_seq"], candidate, int(row["protein_max_score"])))
        if best >= MIN_SCORE_RATIO:
            break  # early exit
    return best


def scan_all_tokens(candidate: str, registry_df: pd.DataFrame, cutoff: float) -> dict[str, float]:
    """Scan candidate against ALL unique tokens in registry, return those above cutoff."""
    found = {}
    for token in registry_df["token"].unique():
        score = score_token(token, candidate, registry_df)
        if score >= cutoff:
            found[token] = score
    return found


# ── Prompt parsing ───────────────────────────────────────────────────────────

def parse_hard_tokens(prompt: str, registry_df: pd.DataFrame) -> list[str]:
    """Extract hard tokens from prompt that exist in registry."""
    all_tokens = re.findall(r"<[^>]+>", prompt)
    known = set(registry_df.index.unique())
    hard = []
    for t in all_tokens:
        inner = t.strip("<>")
        if any(inner.startswith(p) for p in HARD_PREFIXES):
            if t in known and t not in EXCLUDE_TOKENS:
                hard.append(t)
    return hard


def extract_dna(text: str) -> str:
    seq = re.sub(r"<[^>]+>", "", text.upper())
    return re.sub(r"[^ATGCN]", "", seq)


def has_eos(text: str) -> bool:
    return "<EOS>" in text or "</s>" in text


# ── Val prompts ──────────────────────────────────────────────────────────────

def load_val_prompts(parquet_path: str, n: int, seed: int = 42) -> list[str]:
    table = pq.read_table(parquet_path)
    col = "prompt" if "prompt" in table.column_names else "token_prompt"
    all_prompts = table.column(col).to_pylist()
    n_total = len(all_prompts)
    n_val = int(n_total * 0.05)
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    val_indices = indices[:n_val]
    torch.manual_seed(seed)
    sample_idx = torch.randperm(len(val_indices))[:n].tolist()
    return [all_prompts[val_indices[i]] for i in sample_idx]


# ── MoE dense patch ─────────────────────────────────────────────────────────

def patch_moe_dense(model):
    patched = 0
    for layer in model.model.layers:
        if not hasattr(layer, "moe"):
            continue
        moe = layer.moe
        moe.register_buffer("_up_w", torch.stack([e.up_proj.weight.T for e in moe.experts]).contiguous(), persistent=False)
        moe.register_buffer("_down_w", torch.stack([e.down_proj.weight.T for e in moe.experts]).contiguous(), persistent=False)

        def _make(m):
            def fwd(h):
                B, S, D = h.shape
                flat = h.view(-1, D)
                N = flat.shape[0]
                rp = F.softmax(m.router(flat), dim=-1)
                tw, ti = torch.topk(rp, m.top_k, dim=-1)
                tw = tw / tw.sum(-1, keepdim=True)
                mask = torch.zeros(N, m.num_experts, device=flat.device, dtype=flat.dtype)
                mask.scatter_(1, ti, tw)
                x = flat.unsqueeze(0).expand(m.num_experts, -1, -1)
                out = torch.bmm(F.gelu(torch.bmm(x, m._up_w)), m._down_w)
                result = torch.einsum("enh,ne->nh", out, mask).view(B, S, D)
                with torch.no_grad():
                    oh = torch.zeros(N, m.num_experts, device=flat.device)
                    oh.scatter_(1, ti, 1.0)
                return result, m.num_experts * ((oh.sum(0) / (N * m.top_k)) * rp.mean(0)).sum()
            return fwd
        moe.forward = _make(moe)
        patched += 1
    print(f"  Patched {patched} MoE layers for dense dispatch")


def is_moe_model(model_id: str) -> bool:
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return getattr(config, "use_moe", False)
    except Exception:
        return "moe" in model_id.lower()


# ── Generation ───────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """Load model + tokenizer, apply MoE patch if needed. Returns (model, tokenizer, device)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    moe = is_moe_model(model_id)
    print(f"  Loading {'MoE' if moe else 'dense'} model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if moe:
        patch_moe_dense(model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"
    return model, tokenizer, device


def generate_batch(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    best_of: int = 1,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    seed: int = 42,
    batch_size: int = 16,
) -> tuple[list[list[str]], float]:
    """Generate sequences with HF generate(). Returns (candidates_per_prompt, tok/s)."""
    torch.manual_seed(seed)
    formatted = [p + "<SEP>" for p in prompts]
    all_prompts = []
    for p in formatted:
        all_prompts.extend([p] * best_of)

    all_texts = []
    total_tokens = 0
    t0 = time.time()

    for start in tqdm(range(0, len(all_prompts), batch_size), desc="  Gen", unit="batch"):
        batch = all_prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        gen_kwargs = dict(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, do_sample=True, top_k=50)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        plen = encoded["input_ids"].shape[1]
        for ids in out:
            gen = ids[plen:]
            gen = gen[gen != tokenizer.pad_token_id]
            total_tokens += len(gen)
            all_texts.append(tokenizer.decode(gen.tolist()))

    elapsed = time.time() - t0
    tps = total_tokens / elapsed if elapsed > 0 else 0
    print(f"  {total_tokens} tokens in {elapsed:.1f}s ({tps:.0f} tok/s)")

    candidates = []
    for i in range(len(prompts)):
        candidates.append(all_texts[i * best_of:(i + 1) * best_of])

    return candidates, tps


# ── Scoring ──────────────────────────────────────────────────────────────────

def _score_one_sequence(args):
    """Score one (prompt, dna) pair. For use with ProcessPoolExecutor."""
    prompt, dna, registry_path, cutoff, do_fp_scan = args
    registry_df = _score_one_sequence._registry
    requested = set(parse_hard_tokens(prompt + "<SEP>", registry_df))

    # Score requested tokens
    requested_scores = {}
    for token in requested:
        requested_scores[token] = score_token(token, dna, registry_df)

    tp = {t for t, s in requested_scores.items() if s >= cutoff}
    fn = requested - tp

    # Optional: scan for all tokens (false positive detection)
    if do_fp_scan:
        all_found = scan_all_tokens(dna, registry_df, cutoff)
        fp = set(all_found.keys()) - requested
    else:
        all_found = {t: s for t, s in requested_scores.items() if s >= cutoff}
        fp = set()

    return {
        "requested": sorted(requested),
        "n_requested": len(requested),
        "tp": sorted(tp),
        "fn": sorted(fn),
        "fp": sorted(fp),
        "n_tp": len(tp),
        "n_fn": len(fn),
        "n_fp": len(fp),
        "hit_rate": len(tp) / len(requested) if requested else 0.0,
        "precision": len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0,
        "requested_scores": {t: float(s) for t, s in requested_scores.items()},
        "all_found": {t: float(s) for t, s in all_found.items()},
        "dna_len": len(dna),
    }


def _init_scorer(registry_path):
    """Initialize registry in worker process."""
    _score_one_sequence._registry = load_registry(registry_path)


def score_sequences(
    prompts: list[str],
    candidates_per_prompt: list[list[str]],
    registry_path: str,
    cutoff: float,
    n_workers: int = 4,
    fp_scan: bool = False,
) -> list[dict]:
    """Score all prompts, picking best candidate per prompt."""
    # Build (prompt, dna) pairs — flatten best_of candidates
    tasks = []
    task_map = []  # (prompt_idx, cand_idx)
    for i, (prompt, cands) in enumerate(zip(prompts, candidates_per_prompt)):
        for j, text in enumerate(cands):
            dna = extract_dna(text)
            if len(dna) >= 100:
                tasks.append((prompt, dna, registry_path, cutoff, fp_scan))
                task_map.append((i, j, text))

    print(f"  Scoring {len(tasks)} sequences (cutoff={cutoff:.0%})...")

    # Run alignment in parallel
    results_raw = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_scorer, initargs=(registry_path,)) as pool:
        for result in tqdm(pool.map(_score_one_sequence, tasks, chunksize=1),
                           total=len(tasks), desc="  Align", unit="seq"):
            results_raw.append(result)

    # Best-of selection: pick candidate with most TPs per prompt
    prompt_best: dict[int, dict] = {}
    for idx, ((pi, ci, text), result) in enumerate(zip(task_map, results_raw)):
        result["candidate_idx"] = ci
        result["has_eos"] = has_eos(text)

        prev = prompt_best.get(pi)
        if prev is None or result["n_tp"] > prev["n_tp"] or (
            result["n_tp"] == prev["n_tp"] and len(result["all_found"]) > len(prev["all_found"])
        ):
            prompt_best[pi] = result

    return [prompt_best[i] for i in sorted(prompt_best.keys())]


# ── Report ───────────────────────────────────────────────────────────────────

def write_report(all_results: dict, output_path: Path, cutoff: float):
    lines = []
    lines.append("# PlasmidLLM Alignment Evaluation Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Cutoff: {cutoff:.0%} score ratio\n")

    # Summary table across temps
    lines.append("## Summary\n")
    lines.append("| Model | Temp | Hit Rate | Precision | TP | FN | FP | EOS% | Avg Len | TPS |")
    lines.append("|-------|------|----------|-----------|----|----|----|----- |---------|-----|")

    for model_id, temp_results in all_results.items():
        name = model_id.split("/")[-1]
        for temp, data in sorted(temp_results.items()):
            scores = data["scores"]
            if not scores:
                continue
            hr = np.mean([s["hit_rate"] for s in scores])
            prec = np.mean([s["precision"] for s in scores])
            tp = sum(s["n_tp"] for s in scores)
            fn = sum(s["n_fn"] for s in scores)
            fp = sum(s["n_fp"] for s in scores)
            eos = np.mean([s["has_eos"] for s in scores])
            avg_len = np.mean([s["dna_len"] for s in scores])
            tps = data["tps"]
            lines.append(
                f"| {name} | {temp} | {hr:.1%} | {prec:.1%} | {tp} | {fn} | {fp} | {eos:.0%} | {avg_len:.0f} | {tps:.0f} |"
            )

    # Per-category breakdown (best temp per model)
    lines.append("\n## Per-Category Breakdown (best temp per model)\n")

    for model_id, temp_results in all_results.items():
        name = model_id.split("/")[-1]
        # Find best temp by hit rate
        best_temp = max(temp_results.keys(),
                        key=lambda t: np.mean([s["hit_rate"] for s in temp_results[t]["scores"]]) if temp_results[t]["scores"] else 0)
        scores = temp_results[best_temp]["scores"]

        lines.append(f"### {name} (temp={best_temp})\n")
        lines.append("| Category | Requested | TP | FN | FP | Hit Rate |")
        lines.append("|----------|-----------|----|----|----|----- ---|")

        cat_stats = defaultdict(lambda: {"req": 0, "tp": 0, "fn": 0, "fp": 0})
        for s in scores:
            for t in s["requested"]:
                cat = t.strip("<>").split("_")[0]
                cat_stats[cat]["req"] += 1
                if t in s["tp"]:
                    cat_stats[cat]["tp"] += 1
                else:
                    cat_stats[cat]["fn"] += 1
            for t in s["fp"]:
                cat = t.strip("<>").split("_")[0]
                cat_stats[cat]["fp"] += 1

        tot = {"req": 0, "tp": 0, "fn": 0, "fp": 0}
        for cat in sorted(cat_stats.keys()):
            c = cat_stats[cat]
            tot["req"] += c["req"]; tot["tp"] += c["tp"]; tot["fn"] += c["fn"]; tot["fp"] += c["fp"]
            rate = c["tp"] / c["req"] if c["req"] else 0
            lines.append(f"| {cat} | {c['req']} | {c['tp']} | {c['fn']} | {c['fp']} | {rate:.1%} |")
        rate = tot["tp"] / tot["req"] if tot["req"] else 0
        lines.append(f"| **TOTAL** | **{tot['req']}** | **{tot['tp']}** | **{tot['fn']}** | **{tot['fp']}** | **{rate:.1%}** |")
        lines.append("")

    # Per-prompt details (first 15, best temp)
    lines.append("## Per-Prompt Details (first 15, best temp)\n")

    for model_id, temp_results in all_results.items():
        name = model_id.split("/")[-1]
        best_temp = max(temp_results.keys(),
                        key=lambda t: np.mean([s["hit_rate"] for s in temp_results[t]["scores"]]) if temp_results[t]["scores"] else 0)
        scores = temp_results[best_temp]["scores"]

        lines.append(f"### {name} (temp={best_temp})\n")
        for i, s in enumerate(scores[:15]):
            status = "PERFECT" if s["hit_rate"] == 1.0 else f"{s['hit_rate']:.0%}"
            lines.append(
                f"- **[{status}]** TP={s['n_tp']} FN={s['n_fn']} FP={s['n_fp']} "
                f"len={s['dna_len']} eos={s['has_eos']}"
            )
            if s["tp"]:
                lines.append(f"  - TP: {' '.join(s['tp'])}")
            if s["fn"]:
                scores_str = ", ".join(f"{t}({s['requested_scores'].get(t, 0):.2f})" for t in s["fn"])
                lines.append(f"  - FN: {scores_str}")
            if s["fp"]:
                lines.append(f"  - FP: {' '.join(s['fp'])}")
        lines.append("")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport: {output_path}")


def print_summary(all_results: dict):
    print("\n" + "=" * 80)
    print("ALIGNMENT EVALUATION SUMMARY")
    print("=" * 80)

    for model_id, temp_results in all_results.items():
        name = model_id.split("/")[-1]
        print(f"\n  {name}")
        print(f"  {'Temp':>5s} {'HitRate':>8s} {'Prec':>6s} {'TP':>5s} {'FN':>5s} {'FP':>5s} {'EOS%':>5s} {'Len':>6s}")
        print(f"  {'-'*5} {'-'*8} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")

        for temp in sorted(temp_results.keys()):
            scores = temp_results[temp]["scores"]
            if not scores:
                continue
            hr = np.mean([s["hit_rate"] for s in scores])
            prec = np.mean([s["precision"] for s in scores])
            tp = sum(s["n_tp"] for s in scores)
            fn = sum(s["n_fn"] for s in scores)
            fp = sum(s["n_fp"] for s in scores)
            eos = 100 * np.mean([s["has_eos"] for s in scores])
            avg_len = np.mean([s["dna_len"] for s in scores])
            print(f"  {temp:5.2f} {hr:7.1%} {prec:5.1%} {tp:5d} {fn:5d} {fp:5d} {eos:4.0f}% {avg_len:6.0f}")

    print("\n" + "=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alignment-based PlasmidLLM evaluation")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=50, help="Number of val prompts")
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--temps", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--cutoff", type=float, default=0.70, help="Min score ratio to count as found")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="eval_alignment")
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    parser.add_argument("--n-workers", type=int, default=4, help="Parallel alignment workers")
    parser.add_argument("--fp-scan", action="store_true", help="Scan all tokens for FP detection (slow)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts (once — same prompts for all temps)
    print(f"Loading val prompts: {args.parquet}")
    prompts = load_val_prompts(args.parquet, args.n, seed=args.seed)
    print(f"  {len(prompts)} prompts")

    # Verify registry loads
    print(f"Loading registry: {args.registry}")
    reg = load_registry(args.registry)
    print(f"  {len(reg)} entries, {reg['token'].nunique()} tokens")
    del reg  # workers will load their own copy

    all_results: dict[str, dict[float, dict]] = {}

    for model_id in args.models:
        name = model_id.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {model_id}")
        print(f"{'='*80}")

        model, tokenizer, device = load_model(model_id)
        all_results[model_id] = {}

        for temp in args.temps:
            print(f"\n  --- Temperature {temp} ---")

            # Generate
            candidates, tps = generate_batch(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                best_of=args.best_of,
                max_tokens=args.max_tokens,
                temperature=temp,
                seed=args.seed,
            )

            # Score
            scores = score_sequences(
                prompts, candidates,
                registry_path=args.registry,
                cutoff=args.cutoff,
                n_workers=args.n_workers,
                fp_scan=args.fp_scan,
            )

            all_results[model_id][temp] = {"scores": scores, "tps": tps}

            # Quick summary
            hr = np.mean([s["hit_rate"] for s in scores])
            tp = sum(s["n_tp"] for s in scores)
            fn = sum(s["n_fn"] for s in scores)
            fp = sum(s["n_fp"] for s in scores)
            print(f"  temp={temp}: hit_rate={hr:.1%} TP={tp} FN={fn} FP={fp}")

        # Free GPU between models
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save raw results as JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(
            {mid: {str(t): {"tps": d["tps"], "scores": d["scores"]}
                   for t, d in trs.items()}
             for mid, trs in all_results.items()},
            f, indent=2,
        )
    print(f"\nRaw results: {json_path}")

    # Report
    report_path = output_dir / "eval_report.md"
    write_report(all_results, report_path, args.cutoff)
    print_summary(all_results)


if __name__ == "__main__":
    main()
