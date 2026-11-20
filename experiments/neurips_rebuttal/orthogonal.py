#!/usr/bin/env python3
"""Re-score requested resistance markers with an annotator independent of the reward.

Written for the NeurIPS 2026 rebuttal (Dok5 Q3; meta-review point 1).

The objection is that reward, faithfulness metric and best-of-K ranker all run through one
660-entry motif registry and one pLannotate install, so the reported gains might be gains
against that scorer rather than real controllability. NCBI AMRFinderPlus shares neither the
registry nor the databases nor the detection method: it calls curated HMMs and a
hand-curated reference gene catalogue over translated ORFs, so agreement with it is
genuinely independent evidence.

Scope: AMR tokens only — that is what AMRFinderPlus covers. It is a partial answer to the
coupling objection, not a complete one, and should be described that way.

Input is Prodigal protein FASTA (already archived per run) run through
`amrfinder --protein --plus`. Prodigal ids are `<seq_id>_<orf_number>`, so the generation a
hit belongs to is the id minus its last underscore field.

Usage:
  amrfinder --protein prodigal_proteins.faa --plus -o run_amr.tsv
  python orthogonal.py \\
      --amr base_amr.tsv --generations base_generations.parquet --label base
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

import duckdb

AMR_TOKEN_RE = re.compile(r"<AMR_[A-Z0-9_]+>")

# AMRFinderPlus element symbol -> the resistance token(s) that symbol satisfies.
# Sets, not single tokens: aph(3')-II is the classic NeoR/KanR cassette and legitimately
# satisfies either request, so crediting only one would understate recall.
SYMBOL_RULES: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"^blaTEM|^bla", re.I), {"<AMR_AMPICILLIN>"}),
    (re.compile(r"^aph\(3'\)-II", re.I), {"<AMR_KANAMYCIN>", "<AMR_NEOMYCIN>"}),
    (re.compile(r"^aph\(3'\)-I", re.I), {"<AMR_KANAMYCIN>", "<AMR_NEOMYCIN>"}),
    (re.compile(r"^aph\(4\)|^hph", re.I), {"<AMR_HYGROMYCIN>"}),
    (re.compile(r"^aac\(3\)", re.I), {"<AMR_GENTAMICIN>"}),
    (re.compile(r"^ble", re.I), {"<AMR_ZEOCIN>"}),
    (re.compile(r"^aadA", re.I), {"<AMR_SPECTINOMYCIN>", "<AMR_STREPTOMYCIN>"}),
    (re.compile(r"^tet", re.I), {"<AMR_TETRACYCLINE>"}),
    (re.compile(r"^cat|^cml|^flo", re.I), {"<AMR_CHLORAMPHENICOL>"}),
    (re.compile(r"^pac|^puro", re.I), {"<AMR_PUROMYCIN>"}),
    (re.compile(r"^bsd|^bsr", re.I), {"<AMR_BLASTICIDIN>"}),
    (re.compile(r"^nat|^sat", re.I), {"<AMR_NOURSEOTHRICIN>"}),
    (re.compile(r"^aac\(6", re.I), {"<AMR_APRAMYCIN>"}),
]

# Fallback when no symbol rule fires but the drug class is unambiguous.
CLASS_RULES = {
    "BETA-LACTAM": {"<AMR_AMPICILLIN>"},
    "BLEOMYCIN": {"<AMR_ZEOCIN>"},
    "TETRACYCLINE": {"<AMR_TETRACYCLINE>"},
    "PHENICOL": {"<AMR_CHLORAMPHENICOL>"},
}


def complete_protein_ids(proteins: Path) -> set[str]:
    """Require explicit Prodigal completeness and an unambiguous translation."""
    from Bio import SeqIO

    complete = set()
    for record in SeqIO.parse(str(proteins), "fasta"):
        partial = re.search(r"\bpartial=([01]{2})\b", record.description)
        if partial is None:
            raise ValueError(f"Missing Prodigal partial flag for {record.id}")
        peptide = str(record.seq).removesuffix("*")
        if partial.group(1) == "00" and re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", peptide):
            complete.add(record.id)
    return complete


def is_intact_hit(row: dict[str, str], complete: set[str], coverage: float = 90.0) -> bool:
    """Conservative intact endpoint; partial/HMM-only hits stay in legacy recall."""
    if "Method" not in row:
        raise ValueError("AMRFinder output missing columns: ['Method']")
    coverage_col = next(
        (col for col in ("% Coverage of reference", "% Coverage of reference protein") if col in row),
        None,
    )
    if coverage_col is None:
        raise ValueError("AMRFinder output missing columns: ['% Coverage of reference']")
    pid = row.get("Protein id") or row.get("Protein identifier")
    if pid is None:
        raise ValueError("AMRFinder output lacks protein identifiers")
    method = row["Method"]
    if method not in {"ALLELE", "EXACT", "BLAST", "ALLELEP", "EXACTP", "BLASTP"} or pid not in complete:
        return False
    value = row[coverage_col]
    if value in {"NA", ""}:
        return False
    return float(value) >= coverage


def tokens_for(symbol: str, klass: str) -> set[str]:
    for pat, toks in SYMBOL_RULES:
        if pat.search(symbol or ""):
            return toks
    return CLASS_RULES.get((klass or "").upper(), set())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amr", type=Path, required=True)
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--intact-only", action="store_true")
    ap.add_argument("--proteins", type=Path, help="Prodigal FASTA with partial=00 header flags")
    ap.add_argument("--output", type=Path, help="Per-generation parquet for paired statistics")
    args = ap.parse_args()
    if args.intact_only and args.proteins is None:
        ap.error("--intact-only requires --proteins")
    complete = complete_protein_ids(args.proteins) if args.intact_only else set()

    detected: dict[str, set[str]] = collections.defaultdict(set)
    unmapped: collections.Counter = collections.Counter()
    with open(args.amr) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if args.intact_only and not is_intact_hit(row, complete):
                continue
            pid = row.get("Protein id") or row.get("Protein identifier") or ""
            seq_id = pid.rsplit("_", 1)[0]
            toks = tokens_for(row.get("Element symbol", ""), row.get("Class", ""))
            if toks:
                detected[seq_id] |= toks
            else:
                unmapped[row.get("Element symbol", "")] += 1

    prompts = [r[0] for r in duckdb.read_parquet(str(args.generations)).project("prompt").fetchall()]
    unknown = set(detected) - {f"gen_{i}" for i in range(len(prompts))}
    if unknown:
        raise ValueError(f"AMRFinder sequence IDs absent from generations: {sorted(unknown)[:5]}")

    recalls, n_req, n_any = [], 0, 0
    per_token_req: collections.Counter = collections.Counter()
    per_token_hit: collections.Counter = collections.Counter()
    records = []
    for i, prompt in enumerate(prompts):
        req = set(AMR_TOKEN_RE.findall(prompt))
        if not req:
            continue
        det = detected.get(f"gen_{i}", set())
        n_req += 1
        hit = req & det
        records.append(dict(sequence_id=f"gen_{i}", prompt=prompt, requested=sorted(req),
                            detected=sorted(det), recall=len(hit) / len(req),
                            all_requested=req <= det, intact_only=args.intact_only))
        n_any += bool(hit)
        recalls.append(len(hit) / len(req))
        for t in req:
            per_token_req[t] += 1
            if t in det:
                per_token_hit[t] += 1

    if not n_req:
        raise ValueError("No AMR-requesting prompts in the generation panel")
    if args.output:
        import pandas as pd

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            pd.DataFrame(records).to_parquet(output, index=False)
    print(f"\n=== {args.label} — AMRFinderPlus (independent of registry & pLannotate) ===")
    print(f"  intact-only filtering            : {args.intact_only}")
    print(f"  prompts requesting >=1 AMR token : {n_req}")
    print(
        f"  token-pooled recall              : "
        f"{sum(per_token_hit.values()) / max(sum(per_token_req.values()), 1) * 100:5.1f}%"
    )
    print(f"  mean per-prompt recall           : {statistics.mean(recalls) * 100:5.1f}%")
    print(f"  prompts with >=1 requested AMR found: {n_any / max(n_req, 1) * 100:5.1f}%")
    print(f"\n  {'token':<26}{'requested':>10}{'found':>8}{'rate':>8}")
    for t, r in per_token_req.most_common(10):
        print(f"  {t:<26}{r:>10}{per_token_hit[t]:>8}{per_token_hit[t] / r * 100:>7.1f}%")
    if unmapped:
        print(f"\n  unmapped AMRFinder symbols: {unmapped.most_common(5)}")


if __name__ == "__main__":
    main()
