#!/usr/bin/env python3
"""
seq_qc.py - Track 3: Sequence quality and complexity metrics for SPACE pipeline.

Calculates:
- GC content
- Homopolymer runs
- Repeat regions
- Linguistic complexity
- Synthesis risk score (based on DNA Chisel heuristics)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from Bio import SeqIO


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def find_homopolymers(sequence: str, min_length: int = 5) -> Dict[str, Any]:
    """
    Find homopolymer runs in sequence.

    Returns:
        - max_length: longest homopolymer
        - count: number of homopolymers >= min_length
        - positions: list of (start, end, base, length)
    """
    sequence = sequence.upper()
    homopolymers = []

    # Find runs of same nucleotide
    i = 0
    while i < len(sequence):
        base = sequence[i]
        if base not in "ATCG":
            i += 1
            continue

        j = i + 1
        while j < len(sequence) and sequence[j] == base:
            j += 1

        length = j - i
        if length >= min_length:
            homopolymers.append({
                "start": i,
                "end": j,
                "base": base,
                "length": length,
            })
        i = j

    max_length = max((h["length"] for h in homopolymers), default=0)

    return {
        "max_length": max_length,
        "count": len(homopolymers),
        "total_bases": sum(h["length"] for h in homopolymers),
        "positions": homopolymers[:20],  # Limit to first 20 for output size
    }


def find_tandem_repeats(sequence: str, min_unit: int = 2, max_unit: int = 10, min_copies: int = 3) -> Dict[str, Any]:
    """
    Find simple tandem repeats in sequence.

    Returns stats about repeat regions.
    """
    sequence = sequence.upper()
    repeats = []

    for unit_len in range(min_unit, max_unit + 1):
        i = 0
        while i <= len(sequence) - unit_len * min_copies:
            unit = sequence[i:i + unit_len]

            # Count consecutive copies
            copies = 1
            j = i + unit_len
            while j + unit_len <= len(sequence) and sequence[j:j + unit_len] == unit:
                copies += 1
                j += unit_len

            if copies >= min_copies:
                repeats.append({
                    "start": i,
                    "end": j,
                    "unit": unit,
                    "unit_length": unit_len,
                    "copies": copies,
                    "total_length": copies * unit_len,
                })
                i = j  # Skip past this repeat
            else:
                i += 1

    total_repeat_bases = sum(r["total_length"] for r in repeats)

    return {
        "count": len(repeats),
        "total_bases": total_repeat_bases,
        "fraction": total_repeat_bases / len(sequence) if sequence else 0,
        "longest": max((r["total_length"] for r in repeats), default=0),
        "repeats": repeats[:20],  # Limit output size
    }


def calculate_linguistic_complexity(sequence: str, k: int = 3) -> float:
    """
    Calculate linguistic complexity based on k-mer diversity.

    LC = (observed k-mers) / (possible k-mers)
    Higher values indicate more complex/diverse sequences.
    """
    if len(sequence) < k:
        return 0.0

    sequence = sequence.upper()

    # Count unique k-mers
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if all(b in "ATCG" for b in kmer):
            kmers.add(kmer)

    observed = len(kmers)
    possible = min(4 ** k, len(sequence) - k + 1)

    return observed / possible if possible > 0 else 0.0


def find_gc_extremes(sequence: str, window_size: int = 50) -> Dict[str, Any]:
    """
    Find regions with extreme GC content that may cause synthesis issues.
    """
    if len(sequence) < window_size:
        return {"high_gc_regions": [], "low_gc_regions": [], "gc_range": 0}

    sequence = sequence.upper()
    high_gc = []  # >70%
    low_gc = []   # <30%

    for i in range(0, len(sequence) - window_size + 1, window_size // 2):
        window = sequence[i:i + window_size]
        gc = (window.count("G") + window.count("C")) / len(window)

        if gc > 0.70:
            high_gc.append({"start": i, "end": i + window_size, "gc": gc})
        elif gc < 0.30:
            low_gc.append({"start": i, "end": i + window_size, "gc": gc})

    # Calculate GC range
    gc_values = []
    for i in range(0, len(sequence) - window_size + 1, window_size):
        window = sequence[i:i + window_size]
        gc_values.append((window.count("G") + window.count("C")) / len(window))

    gc_range = max(gc_values) - min(gc_values) if gc_values else 0

    return {
        "high_gc_regions": len(high_gc),
        "low_gc_regions": len(low_gc),
        "gc_range": gc_range,
        "high_gc_positions": high_gc[:10],
        "low_gc_positions": low_gc[:10],
    }


def find_hairpins(sequence: str, stem_min: int = 6, loop_min: int = 3, loop_max: int = 8) -> int:
    """
    Estimate number of potential hairpin structures.
    Simple heuristic based on inverted repeats.
    """
    sequence = sequence.upper()
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    hairpin_count = 0

    for i in range(len(sequence) - stem_min * 2 - loop_min):
        stem = sequence[i:i + stem_min]

        # Look for complement in downstream region
        for loop_size in range(loop_min, loop_max + 1):
            j = i + stem_min + loop_size
            if j + stem_min > len(sequence):
                break

            potential_stem = sequence[j:j + stem_min]
            complement_stem = "".join(complement.get(b, "N") for b in stem)[::-1]

            if potential_stem == complement_stem:
                hairpin_count += 1
                break

    return hairpin_count


def calculate_synthesis_risk(
    gc_content: float,
    homopolymers: Dict,
    repeats: Dict,
    gc_extremes: Dict,
    hairpins: int,
    length: int,
) -> Tuple[float, List[str]]:
    """
    Calculate synthesis risk score (0-1) based on DNA Chisel-like heuristics.

    Higher score = higher risk of synthesis failure.
    """
    score = 0.0
    reasons = []

    # GC content issues
    if gc_content > 0.65:
        score += 0.15
        reasons.append(f"High GC content ({gc_content:.1%})")
    elif gc_content < 0.35:
        score += 0.15
        reasons.append(f"Low GC content ({gc_content:.1%})")

    # Homopolymer issues
    if homopolymers["max_length"] >= 8:
        score += 0.25
        reasons.append(f"Long homopolymer ({homopolymers['max_length']} bp)")
    elif homopolymers["max_length"] >= 6:
        score += 0.10
        reasons.append(f"Moderate homopolymer ({homopolymers['max_length']} bp)")

    # Repeat issues
    if repeats["fraction"] > 0.10:
        score += 0.20
        reasons.append(f"High repeat content ({repeats['fraction']:.1%})")
    elif repeats["fraction"] > 0.05:
        score += 0.10
        reasons.append(f"Moderate repeat content ({repeats['fraction']:.1%})")

    # GC extreme regions
    extreme_count = gc_extremes["high_gc_regions"] + gc_extremes["low_gc_regions"]
    if extreme_count > 5:
        score += 0.15
        reasons.append(f"Many GC extreme regions ({extreme_count})")

    # Hairpin structures
    if hairpins > 10:
        score += 0.15
        reasons.append(f"Many potential hairpins ({hairpins})")

    # Clamp score to 0-1
    score = min(1.0, max(0.0, score))

    return score, reasons


def main():
    parser = argparse.ArgumentParser(description="Calculate sequence QC metrics")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--sample-id", required=True, help="Sample ID")

    args = parser.parse_args()

    # Read sequence
    input_path = Path(args.input)
    with open(input_path, "r") as f:
        record = SeqIO.read(f, "fasta")
    sequence = str(record.seq).upper()

    # Calculate metrics
    gc_content = calculate_gc_content(sequence)
    homopolymers = find_homopolymers(sequence)
    repeats = find_tandem_repeats(sequence)
    complexity = calculate_linguistic_complexity(sequence)
    gc_extremes = find_gc_extremes(sequence)
    hairpins = find_hairpins(sequence)

    synthesis_risk, risk_reasons = calculate_synthesis_risk(
        gc_content, homopolymers, repeats, gc_extremes, hairpins, len(sequence)
    )

    # Build result
    result = {
        "sample_id": args.sample_id,
        "length": len(sequence),
        "gc_content": gc_content,
        "linguistic_complexity": complexity,
        "homopolymers": homopolymers,
        "tandem_repeats": repeats,
        "gc_extremes": gc_extremes,
        "hairpin_estimate": hairpins,
        "synthesis_risk": synthesis_risk,
        "synthesis_risk_reasons": risk_reasons,
    }

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"[{args.sample_id}] QC complete")
    print(f"  Length: {len(sequence)} bp")
    print(f"  GC content: {gc_content:.1%}")
    print(f"  Linguistic complexity: {complexity:.3f}")
    print(f"  Max homopolymer: {homopolymers['max_length']} bp")
    print(f"  Synthesis risk: {synthesis_risk:.2f}")
    if risk_reasons:
        print(f"  Risk factors: {', '.join(risk_reasons)}")


if __name__ == "__main__":
    main()
