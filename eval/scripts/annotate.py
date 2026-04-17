#!/usr/bin/env python3
"""Run annotation pipeline on generated sequences.

Runs pLannotate, Prodigal, and dustmasker on a FASTA file.
Outputs GenBank annotations, ORF predictions, and low-complexity stats.

Must be run in the plannotate conda env:
    conda run -n plannotate python eval/scripts/annotate.py \
        --input eval/runs/firstlight_20260408/generations.fasta \
        --output-dir eval/runs/firstlight_20260408

Parallelizes pLannotate with joblib (the bottleneck at ~5-30s/seq).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from Bio import SeqIO
from joblib import Parallel, delayed


def run_plannotate_single(fasta_path: str, output_dir: str, seq_id: str) -> dict:
    """Run pLannotate on a single sequence FASTA, return parsed results."""
    import warnings
    warnings.filterwarnings("ignore")

    try:
        from plannotate.annotate import annotate as plannotate_annotate
        record = next(SeqIO.parse(fasta_path, "fasta"))
        seq_str = str(record.seq)

        if len(seq_str) < 100:
            return {"seq_id": seq_id, "n_features": 0, "features": [], "error": "too_short"}

        hits = plannotate_annotate(seq_str, is_detailed=True, linear=False)

        # Save GenBank
        gbk_dir = Path(output_dir) / "annotations"
        gbk_dir.mkdir(parents=True, exist_ok=True)

        features = []
        if hasattr(hits, "iterrows"):
            for _, row in hits.iterrows():
                feat = {
                    "sseqid": row.get("sseqid", ""),
                    "Feature": row.get("Feature", ""),
                    "Type": row.get("Type", ""),
                    "pident": float(row.get("pident", 0)),
                    "qstart": int(row.get("qstart", 0)),
                    "qend": int(row.get("qend", 0)),
                    "slen": int(row.get("slen", 0)),
                    "length": int(row.get("length", row.get("qend", 0) - row.get("qstart", 0))),
                    "percmatch": float(row.get("percmatch", 0)),
                    "fragment": bool(row.get("fragment", False)),
                    "db": row.get("db", ""),
                }
                features.append(feat)

        return {
            "seq_id": seq_id,
            "n_features": len(features),
            "features": features,
            "coverage_bp": sum(f["length"] for f in features),
            "seq_length": len(seq_str),
        }
    except Exception as e:
        return {"seq_id": seq_id, "n_features": 0, "features": [], "error": str(e)}


def run_prodigal(fasta_path: str, output_dir: str) -> Path:
    """Run Prodigal in meta mode on the entire FASTA."""
    out = Path(output_dir) / "orfs"
    out.mkdir(parents=True, exist_ok=True)
    gff_path = out / "prodigal.gff"
    protein_path = out / "prodigal_proteins.faa"

    cmd = [
        "prodigal",
        "-i", fasta_path,
        "-o", str(gff_path),
        "-a", str(protein_path),
        "-p", "meta",
        "-f", "gff",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Prodigal error: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"Prodigal failed: {result.stderr[:200]}")

    print(f"Prodigal: {gff_path}")
    return gff_path


def run_dustmasker(fasta_path: str, output_dir: str) -> Path:
    """Run dustmasker for low-complexity regions."""
    out = Path(output_dir) / "dustmasker"
    out.mkdir(parents=True, exist_ok=True)
    intervals_path = out / "dustmasker.intervals"

    cmd = [
        "dustmasker",
        "-in", fasta_path,
        "-outfmt", "interval",
        "-out", str(intervals_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"dustmasker error: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"dustmasker failed: {result.stderr[:200]}")

    print(f"dustmasker: {intervals_path}")
    return intervals_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for pLannotate")
    parser.add_argument("--skip-plannotate", action="store_true")
    parser.add_argument("--skip-prodigal", action="store_true")
    parser.add_argument("--skip-dustmasker", action="store_true")
    args = parser.parse_args()

    fasta_path = args.input
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"Input: {len(records)} sequences from {fasta_path}")

    # 1. pLannotate (parallelized — the bottleneck)
    if not args.skip_plannotate:
        print(f"\nRunning pLannotate with {args.workers} workers...")
        t0 = time.time()

        # Write individual FASTAs for parallel processing
        tmp_dir = tempfile.mkdtemp(prefix="plannotate_")
        single_fastas = []
        for i, rec in enumerate(records):
            tmp_fa = Path(tmp_dir) / f"seq_{i}.fasta"
            SeqIO.write(rec, str(tmp_fa), "fasta")
            single_fastas.append((str(tmp_fa), str(out), rec.id))

        results = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(run_plannotate_single)(fa, odir, sid)
            for fa, odir, sid in single_fastas
        )

        plannotate_time = time.time() - t0
        n_annotated = sum(1 for r in results if r.get("n_features", 0) > 0)
        n_errors = sum(1 for r in results if "error" in r)
        print(f"pLannotate: {len(results)} sequences in {plannotate_time:.1f}s "
              f"({plannotate_time/max(len(results),1):.1f}s/seq), "
              f"{n_annotated} annotated, {n_errors} errors")

        # Save aggregated results
        with open(out / "plannotate_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Cleanup temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 2. Prodigal
    if not args.skip_prodigal:
        print("\nRunning Prodigal...")
        t0 = time.time()
        run_prodigal(fasta_path, str(out))
        print(f"Prodigal: {time.time()-t0:.1f}s")

    # 3. dustmasker
    if not args.skip_dustmasker:
        print("\nRunning dustmasker...")
        t0 = time.time()
        run_dustmasker(fasta_path, str(out))
        print(f"dustmasker: {time.time()-t0:.1f}s")

    print(f"\nAnnotation complete. Results in {out}")


if __name__ == "__main__":
    main()
