#!/usr/bin/env python3
"""Build a motif registry mapping feature tokens to canonical reference sequences.

Extracts canonical sequences from plannotate's underlying BLAST/Diamond/Infernal
databases, links them to the <FEAT_*> tokens used in training, and outputs a
registry for use as a GRPO reward function lookup.

Architecture:
    Token (<FEAT_CMR>) → UUID (deterministic) → Canonical Sequence(s)
                                                  ├── CmR_(1)  [snapgene, dna]
                                                  ├── CmR_(2)  [snapgene, dna]
                                                  └── P62577   [swissprot, protein]

Output:
    motif_registry.json   — nested dict keyed by UUID
    motif_registry.parquet — flat table (one row per uuid+sseqid pair)
"""

import argparse
import gzip
import html
import json
import logging
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE = "/mnt/s3/phd-research-storage-1758274488/addgene_clean"

# Deterministic UUID namespace for motif IDs
MOTIF_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0d02b2c3d479")

# Quality thresholds (same as generate_training_pairs_v3.py)
MIN_PERCMATCH = 95.0
EXCLUDE_FRAGMENTS = True
TOP_N_FEATURES = 150


# ── Reused from generate_training_pairs_v3.py ─────────────────────────────────

def feature_to_token(feature: str) -> str:
    """Convert a plannotate feature name to a valid <TOKEN> string."""
    tok = feature.upper()
    tok = re.sub(r"[^A-Z0-9]+", "_", tok)
    tok = tok.strip("_")
    tok = re.sub(r"_+", "_", tok)
    return f"<FEAT_{tok}>"


# ── Step 1: Load plannotate metadata CSVs ──────────────────────────────────────

def load_plannotate_metadata() -> pd.DataFrame:
    """Load and concatenate metadata from plannotate's bundled CSVs."""
    import plannotate
    data_dir = Path(plannotate.__file__).parent / "data" / "data"

    frames = []

    # snapgene.csv: has header (sseqid, Feature, Type, Description)
    sg = pd.read_csv(data_dir / "snapgene.csv")
    sg["db_source"] = "snapgene"
    frames.append(sg)

    # fpbase.csv: has header (sseqid, Feature, Description); Type defaults to CDS
    fp = pd.read_csv(data_dir / "fpbase.csv")
    fp["Type"] = "CDS"
    fp["db_source"] = "fpbase"
    # Decode HTML entities (e.g., &alpha;GFP → αGFP)
    for col in ["sseqid", "Feature", "Description"]:
        fp[col] = fp[col].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)
    frames.append(fp)

    # swissprot.csv.gz: NO header; cols = sseqid, Feature, Description; Type = CDS
    sp = pd.read_csv(
        data_dir / "swissprot.csv.gz",
        header=None,
        names=["sseqid", "Feature", "Description"],
        compression="gzip",
    )
    sp["Type"] = "CDS"
    sp["db_source"] = "swissprot"
    frames.append(sp)

    meta = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded metadata: %d snapgene, %d fpbase, %d swissprot (%d total)",
        len(sg), len(fp), len(sp), len(meta),
    )
    return meta


# ── Step 2: Extract canonical sequences from databases ─────────────────────────

def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into [(header, sequence), ...]."""
    entries = []
    current_id = None
    current_seq = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq)))
            # Take first whitespace-delimited word as ID
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        entries.append((current_id, "".join(current_seq)))
    return entries


def _run_cmd(cmd: list[str], description: str) -> str | None:
    """Run a shell command, return stdout or None on failure."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("%s failed (rc=%d): %s", description, result.returncode, result.stderr[:500])
            return None
        return result.stdout
    except FileNotFoundError:
        logger.warning("%s: command not found (%s)", description, cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("%s: timed out after 600s", description)
        return None


def extract_sequences(
    db_dir: Path,
    needed_sseqids: set[str] | None = None,
) -> pd.DataFrame:
    """Extract canonical sequences from plannotate BLAST/Diamond/Infernal databases.

    Args:
        db_dir: Path to ~/.cache/pLannotate/BLAST_dbs/
        needed_sseqids: If provided, only keep sequences with these sseqids.

    Returns:
        DataFrame with columns [sseqid, sequence, seq_type, db_source].
    """
    all_seqs = []

    # ── snapgene (BLAST nucleotide DB) ──
    snapgene_db = db_dir / "snapgene"
    if snapgene_db.with_suffix(".nsq").exists() or snapgene_db.with_suffix(".ndb").exists():
        out = _run_cmd(
            ["blastdbcmd", "-db", str(snapgene_db), "-entry", "all"],
            "snapgene blastdbcmd",
        )
        if out:
            for sid, seq in _parse_fasta(out):
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "dna", "db_source": "snapgene"})
            logger.info("  snapgene: extracted %d sequences", sum(1 for s in all_seqs if s["db_source"] == "snapgene"))
    else:
        logger.warning("snapgene BLAST DB not found at %s", snapgene_db)

    # ── fpbase (Diamond protein DB) ──
    fpbase_dmnd = db_dir / "fpbase.dmnd"
    if fpbase_dmnd.exists():
        out = _run_cmd(
            ["diamond", "getseq", "-d", str(fpbase_dmnd)],
            "fpbase diamond getseq",
        )
        if out:
            for sid, seq in _parse_fasta(out):
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "protein", "db_source": "fpbase"})
            logger.info("  fpbase: extracted %d sequences", sum(1 for s in all_seqs if s["db_source"] == "fpbase"))
    else:
        logger.warning("fpbase Diamond DB not found at %s", fpbase_dmnd)

    # ── swissprot (Diamond protein DB) — filter to needed sseqids ──
    swissprot_dmnd = db_dir / "swissprot.dmnd"
    if swissprot_dmnd.exists():
        out = _run_cmd(
            ["diamond", "getseq", "-d", str(swissprot_dmnd)],
            "swissprot diamond getseq",
        )
        if out:
            n_total = 0
            n_kept = 0
            for sid, seq in _parse_fasta(out):
                n_total += 1
                if needed_sseqids is not None and sid not in needed_sseqids:
                    continue
                n_kept += 1
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "protein", "db_source": "swissprot"})
            logger.info("  swissprot: extracted %d/%d sequences (filtered to needed)", n_kept, n_total)
    else:
        logger.warning("swissprot Diamond DB not found at %s", swissprot_dmnd)

    # ── Rfam (Infernal covariance model) ──
    rfam_cm = db_dir / "Rfam.cm"
    if rfam_cm.exists():
        out = _run_cmd(
            ["cmemit", "-c", str(rfam_cm)],
            "Rfam cmemit",
        )
        if out:
            for sid, seq in _parse_fasta(out):
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "rna_consensus", "db_source": "Rfam"})
            logger.info("  Rfam: extracted %d consensus sequences", sum(1 for s in all_seqs if s["db_source"] == "Rfam"))
    else:
        logger.warning("Rfam CM not found at %s", rfam_cm)

    if not all_seqs:
        logger.warning("No sequences extracted from any database!")
        return pd.DataFrame(columns=["sseqid", "sequence", "seq_type", "db_source"])

    return pd.DataFrame(all_seqs)


# ── Step 3–6: Build registry ──────────────────────────────────────────────────

def build_registry(
    metadata_df: pd.DataFrame,
    sequences_df: pd.DataFrame | None,
    annotations_df: pd.DataFrame,
    vocab: dict[str, int],
    top_n_features: int,
    include_all_annotated: bool,
) -> dict:
    """Build the motif registry from metadata, sequences, and annotations.

    Returns the registry dict ready for JSON serialization.
    """
    # ── Filter annotations (same quality thresholds as v3) ──
    ann = annotations_df.copy()
    ann = ann[ann["percmatch"] >= MIN_PERCMATCH]
    if EXCLUDE_FRAGMENTS:
        ann = ann[~ann["fragment"].fillna(False).astype(bool)]
    logger.info("Annotations after quality filter: %d", len(ann))

    # ── Find in-scope features ──
    feat_counts = ann.groupby("Feature")["plasmid_id"].nunique().sort_values(ascending=False)
    top_features = set(feat_counts.head(top_n_features).index)

    if include_all_annotated:
        all_annotated_features = set(ann["Feature"].unique())
        in_scope_features = all_annotated_features
        logger.info(
            "Including all %d annotated features (%d are top-%d)",
            len(in_scope_features), len(top_features), top_n_features,
        )
    else:
        in_scope_features = top_features
        logger.info("Using top-%d features only", top_n_features)

    # Get the set of (Feature, sseqid, db) tuples from annotations
    ann_scope = ann[ann["Feature"].isin(in_scope_features)]
    annotation_tuples = set(
        zip(ann_scope["Feature"], ann_scope["sseqid"], ann_scope["db"])
    )
    needed_sseqids = {t[1] for t in annotation_tuples}
    logger.info(
        "In-scope: %d features, %d unique sseqids across %d annotation tuples",
        len(in_scope_features), len(needed_sseqids), len(annotation_tuples),
    )

    # ── Build sseqid → sequence lookup ──
    seq_lookup: dict[str, dict] = {}  # sseqid → {sequence, seq_type, db_source}
    if sequences_df is not None and len(sequences_df) > 0:
        for _, row in sequences_df.iterrows():
            seq_lookup[row["sseqid"]] = {
                "sequence": row["sequence"],
                "seq_type": row["seq_type"],
                "db_source": row["db_source"],
            }

    # ── Build sseqid → metadata lookup ──
    meta_lookup: dict[str, dict] = {}
    for _, row in metadata_df.iterrows():
        meta_lookup[row["sseqid"]] = {
            "Feature": row["Feature"],
            "Type": row.get("Type", "CDS"),
            "Description": row.get("Description", ""),
            "db_source": row["db_source"],
        }

    # ── Map features → tokens, detect collisions ──
    token_map: dict[str, list[str]] = defaultdict(list)  # token → [feature_names]
    for feat in in_scope_features:
        tok = feature_to_token(feat)
        token_map[tok].append(feat)

    # Warn about token collisions
    for tok, feats in token_map.items():
        if len(feats) > 1:
            logger.warning(
                "Token collision: %s maps to %d features: %s",
                tok, len(feats), feats,
            )

    # ── Build motif entries ──
    motifs: dict[str, dict] = {}  # uuid → motif entry
    token_to_uuid: dict[str, str] = {}
    sseqid_to_uuid: dict[str, str] = {}

    for feat in sorted(in_scope_features):
        motif_uuid = str(uuid.uuid5(MOTIF_NAMESPACE, feat))
        tok = feature_to_token(feat)
        is_in_vocab = tok in vocab

        # Collect all sseqids for this feature from annotations
        feat_sseqids = {
            (t[1], t[2]) for t in annotation_tuples if t[0] == feat
        }

        # Also check metadata for this feature (some sseqids may not be in annotations)
        meta_sseqids = {
            sid for sid, m in meta_lookup.items() if m["Feature"] == feat
        }

        # Union: annotation sseqids + metadata sseqids
        all_sseqids_for_feat = {sid for sid, _ in feat_sseqids} | meta_sseqids

        # Build sequence entries
        sequences = []
        for sid in sorted(all_sseqids_for_feat):
            entry = {"sseqid": sid}
            # Get metadata
            if sid in meta_lookup:
                entry["db_source"] = meta_lookup[sid]["db_source"]
            else:
                # Try to infer from annotations
                ann_db = {db for s, db in feat_sseqids if s == sid}
                entry["db_source"] = next(iter(ann_db)) if ann_db else "unknown"

            # Get sequence
            if sid in seq_lookup:
                entry["sequence"] = seq_lookup[sid]["sequence"]
                entry["seq_type"] = seq_lookup[sid]["seq_type"]
                entry["seq_len"] = len(seq_lookup[sid]["sequence"])
            else:
                entry["sequence"] = None
                entry["seq_type"] = None
                entry["seq_len"] = None

            sequences.append(entry)
            sseqid_to_uuid[sid] = motif_uuid

        # Determine type from metadata (prefer snapgene which has explicit Type)
        feat_type = "CDS"  # default
        for sid in all_sseqids_for_feat:
            if sid in meta_lookup and meta_lookup[sid].get("Type"):
                feat_type = meta_lookup[sid]["Type"]
                if meta_lookup[sid]["db_source"] == "snapgene":
                    break  # prefer snapgene type

        # Get description (prefer snapgene, then fpbase, then swissprot)
        description = ""
        for db_pref in ["snapgene", "fpbase", "swissprot"]:
            for sid in all_sseqids_for_feat:
                if sid in meta_lookup and meta_lookup[sid]["db_source"] == db_pref:
                    desc = meta_lookup[sid].get("Description", "")
                    if desc and str(desc) != "nan":
                        description = str(desc)
                        break
            if description:
                break

        motifs[motif_uuid] = {
            "uuid": motif_uuid,
            "feature_name": feat,
            "token": tok,
            "type": feat_type,
            "in_training_vocab": is_in_vocab,
            "in_top_n": feat in top_features,
            "plasmid_count": int(feat_counts.get(feat, 0)),
            "description": description,
            "sequences": sequences,
        }

        token_to_uuid[tok] = motif_uuid

    # ── Summary stats ──
    n_with_seq = sum(
        1 for m in motifs.values()
        if any(s["sequence"] is not None for s in m["sequences"])
    )
    n_in_vocab = sum(1 for m in motifs.values() if m["in_training_vocab"])
    logger.info(
        "Registry: %d motifs (%d with sequences, %d in training vocab)",
        len(motifs), n_with_seq, n_in_vocab,
    )

    # Check vocab coverage
    vocab_feat_tokens = {t for t in vocab if t.startswith("<FEAT_")}
    registry_tokens = set(token_to_uuid.keys())
    missing_from_registry = vocab_feat_tokens - registry_tokens
    if missing_from_registry:
        logger.warning(
            "%d vocab tokens not in registry: %s",
            len(missing_from_registry),
            sorted(missing_from_registry)[:10],
        )

    return {
        "version": "1.0",
        "namespace_uuid": str(MOTIF_NAMESPACE),
        "n_motifs": len(motifs),
        "n_with_sequences": n_with_seq,
        "n_in_training_vocab": n_in_vocab,
        "motifs": motifs,
        "token_to_uuid": token_to_uuid,
        "sseqid_to_uuid": sseqid_to_uuid,
    }


def registry_to_flat_df(registry: dict) -> pd.DataFrame:
    """Convert registry to a flat DataFrame (one row per uuid+sseqid pair)."""
    rows = []
    for motif_uuid, motif in registry["motifs"].items():
        for seq_entry in motif["sequences"]:
            rows.append({
                "uuid": motif_uuid,
                "feature_name": motif["feature_name"],
                "token": motif["token"],
                "type": motif["type"],
                "in_training_vocab": motif["in_training_vocab"],
                "in_top_n": motif["in_top_n"],
                "plasmid_count": motif["plasmid_count"],
                "sseqid": seq_entry["sseqid"],
                "db_source": seq_entry["db_source"],
                "seq_type": seq_entry.get("seq_type"),
                "seq_len": seq_entry.get("seq_len"),
                "sequence": seq_entry.get("sequence"),
            })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        default=f"{BASE}/annotations/plannotate_annotations.parquet",
        help="Path to plannotate_annotations.parquet",
    )
    parser.add_argument(
        "--vocab",
        default=f"{BASE}/tokenization/token_vocabulary_v3.json",
        help="Path to v3 vocab JSON",
    )
    parser.add_argument(
        "--db-dir",
        default=str(Path.home() / ".cache" / "pLannotate" / "BLAST_dbs"),
        help="Path to plannotate BLAST/Diamond database directory",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{BASE}/tokenization",
        help="Directory for output files",
    )
    parser.add_argument(
        "--top-n-features", type=int, default=TOP_N_FEATURES,
        help="Number of top features by plasmid count",
    )
    parser.add_argument(
        "--min-percmatch", type=float, default=MIN_PERCMATCH,
        help="Minimum percmatch for annotation quality filter",
    )
    parser.add_argument(
        "--include-all-annotated", action="store_true",
        help="Include all annotated features (not just top-N)",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Skip sequence extraction (produce entries with sequence=null)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Step 1: Load plannotate metadata ──
    logger.info("Step 1: Loading plannotate metadata CSVs...")
    metadata_df = load_plannotate_metadata()

    # ── Step 3: Load annotations and find in-scope features ──
    logger.info("Step 3: Loading annotations...")
    ann_cols = ["plasmid_id", "Feature", "sseqid", "db", "percmatch", "fragment",
                "pident", "slen", "Type", "Description"]
    ann = pq.read_table(args.annotations, columns=ann_cols).to_pandas()
    ann["plasmid_id"] = ann["plasmid_id"].astype(str)
    logger.info("  Loaded %d annotations for %d plasmids", len(ann), ann["plasmid_id"].nunique())

    # Find sseqids we need (for filtering swissprot extraction)
    ann_filtered = ann[ann["percmatch"] >= args.min_percmatch].copy()
    if EXCLUDE_FRAGMENTS:
        ann_filtered = ann_filtered[~ann_filtered["fragment"].fillna(False).astype(bool)]
    needed_sseqids = set(ann_filtered["sseqid"].unique())
    logger.info("  %d unique sseqids in filtered annotations", len(needed_sseqids))

    # ── Step 2: Extract sequences ──
    sequences_df = None
    if not args.metadata_only:
        db_dir = Path(args.db_dir)
        if db_dir.exists():
            logger.info("Step 2: Extracting sequences from %s...", db_dir)
            sequences_df = extract_sequences(db_dir, needed_sseqids=needed_sseqids)
            logger.info("  Total sequences extracted: %d", len(sequences_df))
        else:
            logger.warning("DB directory not found: %s — skipping sequence extraction", db_dir)
    else:
        logger.info("Step 2: Skipped (--metadata-only)")

    # ── Step 4: Load vocab ──
    logger.info("Step 4: Loading vocab from %s...", args.vocab)
    with open(args.vocab) as f:
        vocab_data = json.load(f)
    if isinstance(vocab_data, dict) and "token_to_id" in vocab_data:
        vocab = vocab_data["token_to_id"]
    else:
        vocab = vocab_data
    logger.info("  Vocab size: %d tokens (%d FEAT tokens)",
                len(vocab), sum(1 for t in vocab if t.startswith("<FEAT_")))

    # ── Step 5: Build registry ──
    logger.info("Step 5: Building registry...")
    registry = build_registry(
        metadata_df=metadata_df,
        sequences_df=sequences_df,
        annotations_df=ann,
        vocab=vocab,
        top_n_features=args.top_n_features,
        include_all_annotated=args.include_all_annotated,
    )

    # ── Step 6: Output ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "motif_registry.json"
    parquet_path = output_dir / "motif_registry.parquet"

    logger.info("Step 6: Writing outputs...")

    # JSON (without full sequences for readability — sequences in parquet)
    # Actually keep sequences in JSON too for self-contained lookup
    with open(json_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("  Written: %s (%.1f MB)", json_path, json_path.stat().st_size / 1e6)

    # Parquet (flat table)
    flat_df = registry_to_flat_df(registry)
    flat_df.to_parquet(parquet_path, index=False)
    logger.info("  Written: %s (%d rows)", parquet_path, len(flat_df))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("MOTIF REGISTRY SUMMARY")
    print("=" * 70)
    print(f"Total motifs:          {registry['n_motifs']}")
    print(f"With sequences:        {registry['n_with_sequences']}")
    print(f"In training vocab:     {registry['n_in_training_vocab']}")
    print(f"Token→UUID mappings:   {len(registry['token_to_uuid'])}")
    print(f"Sseqid→UUID mappings:  {len(registry['sseqid_to_uuid'])}")

    # Show top features
    print(f"\nTop-10 motifs by plasmid count:")
    top_motifs = sorted(
        registry["motifs"].values(),
        key=lambda m: m["plasmid_count"],
        reverse=True,
    )[:10]
    for m in top_motifs:
        n_seq = sum(1 for s in m["sequences"] if s["sequence"] is not None)
        print(
            f"  {m['plasmid_count']:>6,} plasmids  {m['feature_name']:<30}  "
            f"{m['token']:<35}  {n_seq} seq(s)  {'✓' if m['in_training_vocab'] else '✗'} vocab"
        )

    # Vocab coverage
    vocab_feat_tokens = {t for t in vocab if t.startswith("<FEAT_")}
    registry_tokens = set(registry["token_to_uuid"].keys())
    covered = vocab_feat_tokens & registry_tokens
    print(f"\nVocab coverage: {len(covered)}/{len(vocab_feat_tokens)} FEAT tokens in registry")
    missing = vocab_feat_tokens - registry_tokens
    if missing:
        print(f"  Missing: {sorted(missing)[:20]}")

    print(f"\nOutputs:")
    print(f"  {json_path}")
    print(f"  {parquet_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
