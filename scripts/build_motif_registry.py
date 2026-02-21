#!/usr/bin/env python3
"""Build a motif registry mapping categorical tokens to canonical reference sequences.

Maps plannotate annotations to the categorical tokens used in training
(<AMR_*>, <PROM_*>, <ORI_*>, <ELEM_*>, <REPORTER_*>, <TAG_*>), then extracts
canonical sequences from plannotate's BLAST/Diamond/Infernal databases.

Output:
    motif_registry.json   — nested dict keyed by UUID
    motif_registry.parquet — flat table (one row per token+sseqid pair)
"""

import argparse
import html
import json
import logging
import re
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BASE = "/mnt/s3/phd-research-storage-1758274488/addgene_clean"
MOTIF_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0d02b2c3d479")

MIN_PERCMATCH = 95.0
EXCLUDE_FRAGMENTS = True

# ── Token maps: plannotate Feature name → categorical token ──────────────────
# These define how plannotate's Feature field maps to training tokens.
# Pattern matching: if any pattern appears as substring in the Feature name,
# the token is assigned. Order within a map matters for priority.

AMR_TOKEN_MAP = {
    "<AMR_AMPICILLIN>": ["AmpR", "blaTEM", "TEM-116", "TEM-171", "TEM-181", "beta-lactamase"],
    "<AMR_KANAMYCIN>": ["KanR", "kanMX", "nptII", "APH(3')", "NeoR/KanR"],
    "<AMR_SPECTINOMYCIN>": ["SpecR", "spectinomycin", "aadA"],
    "<AMR_CHLORAMPHENICOL>": ["CmR", "CAT", "cat", "CamR"],
    "<AMR_GENTAMICIN>": ["GentR", "gentamicin", "aacC1", "aac(3)-Ia", "GmR"],
    "<AMR_TETRACYCLINE>": ["TetR", "tetA", "tetM", "tetQ", "tetC", "tetL", "tetX", "TcR"],
    "<AMR_ZEOCIN>": ["BleoR", "zeocin", "bleomycin"],
    "<AMR_APRAMYCIN>": ["ApraR", "apramycin", "AAC(3)-IV", "AAC6"],
    "<AMR_STREPTOMYCIN>": ["StrR", "streptomycin"],
    "<AMR_HYGROMYCIN>": ["HygR", "hygromycin"],
    "<AMR_PUROMYCIN>": ["PuroR", "puro"],
    "<AMR_NEOMYCIN>": ["NeoR", "neomycin", "neo"],
    "<AMR_BLASTICIDIN>": ["BSD", "bsr", "blasticidin"],
    "<AMR_NOURSEOTHRICIN>": ["NatR", "nourseothricin"],
}

PROM_TOKEN_MAP = {
    "<PROM_AMPR>": ["AmpR promoter"],
    "<PROM_CMV>": ["CMV promoter", "CMV IE94 promoter", "mCMV promoter", "CMV IE", "CMV_immearly"],
    "<PROM_T5>": ["T5 promoter"],
    "<PROM_SV40>": ["SV40 promoter", "SV40 early promoter"],
    "<PROM_LAC>": ["lac promoter", "tac promoter", "lac UV5 promoter", "lacI promoter"],
    "<PROM_T7>": ["T7 promoter"],
    "<PROM_U6>": ["U6 promoter", "AtU6 promoter", "CeU6 promoter"],
    "<PROM_EF1A>": ["EF-1\u03b1 promoter", "EF-1\u03b1 core promoter", "EF-1 promoter", "EF_1"],
    "<PROM_RSV>": ["RSV promoter", "RSV LTR"],
    "<PROM_SP6>": ["SP6 promoter"],
    "<PROM_CAG>": ["chicken \u03b2-actin promoter", "CAG promoter", "CBA promoter"],
    "<PROM_T3>": ["T3 promoter"],
}

ORI_TOKEN_MAP = {
    "<ORI_F1>": ["f1 ori", "f1_origin", "f1 phage ori", "F1 ori"],
    "<ORI_SV40>": ["SV40 ori", "SV40_origin", "SV40 early ori"],
    "<ORI_RSF>": ["RSF1010", "RSF ori", "RSF1010 oriV", "RSF1010 oriT"],
    "<ORI_2MU>": ["2\u03bc ori", "2u ori", "2 micron ori"],
    "<ORI_P15A>": ["p15A ori", "p15A_origin"],
    "<ORI_PSC101>": ["pSC101 ori", "pSC101_origin"],
    "<ORI_COLE1>": ["ColE1", "pBR322", "pMB1", "ori"],
}

ELEM_TOKEN_MAP = {
    "<ELEM_AAV_ITR>": ["AAV ITR", "AAV2 ITR", "adeno-associated virus ITR"],
    "<ELEM_CMV_ENHANCER>": ["CMV enhancer", "CMV IE enhancer", "hr5 enhancer"],
    "<ELEM_CMV_INTRON>": ["CMV intron", "CMV IE intron", "CMV intron A"],
    "<ELEM_CPPT>": ["cPPT", "central polypurine tract", "CPPT/CTS"],
    "<ELEM_GRNA_SCAFFOLD>": ["gRNA scaffold", "sgRNA scaffold", "tracrRNA scaffold",
                             "Nm gRNA scaffold", "Fn gRNA scaffold", "Sa gRNA scaffold"],
    "<ELEM_IRES>": ["IRES", "internal ribosome entry site", "EMCV IRES"],
    "<ELEM_LTR_3>": ["3' LTR", "3' long terminal repeat", "LTR 3'", "3' LTR (\u0394U3)"],
    "<ELEM_LTR_5>": ["5' LTR", "5' long terminal repeat", "LTR 5'", "5' LTR (truncated)"],
    "<ELEM_MCS>": ["MCS", "multiple cloning site", "polylinker", "multiple cloning region"],
    "<ELEM_POLYA_BGH>": ["bGH poly(A)", "bovine growth hormone polyA", "BGH pA"],
    "<ELEM_POLYA_SV40>": ["SV40 poly(A)", "SV40 late polyA", "SV40 pA signal", "SV40 poly(A) signal"],
    "<ELEM_PSI>": ["psi", "\u03a8", "packaging signal", "HIV packaging signal",
                   "HIV-1 \u03a8", "MESV \u03a8", "MMLV \u03a8", "Ad5 \u03a8"],
    "<ELEM_TRACRRNA>": ["tracrRNA", "trans-activating crRNA", "tracr", "Nm tracrRNA"],
    "<ELEM_WPRE>": ["WPRE", "woodchuck hepatitis virus posttranscriptional regulatory element"],
}

REPORTER_TOKEN_MAP = {
    "<REPORTER_EGFP>": ["EGFP", "enhanced GFP", "eGFP", "mEGFP", "yEGFP", "yeGFP",
                        "rsEGFP2", "rsEGFP", "d1EGFP", "d2EGFP", "d4EGFP", "deGFP4",
                        "cEGFP", "aceGFP"],
    "<REPORTER_GFP>": ["GFP", "green fluorescent protein", "superfolder GFP",
                       "GFP (S65T)", "GFPuv", "TurboGFP", "TagGFP", "EmGFP",
                       "AcGFP1", "\u03b1GFP", "SGFP2", "roGFP2"],
    "<REPORTER_MCHERRY>": ["mCherry", "PAmCherry"],
    "<REPORTER_MEMERALD>": ["mEmerald"],
    "<REPORTER_NANOLUC>": ["NanoLuc", "nanoluciferase", "Nluc"],
    "<REPORTER_YFP>": ["YFP", "yellow fluorescent protein", "Citrine", "EYFP",
                       "LanYFP", "SYFP2", "TagYFP", "PhiYFP"],
}

TAG_TOKEN_MAP = {
    "<TAG_FLAG>": ["FLAG", "DYKDDDDK", "2xFLAG", "3xFLAG", "5xFLAG"],
    "<TAG_GST>": ["GST", "glutathione S-transferase", "GST26_SCHJA"],
    "<TAG_HA>": ["HA tag", "hemagglutinin tag", "YPYDVPDYA", "3xHA"],
    "<TAG_HIS>": ["6xHis", "His tag", "hexahistidine", "HIS6", "10xHis"],
    "<TAG_MYC>": ["Myc", "c-Myc", "EQKLISEEDL", "3xMyc", "13xMyc"],
    "<TAG_NLS>": ["NLS", "nuclear localization signal", "SV40 NLS", "nucleoplasmin NLS"],
    "<TAG_V5>": ["V5", "V5 tag", "GKPIPNPLLGLDST"],
}

ALL_TOKEN_MAPS = {
    "AMR": AMR_TOKEN_MAP,
    "PROM": PROM_TOKEN_MAP,
    "ORI": ORI_TOKEN_MAP,
    "ELEM": ELEM_TOKEN_MAP,
    "REPORTER": REPORTER_TOKEN_MAP,
    "TAG": TAG_TOKEN_MAP,
}


# ── Feature → token mapping ─────────────────────────────────────────────────

def feature_to_category_token(feature: str) -> tuple[str, str] | None:
    """Map a plannotate Feature name to (token, category) or None."""
    feat_lower = feature.lower()

    # Special case: AmpR promoter → PROM, not AMR
    if "ampr promoter" in feat_lower or "ampicillin resistance promoter" in feat_lower:
        return "<PROM_AMPR>", "PROM"

    # Special case: CMV enhancer → ELEM, not PROM
    if "cmv enhancer" in feat_lower:
        return "<ELEM_CMV_ENHANCER>", "ELEM"

    # Try ELEM before PROM (CMV intron vs CMV promoter)
    for category in ["ELEM", "PROM", "ORI", "REPORTER", "TAG", "AMR"]:
        token_map = ALL_TOKEN_MAPS[category]
        for token, patterns in token_map.items():
            for pattern in patterns:
                pat_lower = pattern.lower()
                # Short patterns (<=4 chars): word boundary match
                if len(pat_lower) <= 4:
                    if re.search(r"\b" + re.escape(pat_lower) + r"\b", feat_lower):
                        return token, category
                else:
                    if pat_lower in feat_lower:
                        return token, category

    return None


# ── Plannotate metadata loading ──────────────────────────────────────────────

def load_plannotate_metadata() -> pd.DataFrame:
    """Load metadata from plannotate's bundled CSVs."""
    import plannotate
    data_dir = Path(plannotate.__file__).parent / "data" / "data"

    frames = []

    sg = pd.read_csv(data_dir / "snapgene.csv")
    sg["db_source"] = "snapgene"
    frames.append(sg)

    fp = pd.read_csv(data_dir / "fpbase.csv")
    fp["Type"] = "CDS"
    fp["db_source"] = "fpbase"
    for col in ["sseqid", "Feature", "Description"]:
        fp[col] = fp[col].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)
    frames.append(fp)

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


# ── Sequence extraction ──────────────────────────────────────────────────────

def _parse_fasta(text: str, db_source: str = "") -> list[tuple[str, str]]:
    """Parse FASTA text into [(id, sequence), ...].

    For swissprot diamond output, extracts accession from sp|ACC|ID headers.
    """
    entries = []
    current_id = None
    current_seq: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq)))
            header_id = line[1:].split()[0]
            if db_source == "swissprot" and "|" in header_id:
                parts = header_id.split("|")
                current_id = parts[1] if len(parts) >= 2 else header_id
            else:
                current_id = header_id
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


def extract_sequences(db_dir: Path, needed_sseqids: set[str]) -> pd.DataFrame:
    """Extract canonical sequences from plannotate's BLAST/Diamond/Infernal DBs."""
    all_seqs: list[dict] = []

    # snapgene (BLAST nucleotide DB)
    snapgene_db = db_dir / "snapgene"
    if snapgene_db.with_suffix(".nsq").exists() or snapgene_db.with_suffix(".ndb").exists():
        out = _run_cmd(["blastdbcmd", "-db", str(snapgene_db), "-entry", "all"], "snapgene")
        if out:
            for sid, seq in _parse_fasta(out, "snapgene"):
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "dna", "db_source": "snapgene"})
            logger.info("  snapgene: %d sequences", sum(1 for s in all_seqs if s["db_source"] == "snapgene"))
    else:
        logger.warning("snapgene BLAST DB not found at %s", snapgene_db)

    # fpbase (Diamond protein DB)
    fpbase_dmnd = db_dir / "fpbase.dmnd"
    if fpbase_dmnd.exists():
        out = _run_cmd(["diamond", "getseq", "-d", str(fpbase_dmnd)], "fpbase")
        if out:
            for sid, seq in _parse_fasta(out, "fpbase"):
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "protein", "db_source": "fpbase"})
            logger.info("  fpbase: %d sequences", sum(1 for s in all_seqs if s["db_source"] == "fpbase"))
    else:
        logger.warning("fpbase Diamond DB not found at %s", fpbase_dmnd)

    # swissprot (Diamond protein DB) — filter to needed sseqids
    swissprot_dmnd = db_dir / "swissprot.dmnd"
    if swissprot_dmnd.exists():
        out = _run_cmd(["diamond", "getseq", "-d", str(swissprot_dmnd)], "swissprot")
        if out:
            n_total = 0
            n_kept = 0
            for sid, seq in _parse_fasta(out, "swissprot"):
                n_total += 1
                if sid in needed_sseqids:
                    n_kept += 1
                    all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "protein", "db_source": "swissprot"})
            logger.info("  swissprot: %d/%d sequences (filtered)", n_kept, n_total)
    else:
        logger.warning("swissprot Diamond DB not found at %s", swissprot_dmnd)

    # Rfam (Infernal covariance model)
    rfam_cm = db_dir / "Rfam.cm"
    if rfam_cm.exists():
        out = _run_cmd(["cmemit", "-c", str(rfam_cm)], "Rfam")
        if out:
            for sid, seq in _parse_fasta(out, "Rfam"):
                all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "rna_consensus", "db_source": "Rfam"})
            logger.info("  Rfam: %d sequences", sum(1 for s in all_seqs if s["db_source"] == "Rfam"))
    else:
        logger.warning("Rfam CM not found at %s", rfam_cm)

    if not all_seqs:
        logger.warning("No sequences extracted from any database!")
        return pd.DataFrame(columns=["sseqid", "sequence", "seq_type", "db_source"])

    return pd.DataFrame(all_seqs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        default=f"{BASE}/annotations/plannotate_annotations.parquet",
    )
    parser.add_argument(
        "--vocab",
        default=f"{BASE}/tokenization/token_vocabulary_v3.json",
    )
    parser.add_argument(
        "--db-dir",
        default=str(Path.home() / ".cache" / "pLannotate" / "BLAST_dbs"),
        help="plannotate BLAST/Diamond DB directory",
    )
    parser.add_argument("--output-dir", default=f"{BASE}/tokenization/")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Skip sequence extraction")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load plannotate metadata CSVs ──
    logger.info("Loading plannotate metadata CSVs...")
    metadata_df = load_plannotate_metadata()

    # ── Load annotations ──
    logger.info("Loading plannotate annotations...")
    ann = pq.read_table(
        args.annotations,
        columns=["plasmid_id", "Feature", "sseqid", "db", "percmatch", "fragment"],
    ).to_pandas()
    ann["plasmid_id"] = ann["plasmid_id"].astype(str)
    logger.info("  %d annotations for %d plasmids", len(ann), ann["plasmid_id"].nunique())

    # Quality filter
    ann = ann[ann["percmatch"] >= MIN_PERCMATCH]
    if EXCLUDE_FRAGMENTS:
        ann = ann[~ann["fragment"].fillna(False).astype(bool)]
    logger.info("  After quality filter: %d", len(ann))

    # ── Map features → categorical tokens ──
    logger.info("Mapping features to categorical tokens...")
    ann["mapped"] = ann["Feature"].apply(feature_to_category_token)
    ann_mapped = ann[ann["mapped"].notna()].copy()
    ann_mapped["token"] = ann_mapped["mapped"].apply(lambda x: x[0])
    ann_mapped["category"] = ann_mapped["mapped"].apply(lambda x: x[1])

    n_unmapped = len(ann) - len(ann_mapped)
    logger.info(
        "  Mapped %d/%d annotations (%d unmapped)",
        len(ann_mapped), len(ann), n_unmapped,
    )

    # Show unmapped features for debugging
    if n_unmapped > 0:
        unmapped_feats = ann[ann["mapped"].isna()]["Feature"].value_counts().head(20)
        logger.info("  Top unmapped features:")
        for feat, cnt in unmapped_feats.items():
            logger.info("    %6d  %s", cnt, feat)

    # ── Extract sequences ──
    needed_sseqids = set(ann_mapped["sseqid"].unique())
    logger.info("Need sequences for %d unique sseqids", len(needed_sseqids))

    seq_lookup: dict[str, dict] = {}
    if not args.metadata_only:
        db_dir = Path(args.db_dir)
        if db_dir.exists():
            logger.info("Extracting sequences from %s...", db_dir)
            seq_df = extract_sequences(db_dir, needed_sseqids)
            for _, row in seq_df.iterrows():
                seq_lookup[row["sseqid"]] = {
                    "sequence": row["sequence"],
                    "seq_type": row["seq_type"],
                    "db_source": row["db_source"],
                }
            logger.info("  Total in lookup: %d", len(seq_lookup))
        else:
            logger.warning("DB dir not found: %s — skipping extraction", db_dir)
    else:
        logger.info("Skipping sequence extraction (--metadata-only)")

    # ── Load vocab ──
    logger.info("Loading vocab from %s...", args.vocab)
    with open(args.vocab) as f:
        vocab_data = json.load(f)
    if isinstance(vocab_data, dict) and "token_to_id" in vocab_data:
        vocab = vocab_data["token_to_id"]
    else:
        vocab = vocab_data

    # ── Build sseqid → metadata lookup ──
    meta_lookup: dict[str, dict] = {}
    for _, row in metadata_df.iterrows():
        meta_lookup[row["sseqid"]] = {
            "Feature": row["Feature"],
            "Type": row.get("Type", "CDS"),
            "Description": row.get("Description", ""),
            "db_source": row["db_source"],
        }

    # ── Build registry grouped by token ──
    logger.info("Building motif registry...")
    token_groups = ann_mapped.groupby("token")

    motifs: dict[str, dict] = {}
    token_to_uuid: dict[str, str] = {}
    flat_rows: list[dict] = []

    for token, group in token_groups:
        motif_uuid = str(uuid.uuid5(MOTIF_NAMESPACE, token))
        token_to_uuid[token] = motif_uuid

        features = sorted(group["Feature"].unique().tolist())
        plasmid_count = int(group["plasmid_id"].nunique())
        category = group["category"].iloc[0]

        # Build sequence entries for each unique sseqid
        sequences = []
        for sseqid in sorted(group["sseqid"].unique()):
            entry: dict = {"sseqid": sseqid}

            # Metadata (description, type)
            if sseqid in meta_lookup:
                entry["db_source"] = meta_lookup[sseqid]["db_source"]
                desc = meta_lookup[sseqid].get("Description", "")
                entry["description"] = str(desc) if desc and str(desc) != "nan" else ""
            else:
                # Infer db from annotation
                ann_dbs = group[group["sseqid"] == sseqid]["db"].unique()
                entry["db_source"] = ann_dbs[0] if len(ann_dbs) > 0 else "unknown"
                entry["description"] = ""

            # Sequence
            if sseqid in seq_lookup:
                entry["sequence"] = seq_lookup[sseqid]["sequence"]
                entry["seq_type"] = seq_lookup[sseqid]["seq_type"]
                entry["seq_len"] = len(seq_lookup[sseqid]["sequence"])
            else:
                entry["sequence"] = None
                entry["seq_type"] = None
                entry["seq_len"] = None

            sequences.append(entry)

            flat_rows.append({
                "uuid": motif_uuid,
                "token": token,
                "category": category,
                "features": ",".join(features),
                "plasmid_count": plasmid_count,
                "sseqid": sseqid,
                "db_source": entry["db_source"],
                "seq_type": entry.get("seq_type"),
                "seq_len": entry.get("seq_len"),
                "sequence": entry.get("sequence"),
            })

        # Pick best description (prefer snapgene)
        description = ""
        for db_pref in ["snapgene", "fpbase", "swissprot"]:
            for s in sequences:
                if s["db_source"] == db_pref and s.get("description"):
                    description = s["description"]
                    break
            if description:
                break

        motifs[motif_uuid] = {
            "uuid": motif_uuid,
            "token": token,
            "category": category,
            "features": features,
            "plasmid_count": plasmid_count,
            "description": description,
            "in_vocab": token in vocab,
            "sequences": sequences,
        }

    n_with_seq = sum(
        1 for m in motifs.values()
        if any(s["sequence"] is not None for s in m["sequences"])
    )
    logger.info("Registry: %d motifs (%d with sequences)", len(motifs), n_with_seq)

    # ── Save outputs ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "motif_registry.json"
    parquet_path = output_dir / "motif_registry.parquet"

    with open(json_path, "w") as f:
        json.dump({
            "version": "2.0",
            "namespace_uuid": str(MOTIF_NAMESPACE),
            "n_motifs": len(motifs),
            "n_with_sequences": n_with_seq,
            "motifs": motifs,
            "token_to_uuid": token_to_uuid,
        }, f, indent=2)
    logger.info("Written: %s (%.1f MB)", json_path, json_path.stat().st_size / 1e6)

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_parquet(parquet_path, index=False)
    logger.info("Written: %s (%d rows)", parquet_path, len(flat_df))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("MOTIF REGISTRY SUMMARY")
    print("=" * 70)
    print(f"Total motifs:        {len(motifs)}")
    print(f"With sequences:      {n_with_seq}")

    by_cat = defaultdict(lambda: {"n": 0, "with_seq": 0})
    for m in motifs.values():
        has_seq = any(s.get("sequence") for s in m["sequences"])
        by_cat[m["category"]]["n"] += 1
        if has_seq:
            by_cat[m["category"]]["with_seq"] += 1
    print("\nBy category:")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"  {cat:10s}  {c['n']:3d} tokens  {c['with_seq']:3d} with sequences")

    print(f"\nTop motifs by plasmid count:")
    for m in sorted(motifs.values(), key=lambda x: x["plasmid_count"], reverse=True)[:15]:
        n_seq = sum(1 for s in m["sequences"] if s["sequence"] is not None)
        print(
            f"  {m['plasmid_count']:>6,}  {m['token']:<30}  "
            f"{len(m['features']):>3} features  {n_seq} seq(s)  "
            f"{'V' if m['in_vocab'] else 'X'}"
        )

    # Vocab coverage
    seq_tokens_in_vocab = {t for t in vocab if any(t.startswith(f"<{p}_") for p in
                           ["AMR", "PROM", "ORI", "ELEM", "REPORTER", "TAG"])}
    covered = seq_tokens_in_vocab & set(token_to_uuid.keys())
    print(f"\nVocab coverage: {len(covered)}/{len(seq_tokens_in_vocab)} sequence tokens")
    missing = seq_tokens_in_vocab - set(token_to_uuid.keys())
    if missing:
        print(f"  Not in registry (OTHER/rare): {sorted(missing)}")

    print(f"\nOutputs:")
    print(f"  {json_path}")
    print(f"  {parquet_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
