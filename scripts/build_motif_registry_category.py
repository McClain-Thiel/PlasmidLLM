#!/usr/bin/env python3
"""Build motif registry matching category-based training pairs WITH sequences."""

import argparse
import gzip
import html
import json
import logging
import re
import subprocess
import uuid
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BASE = "/mnt/s3/phd-research-storage-1758274488/addgene_clean"
MOTIF_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0d02b2c3d479")

MIN_PERCMATCH = 95.0
EXCLUDE_FRAGMENTS = True

# ── Token Mappings (match training pairs exactly) ────────────────────────────

AMR_TOKEN_MAP = {
    "<AMR_AMPICILLIN>": ["AmpR", "blaTEM", "TEM-116", "TEM-171", "TEM-181", "Ampicillin", "beta-lactamase"],
    "<AMR_KANAMYCIN>": ["KanR", "kanMX", "nptII", "APH(3')", "Kanamycin", "NeoR/KanR"],
    "<AMR_SPECTINOMYCIN>": ["SpecR", "spectinomycin", "aadA"],
    "<AMR_CHLORAMPHENICOL>": ["CmR", "CAT", "cat", "Chloramphenicol", "CamR"],
    "<AMR_GENTAMICIN>": ["GentR", "gentamicin", "aacC1", "aac(3)-Ia", "GmR"],
    "<AMR_TETRACYCLINE>": ["TetR", "tetA", "tetM", "tetQ", "tetC", "tetL", "tetX", "Tetracycline", "TcR"],
    "<AMR_ZEOCIN>": ["BleoR", "zeocin", "bleomycin"],
    "<AMR_APRAMYCIN>": ["ApraR", "apramycin", "AAC(3)-IV", "AAC6"],
    "<AMR_STREPTOMYCIN>": ["StrR", "streptomycin", "aadA"],
    "<AMR_HYGROMYCIN>": ["HygR", "hygromycin", "Hygromycin"],
    "<AMR_PUROMYCIN>": ["PuroR", "puro", "Puromycin"],
    "<AMR_NEOMYCIN>": ["NeoR", "neomycin", "Neomycin", "neo"],
    "<AMR_BLASTICIDIN>": ["BSD", "bsr", "blasticidin", "Blasticidin"],
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
    "<PROM_EF1A>": ["EF-1α promoter", "EF-1α core promoter", "EF-1 promoter", "EF_1"],
    "<PROM_RSV>": ["RSV promoter", "RSV LTR"],
    "<PROM_SP6>": ["SP6 promoter"],
    "<PROM_CAG>": ["chicken β-actin promoter", "CAG promoter", "CBA promoter"],
    "<PROM_T3>": ["T3 promoter"],
}

ORI_TOKEN_MAP = {
    "<ORI_F1>": ["f1 ori", "f1_origin", "f1 phage ori", "F1 ori"],
    "<ORI_SV40>": ["SV40 ori", "SV40_origin", "SV40 early ori"],
    "<ORI_RSF>": ["RSF1010", "RSF ori", "RSF1010 oriV", "RSF1010 oriT"],
    "<ORI_2MU>": ["2μ ori", "2u ori", "2 micron ori"],
    "<ORI_P15A>": ["p15A ori", "p15A_origin"],
    "<ORI_PSC101>": ["pSC101 ori", "pSC101_origin"],
    "<ORI_COLE1>": ["ori"],  # generic "ori" = ColE1
}

ELEM_TOKEN_MAP = {
    "<ELEM_AAV_ITR>": ["AAV ITR", "AAV2 ITR", "adeno-associated virus ITR"],
    "<ELEM_CMV_ENHANCER>": ["CMV enhancer", "CMV IE enhancer", "hr5 enhancer"],
    "<ELEM_CMV_INTRON>": ["CMV intron", "CMV IE intron", "CMV intron A"],
    "<ELEM_CPPT>": ["cPPT", "central polypurine tract", "CPPT/CTS"],
    "<ELEM_GRNA_SCAFFOLD>": ["gRNA scaffold", "sgRNA scaffold", "tracrRNA scaffold", "Nm gRNA scaffold", "Fn gRNA scaffold", "Sa gRNA scaffold"],
    "<ELEM_IRES>": ["IRES", "internal ribosome entry site", "EMCV IRES", "IRES Picorna 2", "IRES Cx43", "IRES HIF1", "IRES HCV", "FMDV IRES", "IRES Kv1 4", "IRES VEGF A", "IRES Picorna", "IRES Hsp70", "IRES c-myc", "IRES2", "IRES KSHV", "IRES Bip", "IRES EBNA", "IRES APC"],
    "<ELEM_LTR_3>": ["3' LTR", "3' long terminal repeat", "LTR 3'", "3' LTR (ΔU3)"],
    "<ELEM_LTR_5>": ["5' LTR", "5' long terminal repeat", "LTR 5'", "5' LTR (truncated)"],
    "<ELEM_MCS>": ["MCS", "multiple cloning site", "polylinker", "multiple cloning region"],
    "<ELEM_POLYA_BGH>": ["bGH poly(A)", "bovine growth hormone polyA", "BGH pA"],
    "<ELEM_POLYA_SV40>": ["SV40 poly(A)", "SV40 late polyA", "SV40 pA signal", "SV40 poly(A) signal"],
    "<ELEM_PSI>": ["psi", "Ψ", "packaging signal", "HIV packaging signal", "HIV-1 Ψ", "MESV Ψ", "MMLV Ψ", "Ad5 Ψ"],
    "<ELEM_TRACRRNA>": ["tracrRNA", "trans-activating crRNA", "tracr", "Nm tracrRNA"],
    "<ELEM_WPRE>": ["WPRE", "woodchuck hepatitis virus posttranscriptional regulatory element"],
}

REPORTER_TOKEN_MAP = {
    "<REPORTER_EGFP>": ["EGFP", "enhanced GFP", "eGFP", "aceGFP-h", "aceGFP", "rsEGFP2", "deGFP4", "d1EGFP", "cEGFP", "(3-F)Tyr-EGFP", "yEGFP", "mEGFP", "d4EGFP", "rsEGFP", "yeGFP", "d2EGFP"],
    "<REPORTER_GFP>": ["GFP", "green fluorescent protein", "roGFP2", "GFP (S65T)", "daGFP", "GFP11", "GFP(1-10)", "avGFP", "superfolder GFP", "AausGFP", "CGFP", "PA-GFP", "cfSGFP2", "SiriusGFP", "Folding Reporter GFP", "GFP nanobody", "GFPmut3", "mPA-GFP", "Trp-less GFP", "TurboGFP", "muGFP", "hrGFP", "GFPL_CLASP", "GFPhal", "NowGFP", "Cycle 3 GFP", "oxGFP", "moxGFP", "GFPxm163", "Enhanced Cyan-Emitting GFP", "GFP11x7", "&alpha;GFP", "SGFP2(206A)", "Superfolder GFP", "GFPuv", "SGFP2", "msGFP2", "SGFP2(T65G)", "GFPmut2", "EmGFP", "AcGFP1", "ppluGFP2", "vsfGFP-9", "mgfp5", "TagGFP", "TagGFP2"],
    "<REPORTER_MCHERRY>": ["mCherry", "mcherry", "monomeric Cherry", "PAmCherry2", "mCherry2C", "LSSmCherry1", "RDSmCherry0.1", "PAmCherry", "PAmCherry3", "PAmCherry1", "mCherry2"],
    "<REPORTER_MEMERALD>": ["mEmerald", "memerald"],
    "<REPORTER_NANOLUC>": ["NanoLuc", "nanoluciferase", "Nluc"],
    "<REPORTER_YFP>": ["YFP", "yellow fluorescent protein", "LanYFP", "Citrine", "Topaz YFP", "phiYFP", "d2EYFP", "mCitrine", "PhiYFP", "EYFP", "EYFP-Q69K", "Citrine2", "cpCitrine", "TagYFP", "mEYFP", "dLanYFP", "SYFP2"],
}

TAG_TOKEN_MAP = {
    "<TAG_FLAG>": ["FLAG", "FLAG tag", "DYKDDDDK", "2xFLAG", "3xFLAG", "5xFLAG"],
    "<TAG_GST>": ["GST", "glutathione S-transferase", "GST tag", "Gstz1", "Gsta2", "GST26_SCHJA", "GSTP1", "MGST3"],
    "<TAG_HA>": ["HA", "HA tag", "hemagglutinin tag", "YPYDVPDYA", "3xHA"],
    "<TAG_HIS>": ["His", "6xHis", "His tag", "hexahistidine", "HIS6", "10xHis", "7xHis", "8xHis", "9xHis"],
    "<TAG_MYC>": ["Myc", "c-Myc", "Myc tag", "EQKLISEEDL", "3xMyc", "13xMyc", "c-myc NLS"],
    "<TAG_NLS>": ["NLS", "nuclear localization signal", "nuclear localization sequence", "SV40 NLS", "nucleoplasmin NLS", "Rex NLS", "EGL-13 NLS"],
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


def feature_to_category_token(feature: str) -> tuple[str, str] | None:
    """Map feature to (token, category) or None if no match."""
    feat_lower = feature.lower()
    
    # Special case: AmpR promoter vs AmpR
    if feat_lower == "ampr promoter" or feat_lower == "ampicillin resistance promoter":
        return "<PROM_AMPR>", "PROM"
    
    # Try each category
    for category, token_map in ALL_TOKEN_MAPS.items():
        for token, patterns in token_map.items():
            for pattern in patterns:
                if pattern.lower() in feat_lower:
                    return token, category
    
    return None


def load_plannotate_metadata():
    """Load metadata from plannotate CSVs."""
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
    return meta


def _parse_fasta(text, db_source=""):
    """Parse FASTA, handling swissprot sp|ACC|ID format."""
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


def extract_sequences(db_dir, needed_sseqids):
    """Extract sequences from BLAST/Diamond/Infernal databases."""
    db_dir = Path(db_dir)
    all_seqs = []
    
    # snapgene
    snapgene_db = db_dir / "snapgene"
    if snapgene_db.with_suffix(".nsq").exists() or snapgene_db.with_suffix(".ndb").exists():
        try:
            result = subprocess.run(
                ["blastdbcmd", "-db", str(snapgene_db), "-entry", "all"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                for sid, seq in _parse_fasta(result.stdout, "snapgene"):
                    all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "dna", "db_source": "snapgene"})
                logger.info(f"  snapgene: {sum(1 for s in all_seqs if s['db_source'] == 'snapgene')} sequences")
        except Exception as e:
            logger.warning(f"snapgene failed: {e}")
    
    # fpbase
    fpbase_dmnd = db_dir / "fpbase.dmnd"
    if fpbase_dmnd.exists():
        try:
            result = subprocess.run(
                ["diamond", "getseq", "-d", str(fpbase_dmnd)],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                for sid, seq in _parse_fasta(result.stdout, "fpbase"):
                    all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "protein", "db_source": "fpbase"})
                logger.info(f"  fpbase: {sum(1 for s in all_seqs if s['db_source'] == 'fpbase')} sequences")
        except Exception as e:
            logger.warning(f"fpbase failed: {e}")
    
    # swissprot (filtered)
    swissprot_dmnd = db_dir / "swissprot.dmnd"
    if swissprot_dmnd.exists():
        try:
            result = subprocess.run(
                ["diamond", "getseq", "-d", str(swissprot_dmnd)],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                n_total = 0
                n_kept = 0
                for sid, seq in _parse_fasta(result.stdout, "swissprot"):
                    n_total += 1
                    if sid in needed_sseqids:
                        n_kept += 1
                        all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "protein", "db_source": "swissprot"})
                logger.info(f"  swissprot: {n_kept}/{n_total} sequences (filtered)")
        except Exception as e:
            logger.warning(f"swissprot failed: {e}")
    
    # Rfam
    rfam_cm = db_dir / "Rfam.cm"
    if rfam_cm.exists():
        try:
            result = subprocess.run(
                ["cmemit", "-c", str(rfam_cm)],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                for sid, seq in _parse_fasta(result.stdout, "Rfam"):
                    all_seqs.append({"sseqid": sid, "sequence": seq, "seq_type": "rna_consensus", "db_source": "Rfam"})
                logger.info(f"  Rfam: {sum(1 for s in all_seqs if s['db_source'] == 'Rfam')} sequences")
        except Exception as e:
            logger.warning(f"Rfam failed: {e}")
    
    return pd.DataFrame(all_seqs)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=f"{BASE}/annotations/plannotate_annotations.parquet")
    parser.add_argument("--db-dir", default="/opt/dlami/nvme/miniconda3/pkgs/plannotate-1.2.4-pyhdfd78af_0/site-packages/plannotate/data/BLAST_dbs")
    parser.add_argument("--output-dir", default=f"{BASE}/tokenization/")
    args = parser.parse_args()
    
    logger.info("Loading annotations...")
    annot = pd.read_parquet(args.annotations)
    logger.info(f"  Loaded {len(annot):,} annotations")
    
    # Quality filter
    annot = annot[annot["percmatch"] >= MIN_PERCMATCH]
    if EXCLUDE_FRAGMENTS:
        annot = annot[~annot["fragment"].fillna(False).astype(bool)]
    logger.info(f"  After quality filter: {len(annot):,}")
    
    # Map features to tokens
    logger.info("Mapping features to category tokens...")
    annot["mapped"] = annot["Feature"].apply(feature_to_category_token)
    annot_mapped = annot[annot["mapped"].notna()].copy()
    annot_mapped["token"] = annot_mapped["mapped"].apply(lambda x: x[0])
    annot_mapped["category"] = annot_mapped["mapped"].apply(lambda x: x[1])
    logger.info(f"  Mapped {len(annot_mapped):,}/{len(annot):,} annotations")
    
    # Get needed sseqids
    needed_sseqids = set(annot_mapped["sseqid"].unique())
    logger.info(f"  Need sequences for {len(needed_sseqids)} sseqids")
    
    # Extract sequences
    logger.info("Extracting sequences from databases...")
    seq_df = extract_sequences(args.db_dir, needed_sseqids)
    seq_lookup = {row["sseqid"]: row for _, row in seq_df.iterrows()}
    logger.info(f"  Total sequences extracted: {len(seq_lookup)}")
    
    # Group by token
    logger.info("Building motif registry...")
    token_groups = annot_mapped.groupby("token")
    
    motifs = {}
    token_to_uuid = {}
    rows = []
    
    for token, group in token_groups:
        motif_uuid = str(uuid.uuid5(MOTIF_NAMESPACE, token))
        token_to_uuid[token] = motif_uuid
        
        features = group["Feature"].unique().tolist()
        plasmid_count = group["plasmid_id"].nunique()
        category = group["category"].iloc[0]
        
        # Build sequence entries
        sequences = []
        for sseqid in group["sseqid"].unique():
            if sseqid in seq_lookup:
                seq_info = seq_lookup[sseqid]
                sequences.append({
                    "sseqid": sseqid,
                    "sequence": seq_info["sequence"],
                    "seq_type": seq_info["seq_type"],
                    "db_source": seq_info["db_source"],
                    "seq_len": len(seq_info["sequence"]) if seq_info["sequence"] else None,
                })
                rows.append({
                    "uuid": motif_uuid,
                    "token": token,
                    "category": category,
                    "features": ",".join(features),
                    "plasmid_count": int(plasmid_count),
                    "sseqid": sseqid,
                    "db_source": seq_info["db_source"],
                    "seq_type": seq_info["seq_type"],
                    "seq_len": len(seq_info["sequence"]) if seq_info["sequence"] else None,
                    "sequence": seq_info["sequence"],
                })
            else:
                rows.append({
                    "uuid": motif_uuid,
                    "token": token,
                    "category": category,
                    "features": ",".join(features),
                    "plasmid_count": int(plasmid_count),
                    "sseqid": sseqid,
                    "db_source": None,
                    "seq_type": None,
                    "seq_len": None,
                    "sequence": None,
                })
        
        motifs[motif_uuid] = {
            "uuid": motif_uuid,
            "token": token,
            "category": category,
            "features": features,
            "plasmid_count": int(plasmid_count),
            "sequences": sequences,
        }
    
    logger.info(f"Created {len(motifs)} motifs with {len([r for r in rows if r['sequence'] is not None])} sequences")
    
    # Summary
    by_cat = defaultdict(lambda: {"tokens": 0, "with_seq": 0})
    for m in motifs.values():
        has_seq = any(s.get("sequence") for s in m["sequences"])
        by_cat[m["category"]]["tokens"] += 1
        if has_seq:
            by_cat[m["category"]]["with_seq"] += 1
    
    logger.info("By category:")
    for cat, counts in sorted(by_cat.items()):
        logger.info(f"  {cat}: {counts['tokens']} tokens, {counts['with_seq']} with sequences")
    
    # Save parquet
    df = pd.DataFrame(rows)
    output_path = Path(args.output_dir) / "motif_registry.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")
    
    # Save JSON
    json_output = Path(args.output_dir) / "motif_registry.json"
    with open(json_output, 'w') as f:
        json.dump({
            "version": "2.0",
            "n_motifs": len(motifs),
            "motifs": motifs,
            "token_to_uuid": token_to_uuid,
        }, f, indent=2)
    logger.info(f"Saved JSON to {json_output}")


if __name__ == "__main__":
    main()
