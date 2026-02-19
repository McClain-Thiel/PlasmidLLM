#!/usr/bin/env python3
"""Build motif registry matching category-based training pairs (no FEAT tokens).

Maps plannotate features directly to category tokens (AMR, PROM, ORI, ELEM, REPORTER, TAG)
instead of generic FEAT tokens.
"""

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

# Combine all
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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=f"{BASE}/annotations/plannotate_annotations.parquet")
    parser.add_argument("--db-dir", default="~/.cache/pLannotate/BLAST_dbs")
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
    
    # Keep only mapped features
    annot_mapped = annot[annot["mapped"].notna()].copy()
    annot_mapped["token"] = annot_mapped["mapped"].apply(lambda x: x[0])
    annot_mapped["category"] = annot_mapped["mapped"].apply(lambda x: x[1])
    
    logger.info(f"  Mapped {len(annot_mapped):,}/{len(annot):,} annotations to category tokens")
    
    # Group by token and collect metadata
    token_groups = annot_mapped.groupby("token")
    
    motifs = {}
    token_to_uuid = {}
    sseqid_to_uuid = {}
    
    for token, group in token_groups:
        motif_uuid = str(uuid.uuid5(MOTIF_NAMESPACE, token))
        token_to_uuid[token] = motif_uuid
        
        # Collect unique features and sseqids for this token
        features = group["Feature"].unique().tolist()
        sseqids = group["sseqid"].unique().tolist()
        plasmid_count = group["plasmid_id"].nunique()
        
        # Create sequences entries (will be populated later if DBs available)
        sequences = [{"sseqid": sid, "sequence": None} for sid in sseqids]
        for sid in sseqids:
            sseqid_to_uuid[sid] = motif_uuid
        
        category = group["category"].iloc[0]
        
        motifs[motif_uuid] = {
            "uuid": motif_uuid,
            "token": token,
            "category": category,
            "features": features,
            "plasmid_count": int(plasmid_count),
            "sequences": sequences,
        }
    
    logger.info(f"Created {len(motifs)} motif entries")
    
    # Summary
    by_cat = defaultdict(int)
    for m in motifs.values():
        by_cat[m["category"]] += 1
    
    logger.info("By category:")
    for cat, count in sorted(by_cat.items()):
        logger.info(f"  {cat}: {count}")
    
    # Create flat dataframe
    rows = []
    for motif_uuid, motif in motifs.items():
        for seq_entry in motif["sequences"]:
            rows.append({
                "uuid": motif_uuid,
                "token": motif["token"],
                "category": motif["category"],
                "features": ",".join(motif["features"]),
                "plasmid_count": motif["plasmid_count"],
                "sseqid": seq_entry["sseqid"],
                "sequence": seq_entry["sequence"],
            })
    
    df = pd.DataFrame(rows)
    
    # Save
    output_path = Path(args.output_dir) / "motif_registry_category.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")
    
    # Save JSON
    json_output = Path(args.output_dir) / "motif_registry_category.json"
    with open(json_output, 'w') as f:
        json.dump({
            "version": "1.0",
            "n_motifs": len(motifs),
            "motifs": motifs,
            "token_to_uuid": token_to_uuid,
            "sseqid_to_uuid": sseqid_to_uuid,
        }, f, indent=2)
    logger.info(f"Saved JSON to {json_output}")


if __name__ == "__main__":
    main()
