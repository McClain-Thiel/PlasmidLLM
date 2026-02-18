#!/usr/bin/env python3
"""Generate training pairs from PlasmidKit annotations."""

import pandas as pd
from collections import defaultdict

BASE = "/mnt/s3/phd-research-storage-1758274488/addgene_clean"

TOKEN_CATEGORIES = [
    {
        "category": "copy",
        "source": "metadata",
        "field": "plasmid_copy",
        "tokens": {
            "<COPY_HIGH>": ["High Copy"],
            "<COPY_LOW>": ["Low Copy"],
            "<COPY_UNK>": ["Unknown"],
        },
    },
    {
        "category": "amr",
        "source": "metadata",
        "field": "bacterial_resistance",
        "tokens": {
            "<AMR_AMPICILLIN>": ["Ampicillin"],
            "<AMR_KANAMYCIN>": ["Kanamycin"],
            "<AMR_SPECTINOMYCIN>": ["Spectinomycin"],
            "<AMR_CHLORAMPHENICOL>": ["Chloramphenicol"],
            "<AMR_GENTAMICIN>": ["Gentamicin"],
            "<AMR_TETRACYCLINE>": ["Tetracycline"],
            "<AMR_ZEOCIN>": ["Bleocin"],
            "<AMR_APRAMYCIN>": ["Apramycin"],
            "<AMR_STREPTOMYCIN>": ["Streptomycin"],
            "<AMR_HYGROMYCIN>": ["Hygromycin"],
            "<AMR_PUROMYCIN>": ["Puromycin"],
            "<AMR_NEOMYCIN>": ["Neomycin"],
            "<AMR_BLASTICIDIN>": ["Blasticidin"],
            "<AMR_NOURSEOTHRICIN>": ["Nourseothricin"],
        },
        "multi_match": True,
        "other_token": "<AMR_OTHER>",
    },
    {
        "category": "ori",
        "source": "plasmidkit_annotations",
        "filter_type": "rep_origin",
        "tokens": {
            "<ORI_COLE1>": ["ColE1", "pBR322_origin", "pMB1"],
            "<ORI_F1>": ["f1_origin", "f1_ori", "f1_ori_2", "f1_ori_3"],
            "<ORI_SV40>": ["SV40_origin", "SV40_ori"],
            "<ORI_RSF>": ["RSF1030"],
            "<ORI_2MU>": ["2u_ori_2"],
            "<ORI_P15A>": ["p15A"],
            "<ORI_PSC101>": ["pSC101"],
        },
        "other_token": "<ORI_OTHER>",
    },
    {
        "category": "amr_marker",
        "source": "plasmidkit_annotations",
        "filter_type": "marker",
        "tokens": {
            "<AMR_AMPICILLIN>": ["TEM-116", "blaTEM", "Ampicillin", "AmpR_2"],
            "<AMR_KANAMYCIN>": ["APH", "NeoR", "KanR2"],
            "<AMR_PUROMYCIN>": ["puro"],
        },
        "other_token": "<AMR_OTHER>",
    },
    {
        "category": "prom",
        "source": "plasmidkit_annotations",
        "filter_type": "promoter",
        "tokens": {
            "<PROM_AMPR>": ["AmpR_promoter"],
            "<PROM_CMV>": ["CMV_immearly"],
            "<PROM_T7>": ["T7"],
            "<PROM_LAC>": ["lac_promoter"],
            "<PROM_T3>": ["T3_promoter"],
            "<PROM_SV40>": ["SV40_promoter"],
            "<PROM_RSV>": ["RSV_promoter"],
            "<PROM_SP6>": ["Sp6_promoter"],
        },
        "other_token": "<PROM_OTHER>",
    },
]


def gc_content(seq):
    if seq is None:
        return 0.5
    seq_str = str(seq)
    if len(seq_str) == 0:
        return 0.5
    gc = seq_str.count('G') + seq_str.count('C')
    return gc / len(seq_str)


def match_category(pid, cat_def, meta_idx, annot_by_plasmid):
    source = cat_def.get("source", "")
    token_defs = cat_def["tokens"]
    other_token = cat_def.get("other_token")
    matched = []
    
    if source == "metadata":
        field = cat_def["field"]
        if pid not in meta_idx.index:
            return []
        raw = meta_idx.at[pid, field]
        if pd.isna(raw):
            return []
        for tok, match_vals in token_defs.items():
            for mv in match_vals:
                if mv in str(raw):
                    matched.append(tok)
                    break
    elif source == "plasmidkit_annotations":
        filter_type = cat_def["filter_type"]
        rows = [r for r in annot_by_plasmid.get(pid, []) if r.Type == filter_type]
        features_present = {r.Feature for r in rows}
        for tok, match_vals in token_defs.items():
            for mv in match_vals:
                if any(mv in f for f in features_present):
                    matched.append(tok)
                    break
    if not matched and other_token:
        matched.append(other_token)
    return matched


def main():
    print("Loading data...")
    meta = pd.read_parquet(f"{BASE}/metadata/plasmid_metadata.parquet")
    meta_idx = meta.set_index("id")
    print(f"  Metadata: {len(meta)} rows")
    
    seq = pd.read_parquet(f"{BASE}/sequences/full_sequences.parquet", columns=["plasmid_id", "sequence", "sequence_length"])
    seq_idx = seq.set_index("plasmid_id")
    print(f"  Sequences: {len(seq)} rows")
    
    annot = pd.read_parquet(f"{BASE}/annotations/plasmidkit_annotations.parquet", columns=["plasmid_id", "Feature", "Type"])
    print(f"  PlasmidKit annotations: {len(annot)} rows")
    
    annot_by_plasmid = defaultdict(list)
    for row in annot.itertuples():
        annot_by_plasmid[row.plasmid_id].append(row)
    print(f"  Plasmids with annotations: {len(annot_by_plasmid)}")
    
    print("Generating training pairs...")
    training_pairs = []
    
    for pid in seq_idx.index:
        tokens = []
        for cat_def in TOKEN_CATEGORIES:
            matched = match_category(pid, cat_def, meta_idx, annot_by_plasmid)
            tokens.extend(matched)
        
        row = seq_idx.loc[pid]
        sequence = str(row["sequence"]) if isinstance(row, pd.Series) else str(row)
        seq_len = int(row["sequence_length"]) if isinstance(row, pd.Series) else int(seq_idx.at[pid, "sequence_length"])
        
        gc = gc_content(sequence)
        if gc < 0.45:
            tokens.append("<GC_LOW>")
        elif gc > 0.55:
            tokens.append("<GC_HIGH>")
        else:
            tokens.append("<GC_MED>")
            
        if seq_len < 5000:
            tokens.append("<SIZE_SMALL>")
        elif seq_len > 10000:
            tokens.append("<SIZE_LARGE>")
        else:
            tokens.append("<SIZE_MED>")
        
        prompt = " ".join(tokens) + " <SEP>"
        
        training_pairs.append({
            "plasmid_id": pid,
            "token_prompt": prompt,
            "num_tokens": len(tokens),
            "sequence": sequence,
            "sequence_length": seq_len,
        })
        
        if len(training_pairs) % 10000 == 0:
            print(f"  Processed {len(training_pairs)}...")
    
    df = pd.DataFrame(training_pairs)
    output_path = f"{BASE}/tokenization/plasmidkit_training_pairs.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"Saved {len(df)} training pairs")
    print(f"Avg tokens: {df['num_tokens'].mean():.1f}")
    print(f"Sample: {df.iloc[0]['token_prompt'][:80]}...")


if __name__ == "__main__":
    main()
