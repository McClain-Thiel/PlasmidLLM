#!/usr/bin/env python3
"""
normalize.py - Input normalization and metadata extraction for SPACE pipeline.

Extracts rich metadata from GenBank files and converts to FASTA for downstream tools.
Reuses logic from plasmid_pretraining.parsers for consistency.
"""

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# Resistance markers and their patterns (from constants.py)
RESISTANCE_MARKERS = {
    "ampicillin": ["ampicillin", "bla", "amp", "beta-lactam", "penicillin"],
    "kanamycin": ["kanamycin", "kan", "aph", "neomycin", "neo"],
    "chloramphenicol": ["chloramphenicol", "cat", "cm"],
    "tetracycline": ["tetracycline", "tet"],
    "spectinomycin": ["spectinomycin", "spec", "aad"],
    "streptomycin": ["streptomycin", "str", "aada"],
    "gentamicin": ["gentamicin", "gent", "aac"],
    "hygromycin": ["hygromycin", "hyg", "hph"],
    "puromycin": ["puromycin", "puro", "pac"],
    "blasticidin": ["blasticidin", "bsd", "bsr"],
    "zeocin": ["zeocin", "ble", "sh ble"],
    "erythromycin": ["erythromycin", "erm"],
    "trimethoprim": ["trimethoprim", "dfr", "dhfr"],
}

REPORTER_GENES = {
    "EGFP": ["egfp", "gfp", "green fluorescent"],
    "mCherry": ["mcherry", "cherry"],
    "mRFP": ["mrfp", "rfp", "red fluorescent"],
    "BFP": ["bfp", "blue fluorescent", "ebfp"],
    "YFP": ["yfp", "yellow fluorescent", "eyfp", "venus"],
    "CFP": ["cfp", "cyan fluorescent", "ecfp"],
    "tdTomato": ["tdtomato", "tomato"],
    "mVenus": ["mvenus"],
    "mCitrine": ["mcitrine", "citrine"],
    "luciferase": ["luciferase", "luc", "nanoluc", "fluc", "gluc"],
    "lacZ": ["lacz", "beta-galactosidase", "b-gal"],
}

PROTEIN_TAGS = {
    "his": ["his-tag", "his6", "6xhis", "histidine tag", "polyhistidine"],
    "flag": ["flag", "dykddddk"],
    "myc": ["myc", "c-myc"],
    "ha": ["ha-tag", "hemagglutinin"],
    "gst": ["gst", "glutathione"],
    "mbp": ["mbp", "maltose binding"],
    "strep": ["strep-tag", "streptavidin"],
    "v5": ["v5 tag", "v5-tag"],
    "sumo": ["sumo"],
    "t7": ["t7 tag"],
}

PLASMID_TYPES = {
    "mammalian_expression": ["mammalian", "hek", "cho", "hela", "cos", "cmv", "sv40", "ef1a", "cag", "pgk"],
    "bacterial_expression": ["bacterial", "e. coli", "ecoli", "t7", "plac", "ptac", "para", "ptrc"],
    "yeast_expression": ["yeast", "saccharomyces", "pichia", "gal1", "gal10", "adh1"],
    "insect_expression": ["insect", "baculovirus", "sf9", "sf21", "hi5"],
    "plant_expression": ["plant", "agrobacterium", "35s", "camv"],
    "lentiviral": ["lentivirus", "lentiviral", "ltr", "psi packaging"],
    "retroviral": ["retrovirus", "retroviral", "mmlv", "moloney"],
    "adenoviral": ["adenovirus", "adenoviral"],
    "aav": ["aav", "adeno-associated"],
    "crispr": ["crispr", "cas9", "cas12", "grna", "sgrna"],
    "cloning": ["cloning", "entry", "gateway", "topo", "ta cloning"],
    "shuttle": ["shuttle"],
}

COPY_NUMBER_MARKERS = {
    "high": ["pbr322", "puc", "colei", "pmb1", "high copy"],
    "medium": ["p15a", "medium copy"],
    "low": ["psc101", "f plasmid", "low copy", "single copy"],
}


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence) if sequence else 0.0


def calculate_seq_hash(sequence: str) -> str:
    """Calculate SHA256 hash of sequence for deduplication."""
    return hashlib.sha256(sequence.upper().encode()).hexdigest()


def match_keywords(text: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
    """Match text against keyword dictionary."""
    if not text:
        return []
    text_lower = text.lower()
    matches = []
    for category, keywords in keyword_dict.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                if category not in matches:
                    matches.append(category)
                break
    return matches


def extract_feature_info(feature) -> Dict[str, Any]:
    """Extract information from a BioPython SeqFeature."""
    qualifiers = dict(feature.qualifiers)
    return {
        "type": feature.type,
        "start": int(feature.location.start),
        "end": int(feature.location.end),
        "strand": feature.location.strand or 1,
        "gene": qualifiers.get("gene", [None])[0],
        "product": qualifiers.get("product", [None])[0],
        "locus_tag": qualifiers.get("locus_tag", [None])[0],
        "note": qualifiers.get("note", [None])[0] if "note" in qualifiers else None,
    }


def infer_plasmid_type(record: SeqRecord, features: List[Dict]) -> Optional[str]:
    """Infer plasmid type from features and annotations."""
    search_text = [
        record.description or "",
        str(record.annotations.get("keywords", [])),
        str(record.annotations.get("comment", "")),
    ]
    for feature in features:
        search_text.extend([feature.get("product") or "", feature.get("gene") or "", feature.get("note") or ""])
    combined_text = " ".join(search_text)
    matches = match_keywords(combined_text, PLASMID_TYPES)
    return matches[0] if matches else None


def infer_resistance_markers(features: List[Dict]) -> List[str]:
    """Infer resistance markers from CDS features."""
    markers = set()
    for feature in features:
        if feature.get("type") not in ["CDS", "gene"]:
            continue
        search_text = " ".join(filter(None, [
            feature.get("product"), feature.get("gene"),
            feature.get("note"), feature.get("locus_tag")
        ]))
        for marker in match_keywords(search_text, RESISTANCE_MARKERS):
            markers.add(marker)
    return list(markers)


def infer_reporter_genes(features: List[Dict]) -> List[str]:
    """Infer reporter genes from CDS features."""
    reporters = set()
    for feature in features:
        if feature.get("type") not in ["CDS", "gene"]:
            continue
        search_text = " ".join(filter(None, [
            feature.get("product"), feature.get("gene"), feature.get("note")
        ]))
        for reporter in match_keywords(search_text, REPORTER_GENES):
            reporters.add(reporter)
    return list(reporters)


def infer_tags(features: List[Dict]) -> List[str]:
    """Infer protein tags from CDS features."""
    tags = set()
    for feature in features:
        if feature.get("type") != "CDS":
            continue
        search_text = " ".join(filter(None, [
            feature.get("product"), feature.get("gene"), feature.get("note")
        ]))
        for tag in match_keywords(search_text, PROTEIN_TAGS):
            tags.add(tag)
    return list(tags)


def infer_copy_number(record: SeqRecord, features: List[Dict]) -> Optional[str]:
    """Infer copy number from origin of replication."""
    search_text = [record.description or ""]
    for feature in features:
        if feature.get("type") in ["rep_origin", "misc_feature"]:
            search_text.extend([feature.get("product") or "", feature.get("note") or ""])
    combined_text = " ".join(search_text)
    matches = match_keywords(combined_text, COPY_NUMBER_MARKERS)
    return matches[0] if matches else None


def extract_host(record: SeqRecord) -> Optional[str]:
    """Extract host from GenBank record."""
    # Check source feature for host qualifier
    for feature in record.features:
        if feature.type == "source":
            host = feature.qualifiers.get("host", [None])[0]
            if host:
                return host
            lab_host = feature.qualifiers.get("lab_host", [None])[0]
            if lab_host:
                return lab_host
    return None


def parse_genbank(filepath: Path) -> Dict[str, Any]:
    """Parse a GenBank file and extract all metadata."""
    with open(filepath, "r") as f:
        record = SeqIO.read(f, "genbank")

    sequence = str(record.seq).upper()

    # Extract features (skip source)
    features = []
    cds_count = 0
    gene_count = 0
    has_origin = False

    for feature in record.features:
        if feature.type == "source":
            continue
        feature_info = extract_feature_info(feature)
        features.append(feature_info)
        if feature.type == "CDS":
            cds_count += 1
        elif feature.type == "gene":
            gene_count += 1
        elif feature.type == "rep_origin":
            has_origin = True

    # Generate IDs
    record_id = str(uuid.uuid4())
    seq_hash = calculate_seq_hash(sequence)

    # Extract/infer metadata
    topology = record.annotations.get("topology", "linear")
    organism = record.annotations.get("organism", "")
    host = extract_host(record)
    plasmid_type = infer_plasmid_type(record, features)
    resistance_markers = infer_resistance_markers(features)
    reporter_genes = infer_reporter_genes(features)
    tags = infer_tags(features)
    copy_number = infer_copy_number(record, features)

    # Get all annotations
    annotations = {}
    for key, value in record.annotations.items():
        if isinstance(value, (str, int, float, bool)):
            annotations[key] = value
        elif isinstance(value, list):
            annotations[key] = [str(v) for v in value]
        else:
            annotations[key] = str(value)

    return {
        "id": record_id,
        "seq_hash": seq_hash,
        "original_id": record.id,
        "original_name": record.name,
        "filename": filepath.name,
        "length": len(sequence),
        "gc_content": calculate_gc_content(sequence),
        "topology": topology,
        "organism": organism,
        "description": record.description,
        "host": host,
        "plasmid_type": plasmid_type,
        "copy_number": copy_number,
        "resistance_markers": resistance_markers,
        "reporter_genes": reporter_genes,
        "tags": tags,
        "cds_count": cds_count,
        "gene_count": gene_count,
        "has_origin": has_origin,
        "genbank_features": features,
        "annotations": annotations,
        "sequence": sequence,
    }


def parse_fasta(filepath: Path) -> Dict[str, Any]:
    """Parse a FASTA file (minimal metadata)."""
    with open(filepath, "r") as f:
        record = SeqIO.read(f, "fasta")

    sequence = str(record.seq).upper()
    record_id = str(uuid.uuid4())
    seq_hash = calculate_seq_hash(sequence)

    return {
        "id": record_id,
        "seq_hash": seq_hash,
        "original_id": record.id,
        "original_name": record.name,
        "filename": filepath.name,
        "length": len(sequence),
        "gc_content": calculate_gc_content(sequence),
        "topology": None,  # Cannot determine from FASTA
        "organism": None,
        "description": record.description,
        "host": None,
        "plasmid_type": None,
        "copy_number": None,
        "resistance_markers": [],
        "reporter_genes": [],
        "tags": [],
        "cds_count": 0,
        "gene_count": 0,
        "has_origin": False,
        "genbank_features": [],
        "annotations": {},
        "sequence": sequence,
    }


def write_fasta(metadata: Dict[str, Any], output_path: Path):
    """Write sequence to FASTA file."""
    with open(output_path, "w") as f:
        f.write(f">{metadata['id']}\n")
        # Write sequence in 80-character lines
        seq = metadata["sequence"]
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")


def write_metadata(metadata: Dict[str, Any], output_path: Path):
    """Write metadata to JSON file (excluding sequence)."""
    meta_out = {k: v for k, v in metadata.items() if k != "sequence"}
    with open(output_path, "w") as f:
        json.dump(meta_out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Normalize input files and extract metadata")
    parser.add_argument("--input", required=True, help="Input FASTA/GenBank file")
    parser.add_argument("--output-fasta", required=True, help="Output FASTA file")
    parser.add_argument("--output-meta", required=True, help="Output metadata JSON file")

    args = parser.parse_args()

    # Handle Nextflow's backslash escaping of spaces in filenames
    input_str = args.input.replace("\\ ", " ")
    input_path = Path(input_str)
    output_fasta = Path(args.output_fasta)
    output_meta = Path(args.output_meta)

    # Determine file type and parse
    suffix = input_path.suffix.lower()
    if suffix in [".gb", ".gbk", ".genbank"]:
        metadata = parse_genbank(input_path)
    elif suffix in [".fasta", ".fa", ".fna"]:
        metadata = parse_fasta(input_path)
    else:
        # Try GenBank first, fall back to FASTA
        try:
            metadata = parse_genbank(input_path)
        except Exception:
            metadata = parse_fasta(input_path)

    # Write outputs
    write_fasta(metadata, output_fasta)
    write_metadata(metadata, output_meta)

    print(f"Processed {input_path.name}")
    print(f"  ID: {metadata['id']}")
    print(f"  Length: {metadata['length']} bp")
    print(f"  Topology: {metadata['topology']}")
    print(f"  Host: {metadata['host']}")
    print(f"  Type: {metadata['plasmid_type']}")


if __name__ == "__main__":
    main()
