#!/usr/bin/env python3
"""
normalize.py - Input normalization and metadata extraction for SPACE pipeline.

Supports multiple input formats:
- GenBank (.gb, .gbk) - Full metadata extraction
- FASTA (.fasta, .fa) - Minimal metadata
- Addgene JSON (.json) - Rich Addgene metadata

Extracts rich metadata and converts to FASTA for downstream tools.
"""

import argparse
import hashlib
import json
import re
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
    """Match text against keyword dictionary using word boundaries.

    Uses regex word boundaries to avoid false positives like matching
    'cos' (cell line) in 'glucose' (sugar).
    """
    if not text:
        return []
    text_lower = text.lower()
    matches = []
    for category, keywords in keyword_dict.items():
        for keyword in keywords:
            # Use word boundary regex to match whole words/phrases only
            # \b matches word boundaries (start/end of word)
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
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


def parse_addgene_json(filepath: Path) -> Dict[str, Any]:
    """Parse an Addgene JSON file (split from addgene_plasmids.json).

    Maps Addgene fields to our internal schema while preserving all metadata.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Get sequence (added by split_addgene.py)
    sequence = data.get("_sequence", "").upper()
    if not sequence:
        raise ValueError(f"No sequence found in {filepath}")

    # Generate IDs
    record_id = str(uuid.uuid4())
    seq_hash = calculate_seq_hash(sequence)

    # Map Addgene fields to internal schema
    addgene_id = data.get("id", "")
    original_id = f"addgene_{addgene_id}" if addgene_id else filepath.stem

    # Extract resistance markers from bacterial_resistance field
    bacterial_resistance = data.get("bacterial_resistance", "") or ""
    resistance_markers = []
    if bacterial_resistance:
        resistance_markers = match_keywords(bacterial_resistance, RESISTANCE_MARKERS)

    # Extract copy number from plasmid_copy field
    plasmid_copy = (data.get("plasmid_copy") or "").lower()
    copy_number = None
    if "high" in plasmid_copy:
        copy_number = "high"
    elif "low" in plasmid_copy:
        copy_number = "low"
    elif plasmid_copy:
        copy_number = "medium"

    # Extract plasmid type from vector_types
    cloning = data.get("cloning", {}) or {}
    vector_types = cloning.get("vector_types", []) or []
    vector_types_text = " ".join(vector_types) if vector_types else ""
    plasmid_type_matches = match_keywords(vector_types_text, PLASMID_TYPES)
    plasmid_type = plasmid_type_matches[0] if plasmid_type_matches else None

    # Also check name and description for plasmid type if not found
    if not plasmid_type:
        name_desc = f"{data.get('name', '')} {data.get('description', '')}"
        plasmid_type_matches = match_keywords(name_desc, PLASMID_TYPES)
        plasmid_type = plasmid_type_matches[0] if plasmid_type_matches else None

    # Extract reporter genes from tags and gene fields
    inserts = data.get("inserts", []) or []
    all_tags = []
    reporter_genes = []
    genes = []
    for insert in inserts:
        insert_tags = insert.get("tags", []) or []
        all_tags.extend(insert_tags)
        insert_gene = insert.get("gene", "") or ""
        if insert_gene:
            genes.append(insert_gene)

    # Match reporters and tags from collected data
    tags_text = " ".join(all_tags + genes)
    reporter_genes = match_keywords(tags_text, REPORTER_GENES)
    protein_tags = match_keywords(tags_text, PROTEIN_TAGS)

    # Extract host from growth_strain
    growth_strain = data.get("growth_strain", "") or ""
    host = growth_strain if growth_strain else None

    # Build comprehensive annotations dict with all Addgene metadata
    annotations = {
        "addgene_id": addgene_id,
        "sequence_source": data.get("_sequence_source", ""),
        "depositor_name": data.get("depositor_name", ""),
        "depositor_institution": data.get("depositor_institution", ""),
        "pi_name": data.get("pi_name", ""),
        "article_references": data.get("article_references", []),
        "url": data.get("url", ""),
        "vector_types": vector_types,
        "growth_strain": growth_strain,
        "growth_temperature": data.get("growth_temperature", ""),
        "bacterial_resistance_raw": bacterial_resistance,
        "plasmid_copy_raw": data.get("plasmid_copy", ""),
        "gene_insert": data.get("gene_insert", ""),
        "inserts": inserts,
        "cloning": cloning,
        "purpose": data.get("purpose", ""),
        "addgene_alias_ids": data.get("alias_ids", []),
    }

    # Clean up None values
    annotations = {k: v for k, v in annotations.items() if v is not None and v != ""}

    return {
        "id": record_id,
        "seq_hash": seq_hash,
        "original_id": original_id,
        "original_name": data.get("name", ""),
        "filename": filepath.name,
        "length": len(sequence),
        "gc_content": calculate_gc_content(sequence),
        "topology": "circular",  # Addgene plasmids are typically circular
        "organism": None,
        "description": data.get("description", "") or data.get("name", ""),
        "host": host,
        "plasmid_type": plasmid_type,
        "copy_number": copy_number,
        "resistance_markers": resistance_markers,
        "reporter_genes": reporter_genes,
        "tags": protein_tags,
        "cds_count": 0,  # Not available from Addgene JSON
        "gene_count": len(genes),
        "has_origin": False,  # Will be detected by PlasmidKit
        "genbank_features": [],  # Not available from Addgene JSON
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
    if suffix == ".json":
        metadata = parse_addgene_json(input_path)
    elif suffix in [".gb", ".gbk", ".genbank"]:
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
