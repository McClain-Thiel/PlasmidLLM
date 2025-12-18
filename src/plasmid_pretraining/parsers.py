"""Parsing utilities for GenBank files and plasmid data."""

import tarfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature

from .constants import (
    RESISTANCE_MARKERS, REPORTER_GENES, PROTEIN_TAGS,
    PLASMID_TYPES, COPY_NUMBER_MARKERS
)


@dataclass
class FeatureInfo:
    """Information about a sequence feature."""
    feature_type: str
    start: int
    end: int
    strand: int
    gene: Optional[str] = None
    product: Optional[str] = None
    locus_tag: Optional[str] = None
    note: Optional[str] = None
    translation: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlasmidRecord:
    """Parsed plasmid record."""
    plasmid_id: str
    source: str
    filename: str
    sequence: str
    sequence_length: int
    gc_content: float
    topology: str
    description: str
    organism: str

    host: Optional[str] = None
    mob_type: Optional[str] = None
    predicted_mobility: Optional[str] = None
    primary_cluster_id: Optional[str] = None
    secondary_cluster_id: Optional[str] = None

    plasmid_type: Optional[str] = None
    resistance_markers: List[str] = field(default_factory=list)
    copy_number: Optional[str] = None
    reporter_genes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    features: List[Dict] = field(default_factory=list)
    cds_count: int = 0
    gene_count: int = 0

    is_complete: bool = True
    has_origin: bool = False
    parse_warnings: List[str] = field(default_factory=list)


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence) if sequence else 0.0


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


def extract_plasmid_id(filename: str, record: SeqRecord) -> str:
    """Extract plasmid ID from filename or record."""
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    if len(parts) > 1:
        return parts[1]
    if record.id and record.id != ".":
        return record.id
    if record.name and record.name != ".":
        return record.name
    return base_name


def parse_feature(feature: SeqFeature) -> FeatureInfo:
    """Parse a BioPython SeqFeature into FeatureInfo."""
    qualifiers = dict(feature.qualifiers)
    return FeatureInfo(
        feature_type=feature.type,
        start=int(feature.location.start),
        end=int(feature.location.end),
        strand=feature.location.strand or 1,
        gene=qualifiers.get("gene", [None])[0],
        product=qualifiers.get("product", [None])[0],
        locus_tag=qualifiers.get("locus_tag", [None])[0],
        note=qualifiers.get("note", [None])[0] if "note" in qualifiers else None,
        translation=qualifiers.get("translation", [None])[0],
        qualifiers={k: v[0] if len(v) == 1 else v for k, v in qualifiers.items()},
    )


def infer_plasmid_type(record: SeqRecord, features: List[FeatureInfo]) -> Optional[str]:
    """Infer plasmid type from features and annotations."""
    search_text = [
        record.description or "",
        str(record.annotations.get("keywords", [])),
        str(record.annotations.get("comment", "")),
    ]
    for feature in features:
        search_text.extend([feature.product or "", feature.gene or "", feature.note or ""])
    combined_text = " ".join(search_text)
    matches = match_keywords(combined_text, PLASMID_TYPES)
    return matches[0] if matches else None


def infer_resistance_markers(features: List[FeatureInfo]) -> List[str]:
    """Infer resistance markers from CDS features."""
    markers = set()
    for feature in features:
        if feature.feature_type not in ["CDS", "gene"]:
            continue
        search_text = " ".join(filter(None, [feature.product, feature.gene, feature.note, feature.locus_tag]))
        for marker in match_keywords(search_text, RESISTANCE_MARKERS):
            markers.add(marker)
    return list(markers)


def infer_reporter_genes(features: List[FeatureInfo]) -> List[str]:
    """Infer reporter genes from CDS features."""
    reporters = set()
    for feature in features:
        if feature.feature_type not in ["CDS", "gene"]:
            continue
        search_text = " ".join(filter(None, [feature.product, feature.gene, feature.note]))
        for reporter in match_keywords(search_text, REPORTER_GENES):
            reporters.add(reporter)
    return list(reporters)


def infer_tags(features: List[FeatureInfo]) -> List[str]:
    """Infer protein tags from CDS features."""
    tags = set()
    for feature in features:
        if feature.feature_type != "CDS":
            continue
        search_text = " ".join(filter(None, [feature.product, feature.gene, feature.note, feature.translation]))
        for tag in match_keywords(search_text, PROTEIN_TAGS):
            tags.add(tag)
    return list(tags)


def infer_copy_number(record: SeqRecord, features: List[FeatureInfo]) -> Optional[str]:
    """Infer copy number from origin of replication."""
    search_text = [record.description or ""]
    for feature in features:
        if feature.feature_type in ["rep_origin", "misc_feature"]:
            search_text.extend([feature.product or "", feature.note or ""])
    combined_text = " ".join(search_text)
    matches = match_keywords(combined_text, COPY_NUMBER_MARKERS)
    return matches[0] if matches else None


def parse_genbank_file(
    filepath: Path,
    source: str,
    metadata_lookup: Dict[str, Dict]
) -> Optional[PlasmidRecord]:
    """Parse a single GenBank file and extract all relevant information."""
    warnings = []

    try:
        with open(filepath, "r") as f:
            record = SeqIO.read(f, "genbank")
    except Exception:
        return None

    sequence = str(record.seq).upper()
    if not sequence or sequence == "N" * len(sequence):
        warnings.append("Empty or all-N sequence")

    valid_bases = set("ATCGN")
    invalid_chars = set(sequence) - valid_bases
    if invalid_chars:
        warnings.append(f"Invalid characters in sequence: {invalid_chars}")

    plasmid_id = extract_plasmid_id(filepath.name, record)
    topology = record.annotations.get("topology", "linear")

    features = []
    cds_count = 0
    gene_count = 0
    has_origin = False

    for feature in record.features:
        if feature.type == "source":
            continue
        feature_info = parse_feature(feature)
        features.append(feature_info)
        if feature.type == "CDS":
            cds_count += 1
        elif feature.type == "gene":
            gene_count += 1
        elif feature.type == "rep_origin":
            has_origin = True

    features_dict = [asdict(f) for f in features]
    plasmid_type = infer_plasmid_type(record, features)
    resistance_markers = infer_resistance_markers(features)
    reporter_genes = infer_reporter_genes(features)
    tags = infer_tags(features)
    copy_number = infer_copy_number(record, features)

    ext_meta = metadata_lookup.get(plasmid_id, {})
    is_complete = ext_meta.get("Completeness", "") == "complete"

    return PlasmidRecord(
        plasmid_id=plasmid_id,
        source=source,
        filename=filepath.name,
        sequence=sequence,
        sequence_length=len(sequence),
        gc_content=calculate_gc_content(sequence),
        topology=topology,
        description=record.description or "",
        organism=record.annotations.get("organism", ""),
        host=ext_meta.get("Host"),
        mob_type=ext_meta.get("MOB_type(s)"),
        predicted_mobility=ext_meta.get("Predicted_Mobility"),
        primary_cluster_id=ext_meta.get("Primary_Cluster_ID"),
        secondary_cluster_id=ext_meta.get("Secondary_Cluster_ID"),
        plasmid_type=plasmid_type,
        resistance_markers=resistance_markers,
        copy_number=copy_number,
        reporter_genes=reporter_genes,
        tags=tags,
        features=features_dict,
        cds_count=cds_count,
        gene_count=gene_count,
        is_complete=is_complete,
        has_origin=has_origin,
        parse_warnings=warnings,
    )


def extract_tar_file(tar_path: Path, output_dir: Path, source_name: str) -> Dict[str, int]:
    """Extract all .gbk files from a tar archive."""
    stats = {"source": source_name, "total_files": 0, "extracted": 0, "errors": 0, "skipped": 0}
    source_dir = output_dir / source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".gbk")]
            stats["total_files"] = len(members)

            for member in members:
                try:
                    filename = Path(member.name).name
                    output_path = source_dir / filename
                    if output_path.exists():
                        stats["skipped"] += 1
                        continue
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        content = file_obj.read()
                        output_path.write_bytes(content)
                        stats["extracted"] += 1
                    else:
                        stats["errors"] += 1
                except Exception:
                    stats["errors"] += 1
    except Exception:
        pass

    return stats
