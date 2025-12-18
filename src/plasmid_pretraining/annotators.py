"""Annotation utilities using PlasmidKit."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import logging

import plasmidkit as pk

from .constants import BACKBONE_KEYWORDS, INSERT_KEYWORDS

logger = logging.getLogger(__name__)

@dataclass
class InsertRegion:
    """Identified insert region in the sequence."""
    start: int
    end: int
    length: int
    gene_name: Optional[str] = None
    product: Optional[str] = None
    insert_type: Optional[str] = None
    confidence: float = 0.5
    features_included: List[str] = field(default_factory=list)
    strand: int = 1


@dataclass
class AnnotatedPlasmid:
    """Plasmid with insert annotations."""
    plasmid_id: str
    source: str
    filename: str
    sequence: str
    sequence_length: int
    gc_content: float
    topology: str
    description: str
    organism: str
    
    # PlasmidKit Metadata
    pk_version: Optional[str] = None
    pk_features: List[Dict] = field(default_factory=list)

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

    insert_regions: List[Dict] = field(default_factory=list)
    backbone_regions: List[Dict] = field(default_factory=list)
    annotated_sequence: str = ""
    has_identifiable_insert: bool = False
    insert_detection_method: str = ""
    annotation_confidence: float = 0.0


def identify_insert_regions_pk(sequence: str, pk_features: List[Dict]) -> Tuple[List[InsertRegion], float, str]:
    """Identify insert regions using PlasmidKit features."""
    # Simplified Logic: 
    # Everything NOT a backbone feature (origin, marker, promoter) is potential insert space.
    # In reality, PlasmidKit might not tag the insert gene itself, but finding the backbone gaps is safer.
    
    seq_len = len(sequence)
    if seq_len == 0:
        return [], 0.0, "pk_empty"
        
    # Mark backbone mask
    # 0 = unknown, 1 = backbone
    # For a circular plasmid, we need to handle wrapping, but for simplistic gap finding:
    
    # Let's assume standard behavior:
    # 1. Identify all Backbone features from PK
    backbone_types = {"rep_origin", "marker", "promoter", "terminator"}
    backbone_feats = [f for f in pk_features if f.get("type") in backbone_types]
    
    # If no backbone found, we can't be sure about insert
    if not backbone_feats:
        return [], 0.0, "pk_no_backbone"

    # Sort backbone features
    backbone_feats.sort(key=lambda x: x.get("start", 0))

    # Identify Gaps > Min Size (e.g. 300bp)
    gaps = []
    
    # Simple linear gap finding (ignoring circular wrap for moment for robustness)
    current_pos = 0
    min_gap_size = 300
    
    # Merging overlapping backbone features first would be better
    # But let's look for known non-backbone features if possible? No, user said use PK for annotation.
    # PK outputs known features. Is the insert "unknown"?
    # "The insert is the large chunk of DNA that ISN'T labeled as origin/resistance/promoter"
    
    # Let's create a boolean mask for the sequence
    is_backbone = [False] * seq_len
    
    for f in backbone_feats:
        start = max(0, f.get("start", 0))
        end = min(seq_len, f.get("end", 0))
        for i in range(start, end):
            is_backbone[i] = True
            
    # Find contiguous False regions
    insert_regions = []
    current_start = -1
    
    for i in range(seq_len):
        if not is_backbone[i]:
            if current_start == -1:
                current_start = i
        else:
            if current_start != -1:
                # Gap ended
                length = i - current_start
                if length >= min_gap_size:
                    insert_regions.append(InsertRegion(
                        start=current_start, end=i, length=length,
                        insert_type="gap_in_backbone", confidence=0.7,
                        features_included=["unknown_cargo"]
                    ))
                current_start = -1
                
    # Check trailing gap
    if current_start != -1:
        length = seq_len - current_start
        if length >= min_gap_size:
             insert_regions.append(InsertRegion(
                start=current_start, end=seq_len, length=length,
                insert_type="gap_in_backbone", confidence=0.7,
                features_included=["unknown_cargo"]
            ))
            
    return insert_regions, 0.8, "plasmid_kit_gap"


def calculate_backbone_regions(seq_len: int, insert_regions: List[InsertRegion]) -> List[Dict]:
    """
    Compute backbone regions as the complement of insert regions.
    Assumes linear coordinates; for circular sequences this is an approximation.
    """
    if seq_len <= 0:
        return []
    # Sort inserts by start
    inserts = sorted(insert_regions, key=lambda r: r.start)
    backbone = []
    cursor = 0
    for r in inserts:
        if r.start > cursor:
            backbone.append({"start": cursor, "end": r.start, "length": r.start - cursor, "type": "backbone"})
        cursor = max(cursor, r.end)
    if cursor < seq_len:
        backbone.append({"start": cursor, "end": seq_len, "length": seq_len - cursor, "type": "backbone"})
    return backbone


def create_annotated_sequence(sequence: str, insert_regions: List[InsertRegion]) -> str:
    """
    Create a sequence string with simple markers around inferred insert regions.
    """
    if not sequence:
        return ""
    if not insert_regions:
        return sequence
    # Replace inserts with <INSERT_n> tokens from the end to keep indices valid
    annotated = sequence
    for idx, region in enumerate(sorted(insert_regions, key=lambda r: r.start, reverse=True), start=1):
        token = f"<INSERT_{idx}>"
        annotated = annotated[: region.start] + token + annotated[region.end :]
    return annotated


def annotate_record(record: Dict) -> Dict:
    """Annotate a single plasmid record using PlasmidKit."""


    sequence = record.get("sequence", "")
    

    
    try:
        # Load the sequence string as a record (not a file path)
        pk_record = pk.load_record(sequence, is_sequence=True)
        # Convert Feature objects to dicts immediately
        # Skip pyrodigal since GenBank files already have CDS annotations
        pk_features_objs = pk.annotate(
            pk_record, 
            is_sequence=True,
            skip_prodigal=True
        )
        pk_features = [f.to_dict() for f in pk_features_objs]
        pk_version = "v0.1.0" # Hardcoded for now, or get via pkg_resources
    except Exception as e:
        logger.error(f"PlasmidKit failed: {e}")
        pk_features = []
        pk_version = "error"

    # 2. Identify Inserts based on PK features
    insert_regions, confidence, method = identify_insert_regions_pk(sequence, pk_features)
    
    # 3. Backbone Calculation
    backbone_regions = calculate_backbone_regions(len(sequence), insert_regions)
    annotated_seq = create_annotated_sequence(sequence, insert_regions)
    
    # 4. Merge
    # We keep GenBank metadata (host, organism) but use PK for resistance/features
    
    # Extract resistance markers from PK features
    resistance = [f["id"] for f in pk_features if f.get("type") == "marker"]
    origins = [f["id"] for f in pk_features if f.get("type") == "rep_origin"]
    promoters = [f["id"] for f in pk_features if f.get("type") == "promoter"]
    
    annotated = AnnotatedPlasmid(
        plasmid_id=record.get("plasmid_id", ""),
        source=record.get("source", ""),
        filename=record.get("filename", ""),
        sequence=sequence,
        sequence_length=len(sequence),
        gc_content=record.get("gc_content", 0.0),
        topology=record.get("topology", "linear"),
        description=record.get("description", ""),
        organism=record.get("organism", ""),
        
        pk_version=pk_version,
        pk_features=pk_features,
        
        host=record.get("host"),
        plasmid_type=record.get("plasmid_type"), # Keep genbank type for now
        resistance_markers=resistance if resistance else record.get("resistance_markers", []),
        has_origin=bool(origins) or record.get("has_origin", False),
        
        insert_regions=[asdict(r) for r in insert_regions],
        backbone_regions=backbone_regions,
        annotated_sequence=annotated_seq,
        has_identifiable_insert=len(insert_regions) > 0,
        insert_detection_method=method,
        annotation_confidence=confidence,
        
        features=record.get("features", []) # Keep original features for reference?
    )

    return asdict(annotated)
