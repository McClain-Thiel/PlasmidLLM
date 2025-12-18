"""Training pair generation utilities."""

import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    TOKEN_PREFIX, STANDARD_VALUES, LENGTH_BINS, INSERT_CATEGORIES
)


@dataclass
class TrainingPair:
    """A single training pair for the model."""
    prompt: str
    response: str
    plasmid_id: str
    source: str
    metadata: Dict[str, Any]
    insert_mode: str
    sequence_length: int
    insert_length: int
    backbone_length: int


def get_length_bin(length: int) -> str:
    """Get length bin for a sequence length."""
    for min_len, max_len, label in LENGTH_BINS:
        if min_len <= length < max_len:
            return label
    return "unknown"


def normalize_host(host: Optional[str]) -> str:
    """Normalize host organism to standard category."""
    if not host:
        return "unknown"
    host_lower = host.lower()
    if any(x in host_lower for x in ["e. coli", "escherichia", "ecoli"]):
        return "e_coli"
    elif any(x in host_lower for x in ["human", "hek", "hela", "cho", "mammal", "mouse", "rat"]):
        return "mammalian"
    elif any(x in host_lower for x in ["yeast", "saccharomyces", "pichia", "candida"]):
        return "yeast"
    elif any(x in host_lower for x in ["insect", "sf9", "sf21", "drosophila"]):
        return "insect"
    elif any(x in host_lower for x in ["plant", "arabidopsis", "tobacco", "nicotiana"]):
        return "plant"
    elif any(x in host_lower for x in ["bacteria", "bacillus", "pseudomonas", "salmonella"]):
        return "bacterial"
    return "unknown"


def normalize_type(plasmid_type: Optional[str]) -> str:
    """Normalize plasmid type to standard value."""
    if not plasmid_type:
        return "unknown"
    if plasmid_type in STANDARD_VALUES["type"]:
        return plasmid_type
    return "unknown"


def normalize_resistance(markers: List[str]) -> str:
    """Normalize resistance markers to token value."""
    if not markers:
        return "unknown"
    if len(markers) > 1:
        return "multiple"
    marker = markers[0].lower()
    for std in STANDARD_VALUES["resistance"]:
        if std in marker or marker in std:
            return std
    return markers[0].lower()


def normalize_tags(tags: List[str]) -> str:
    """Normalize protein tags to token value."""
    if not tags:
        return "none"
    if len(tags) > 1:
        return "multiple"
    tag = tags[0].lower()
    for std in STANDARD_VALUES["tag"]:
        if std in tag or tag in std:
            return std
    return tags[0].lower()


def get_insert_token(record: Dict, use_sequence: bool = False) -> Tuple[str, str]:
    """Get insert token for the prompt."""
    insert_regions = record.get("insert_regions", [])
    sequence = record.get("sequence", "")

    if not insert_regions:
        return "none", ""

    main_insert = insert_regions[0]
    if len(insert_regions) > 1:
        main_insert = max(insert_regions, key=lambda x: x.get("length", 0))

    insert_start = main_insert.get("start", 0)
    insert_end = main_insert.get("end", 0)
    insert_sequence = sequence[insert_start:insert_end] if sequence else ""

    if use_sequence:
        if len(insert_sequence) > 2000:
            return f"SEQ:{insert_sequence[:1000]}...{insert_sequence[-1000:]}", insert_sequence
        return f"SEQ:{insert_sequence}", insert_sequence

    insert_type = main_insert.get("insert_type", "")
    gene_name = main_insert.get("gene_name", "")
    product = main_insert.get("product", "")

    if gene_name:
        return gene_name.upper(), insert_sequence
    elif product:
        product_words = product.split()
        if product_words:
            return product_words[0].upper(), insert_sequence
    elif insert_type and insert_type != "potential_insert":
        return insert_type.upper(), insert_sequence

    for category, genes in INSERT_CATEGORIES.items():
        for gene in genes:
            if gene.lower() in (product or "").lower():
                return gene, insert_sequence

    return "GOI", insert_sequence


def create_backbone_sequence(record: Dict) -> str:
    """Create backbone sequence with <INSERT> token replacing inserts."""
    sequence = record.get("sequence", "")
    insert_regions = record.get("insert_regions", [])

    if not insert_regions:
        return sequence

    sorted_regions = sorted(insert_regions, key=lambda x: x.get("start", 0), reverse=True)
    result = sequence

    for i, region in enumerate(sorted_regions):
        start = region.get("start", 0)
        end = region.get("end", 0)
        if len(sorted_regions) > 1:
            token = f"<INSERT_{len(sorted_regions) - i}>"
        else:
            token = "<INSERT>"
        result = result[:start] + token + result[end:]

    return result


def build_prompt(record: Dict, insert_token: str, include_optional: bool = True) -> str:
    """Build the prompt string from record metadata."""
    tokens = []

    plasmid_type = normalize_type(record.get("plasmid_type"))
    tokens.append(f"<{TOKEN_PREFIX['type']}:{plasmid_type}>")

    resistance = normalize_resistance(record.get("resistance_markers", []))
    tokens.append(f"<{TOKEN_PREFIX['resistance']}:{resistance}>")

    copy_number = record.get("copy_number") or "unknown"
    tokens.append(f"<{TOKEN_PREFIX['copy_number']}:{copy_number}>")

    tokens.append(f"<{TOKEN_PREFIX['insert']}:{insert_token}>")

    tags = normalize_tags(record.get("tags", []))
    tokens.append(f"<{TOKEN_PREFIX['tag']}:{tags}>")

    if include_optional:
        host = normalize_host(record.get("host"))
        if host != "unknown":
            tokens.append(f"<{TOKEN_PREFIX['host']}:{host}>")
        topology = record.get("topology", "linear")
        tokens.append(f"<{TOKEN_PREFIX['topology']}:{topology}>")
        seq_len = record.get("sequence_length", 0)
        length_bin = get_length_bin(seq_len)
        tokens.append(f"<{TOKEN_PREFIX['length']}:{length_bin}>")

    return " ".join(tokens)


def create_training_pair(
    record: Dict,
    use_sequence_for_insert: bool = False,
    include_optional_tokens: bool = True
) -> Optional[TrainingPair]:
    """Create a single training pair from an annotated record."""
    insert_token, insert_sequence = get_insert_token(record, use_sequence_for_insert)
    backbone = create_backbone_sequence(record)

    backbone_length = len(backbone.replace("<INSERT>", "").replace("<INSERT_1>", "").replace("<INSERT_2>", ""))
    if backbone_length < 500:
        return None

    prompt = build_prompt(record, insert_token, include_optional_tokens)
    insert_length = len(insert_sequence) if insert_sequence else 0

    return TrainingPair(
        prompt=prompt,
        response=backbone,
        plasmid_id=record.get("plasmid_id", ""),
        source=record.get("source", ""),
        metadata={
            "plasmid_type": record.get("plasmid_type"),
            "resistance_markers": record.get("resistance_markers", []),
            "copy_number": record.get("copy_number"),
            "tags": record.get("tags", []),
            "reporter_genes": record.get("reporter_genes", []),
            "host": record.get("host"),
            "topology": record.get("topology"),
            "insert_regions": record.get("insert_regions", []),
            "annotation_confidence": record.get("annotation_confidence", 0),
        },
        insert_mode="sequence" if use_sequence_for_insert else "name",
        sequence_length=record.get("sequence_length", 0),
        insert_length=insert_length,
        backbone_length=backbone_length,
    )


def process_records_to_pairs(
    records: List[Dict],
    sequence_insert_ratio: float = 0.1
) -> Tuple[List[Dict], Dict[str, int]]:
    """Process a list of records into training pairs."""
    stats = {
        "total_records": 0,
        "valid_pairs": 0,
        "with_insert": 0,
        "name_mode": 0,
        "sequence_mode": 0,
        "skipped": 0,
        "errors": 0,
    }

    training_pairs = []

    for record in records:
        stats["total_records"] += 1
        try:
            use_sequence = random.random() < sequence_insert_ratio
            pair = create_training_pair(record, use_sequence_for_insert=use_sequence)
            if pair:
                training_pairs.append(asdict(pair))
                stats["valid_pairs"] += 1
                if record.get("has_identifiable_insert"):
                    stats["with_insert"] += 1
                if use_sequence:
                    stats["sequence_mode"] += 1
                else:
                    stats["name_mode"] += 1
            else:
                stats["skipped"] += 1
        except Exception:
            stats["errors"] += 1

    return training_pairs, stats
