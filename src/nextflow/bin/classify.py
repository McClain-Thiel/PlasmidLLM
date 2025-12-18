#!/usr/bin/env python3
"""
classify.py - Merge track results and classify plasmids for SPACE pipeline.

Classification logic:
- Simple: has_synthetic_ori -> Engineered, else Natural
- Primary columns use GenBank > Prediction precedence
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# Copy number mapping from origin types
COPY_NUMBER_HIGH = ['cole1', 'colei', 'puc', 'pmb1', 'pbr322']
COPY_NUMBER_MEDIUM = ['p15a', 'pbb1', 'psa']
COPY_NUMBER_LOW = ['psc101', 'f_plasmid', 'f1', 'p1', 'rk2']


def predict_copy_number(origins: List[str]) -> Optional[str]:
    """Predict copy number from detected origin types."""
    if not origins:
        return None

    for ori in origins:
        ori_lower = ori.lower()
        if any(h in ori_lower for h in COPY_NUMBER_HIGH):
            return 'high'
        if any(m in ori_lower for m in COPY_NUMBER_MEDIUM):
            return 'medium'
        if any(l in ori_lower for l in COPY_NUMBER_LOW):
            return 'low'

    return None


def predict_topology(qc_result: Dict, engineered_result: Dict) -> Optional[str]:
    """
    Predict topology from sequence features.

    Heuristics:
    - Presence of synthetic ori suggests circular (most cloning vectors are circular)
    - ITRs (inverted terminal repeats) suggest linear
    - Default to circular for plasmids
    """
    # If we detected synthetic origins, likely circular
    if engineered_result.get('has_synthetic_ori'):
        return 'circular'

    # Check for ITR signatures in QC (would need to add this detection)
    # For now, default to None (let GenBank decide)
    return None


def predict_plasmid_type(metadata: Dict, engineered_result: Dict) -> Optional[str]:
    """
    Predict plasmid type from features and annotations.

    Uses keyword matching with word boundaries on description, features, and detected markers.
    """
    # Check description for keywords
    description = (metadata.get('description') or '').lower()

    # Promoter-based prediction
    origin_names = [o.lower() for o in engineered_result.get('origin_names', [])]
    marker_names = [m.lower() for m in engineered_result.get('marker_names', [])]
    all_text = description + ' ' + ' '.join(origin_names) + ' ' + ' '.join(marker_names)

    # Type keywords (from constants.py)
    type_keywords = {
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

    for ptype, keywords in type_keywords.items():
        for kw in keywords:
            # Use word boundary regex to avoid false positives
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, all_text):
                return ptype

    return None


def classify(
    engineered_result: Dict,
    genbank_meta: Dict,
    natural_result: Dict,
    qc_result: Dict,
) -> Dict[str, Any]:
    """
    Classify plasmid and merge all track results.

    Classification: has_synthetic_ori -> Engineered, else Natural
    Primary columns: GenBank > Prediction precedence
    """
    # Simple classification based on ori detection
    has_synthetic_ori = engineered_result.get('has_synthetic_ori', False)
    classification = 'Engineered' if has_synthetic_ori else 'Natural'

    # PRIMARY COLUMNS: GenBank takes precedence over predictions

    # Topology: GenBank > prediction
    topology = genbank_meta.get('topology')
    if not topology:
        topology = predict_topology(qc_result, engineered_result)
    if not topology:
        topology = 'circular'  # Default for plasmids

    # Copy number: GenBank > prediction from ori
    copy_number = genbank_meta.get('copy_number')
    if not copy_number:
        copy_number = predict_copy_number(engineered_result.get('origin_names', []))
    if not copy_number:
        copy_number = 'unknown'

    # Plasmid type: GenBank > prediction from features
    plasmid_type = genbank_meta.get('plasmid_type')
    if not plasmid_type:
        plasmid_type = predict_plasmid_type(genbank_meta, engineered_result)
    if not plasmid_type:
        plasmid_type = 'unknown'

    # Host: GenBank > COPLA prediction
    host = genbank_meta.get('host')
    if not host:
        host = natural_result.get('predicted_host')
    if not host:
        host = 'unknown'

    # Build result
    result = {
        # Identity
        "id": genbank_meta.get('id'),
        "seq_hash": genbank_meta.get('seq_hash'),
        "length": genbank_meta.get('length'),

        # PRIMARY COLUMNS
        "classification": classification,
        "topology": topology,
        "copy_number": copy_number,
        "plasmid_type": plasmid_type,
        "host": host,
        "origins": engineered_result.get('origin_names', []),

        # FEATURES - all track results
        "features": {
            "engineered": {
                "has_synthetic_ori": engineered_result.get('has_synthetic_ori'),
                "origins": engineered_result.get('origins', []),
                "markers": engineered_result.get('markers', []),
            },
            "natural": {
                "amr_genes": natural_result.get('amr_gene_names', []),
                "mobility": natural_result.get('mobility'),
                "replicon_type": natural_result.get('replicon_type'),
                "coding_density": natural_result.get('coding_density'),
                "predicted_host": natural_result.get('predicted_host'),
                "ptu": natural_result.get('ptu'),
            },
            "qc": {
                "gc_content": qc_result.get('gc_content'),
                "linguistic_complexity": qc_result.get('linguistic_complexity'),
                "synthesis_risk": qc_result.get('synthesis_risk'),
                "synthesis_risk_reasons": qc_result.get('synthesis_risk_reasons', []),
                "max_homopolymer": qc_result.get('homopolymers', {}).get('max_length'),
            },
        },

        # METADATA - remaining GenBank info
        "metadata": {
            "organism": genbank_meta.get('organism'),
            "description": genbank_meta.get('description'),
            "original_id": genbank_meta.get('original_id'),
            "original_name": genbank_meta.get('original_name'),
            "filename": genbank_meta.get('filename'),
            "gc_content": genbank_meta.get('gc_content'),
            "resistance_markers": genbank_meta.get('resistance_markers', []),
            "reporter_genes": genbank_meta.get('reporter_genes', []),
            "tags": genbank_meta.get('tags', []),
            "cds_count": genbank_meta.get('cds_count'),
            "gene_count": genbank_meta.get('gene_count'),
            "has_origin": genbank_meta.get('has_origin'),
            "genbank_features": genbank_meta.get('genbank_features', []),
            "annotations": genbank_meta.get('annotations', {}),
        },
    }

    return result


def load_json(path: Path) -> Dict:
    """Load JSON file, return empty dict if missing or invalid."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Classify plasmids and merge track results")
    parser.add_argument("--sample-id", required=True, help="Sample ID")
    parser.add_argument("--engineered", required=True, help="Engineered scan JSON")
    parser.add_argument("--metadata", required=True, help="Metadata JSON from normalize")
    parser.add_argument("--natural", required=True, help="Natural scan JSON")
    parser.add_argument("--qc", required=True, help="QC JSON")
    parser.add_argument("--output", required=True, help="Output classified JSON")

    args = parser.parse_args()

    # Load all inputs
    engineered_result = load_json(Path(args.engineered))
    genbank_meta = load_json(Path(args.metadata))
    natural_result = load_json(Path(args.natural))
    qc_result = load_json(Path(args.qc))

    # Classify
    result = classify(engineered_result, genbank_meta, natural_result, qc_result)
    result["sample_id"] = args.sample_id

    # Write output
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"[{args.sample_id}] Classification: {result['classification']}")
    print(f"  Topology: {result['topology']}")
    print(f"  Copy number: {result['copy_number']}")
    print(f"  Plasmid type: {result['plasmid_type']}")
    print(f"  Host: {result['host']}")
    print(f"  Origins: {', '.join(result['origins']) if result['origins'] else 'none'}")


if __name__ == "__main__":
    main()
