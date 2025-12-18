#!/usr/bin/env python3
"""
scan_natural.py - Track 2: Parse natural plasmid analysis outputs for SPACE pipeline.

Parses outputs from Bakta, MOB-suite, and COPLA into unified JSON format.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_bakta_gff(gff_path: Path) -> Dict[str, Any]:
    """
    Parse Bakta GFF3 output for AMR genes, virulence factors, and annotation stats.
    """
    result = {
        "amr_genes": [],
        "virulence_factors": [],
        "cds_features": [],
        "cds_count": 0,
        "gene_count": 0,
        "trna_count": 0,
        "rrna_count": 0,
        "total_features": 0,
        "coding_density": 0.0,
        "ori_features": [],
    }

    if not gff_path.exists() or gff_path.stat().st_size == 0:
        return result

    total_length = 0
    coding_length = 0

    try:
        with open(gff_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    # Parse sequence-region for total length
                    if line.startswith("##sequence-region"):
                        parts = line.split()
                        if len(parts) >= 4:
                            total_length = int(parts[3])
                    continue

                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 9:
                    continue

                seqid, source, ftype, start, end, score, strand, phase, attributes = parts
                result["total_features"] += 1

                start_int = int(start)
                end_int = int(end)
                feature_length = end_int - start_int + 1

                # Parse attributes
                attr_dict = {}
                for attr in attributes.split(";"):
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        attr_dict[key] = value

                feature_info = {
                    "type": ftype,
                    "start": start_int,
                    "end": end_int,
                    "strand": strand,
                    "name": attr_dict.get("Name", ""),
                    "product": attr_dict.get("product", ""),
                    "gene": attr_dict.get("gene", ""),
                }

                if ftype == "CDS":
                    result["cds_count"] += 1
                    coding_length += feature_length
                    result["cds_features"].append(feature_info)

                    # Check for AMR genes
                    product_lower = (feature_info["product"] or "").lower()
                    gene_lower = (feature_info["gene"] or "").lower()
                    name_lower = (feature_info["name"] or "").lower()

                    amr_keywords = ["resistance", "beta-lactam", "aminoglycoside", "tetracycline",
                                   "chloramphenicol", "macrolide", "quinolone", "sulfonamide",
                                   "bla", "aph", "aac", "tet", "cat", "erm", "qnr", "sul"]
                    if any(kw in product_lower or kw in gene_lower or kw in name_lower for kw in amr_keywords):
                        result["amr_genes"].append({
                            "gene": feature_info["gene"] or feature_info["name"],
                            "product": feature_info["product"],
                            "start": start_int,
                            "end": end_int,
                        })

                    # Check for virulence factors
                    vf_keywords = ["virulence", "toxin", "adhesin", "invasin", "hemolysin",
                                  "enterotoxin", "exotoxin", "fimbr", "pilus", "secretion"]
                    if any(kw in product_lower or kw in gene_lower for kw in vf_keywords):
                        result["virulence_factors"].append({
                            "gene": feature_info["gene"] or feature_info["name"],
                            "product": feature_info["product"],
                        })

                elif ftype == "gene":
                    result["gene_count"] += 1
                elif ftype == "tRNA":
                    result["trna_count"] += 1
                elif ftype == "rRNA":
                    result["rrna_count"] += 1
                elif ftype in ("rep_origin", "oriV", "oriT"):
                    result["ori_features"].append(feature_info)

        # Calculate coding density
        if total_length > 0:
            result["coding_density"] = coding_length / total_length

    except Exception as e:
        result["parse_error"] = str(e)

    return result


def parse_mobsuite_tsv(tsv_path: Path) -> Dict[str, Any]:
    """
    Parse MOB-suite mob_typer output for mobility and replicon typing.
    """
    result = {
        "mobility": None,
        "replicon_type": None,
        "relaxase_type": None,
        "mpf_type": None,
        "orit_type": None,
        "predicted_mobility": None,
        "mash_nearest_neighbor": None,
        "mash_distance": None,
    }

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return result

    try:
        with open(tsv_path, "r") as f:
            content = f.read()

        # Skip comment lines
        if content.startswith("#"):
            return result

        # MOB-suite outputs tab-separated key-value pairs
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("#"):
                continue
            if "\t" not in line:
                continue

            # Parse TSV format
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            # MOB-typer output columns vary, try to parse common fields
            # Typical columns: file_id, num_contigs, total_length, gc, rep_type, etc.
            # This is a simplified parser - adjust based on actual output format

        # Try parsing as header + data row
        if len(lines) >= 2:
            header = lines[0].split("\t")
            data = lines[1].split("\t") if len(lines) > 1 else []

            header_map = {h.lower().replace(" ", "_"): i for i, h in enumerate(header)}

            def get_field(name, default=None):
                idx = header_map.get(name)
                if idx is not None and idx < len(data):
                    val = data[idx].strip()
                    return val if val and val != "-" else default
                return default

            result["replicon_type"] = get_field("rep_type(s)") or get_field("replicon_type")
            result["relaxase_type"] = get_field("relaxase_type(s)") or get_field("relaxase_type")
            result["mpf_type"] = get_field("mpf_type")
            result["orit_type"] = get_field("orit_type(s)") or get_field("orit_type")
            result["predicted_mobility"] = get_field("predicted_mobility")
            result["mash_nearest_neighbor"] = get_field("mash_nearest_neighbor")
            result["mash_distance"] = get_field("mash_neighbor_distance")

            # Determine mobility class
            mob = result["predicted_mobility"]
            if mob:
                mob_lower = mob.lower()
                if "conjugative" in mob_lower:
                    result["mobility"] = "Conjugative"
                elif "mobilizable" in mob_lower:
                    result["mobility"] = "Mobilizable"
                else:
                    result["mobility"] = "Non-mobilizable"

    except Exception as e:
        result["parse_error"] = str(e)

    return result


def parse_copla_tsv(tsv_path: Path) -> Dict[str, Any]:
    """
    Parse COPLA output for host range prediction.
    """
    result = {
        "predicted_host": None,
        "ptu": None,  # Plasmid Taxonomic Unit
        "host_range": None,
        "confidence": None,
    }

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return result

    try:
        with open(tsv_path, "r") as f:
            content = f.read()

        if content.startswith("#") and "skipped" in content.lower():
            return result

        lines = content.strip().split("\n")
        if len(lines) >= 2:
            header = lines[0].split("\t")
            data = lines[1].split("\t") if len(lines) > 1 else []

            header_map = {h.lower().replace(" ", "_"): i for i, h in enumerate(header)}

            def get_field(name, default=None):
                idx = header_map.get(name)
                if idx is not None and idx < len(data):
                    val = data[idx].strip()
                    return val if val and val != "-" else default
                return default

            # COPLA output fields (adjust based on actual output)
            result["predicted_host"] = get_field("host") or get_field("predicted_host")
            result["ptu"] = get_field("ptu") or get_field("ptuid")
            result["host_range"] = get_field("host_range")
            result["confidence"] = get_field("confidence") or get_field("score")

    except Exception as e:
        result["parse_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse natural plasmid analysis outputs")
    parser.add_argument("--sample-id", required=True, help="Sample ID")
    parser.add_argument("--bakta-gff", required=True, help="Bakta GFF3 output")
    parser.add_argument("--mobsuite-tsv", required=True, help="MOB-suite TSV output")
    parser.add_argument("--copla-tsv", required=True, help="COPLA TSV output")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    # Parse all inputs
    bakta_result = parse_bakta_gff(Path(args.bakta_gff))
    mobsuite_result = parse_mobsuite_tsv(Path(args.mobsuite_tsv))
    copla_result = parse_copla_tsv(Path(args.copla_tsv))

    # Combine results
    result = {
        "sample_id": args.sample_id,
        # From Bakta
        "amr_genes": bakta_result["amr_genes"],
        "amr_gene_names": [g["gene"] for g in bakta_result["amr_genes"]],
        "virulence_factors": bakta_result["virulence_factors"],
        "cds_count": bakta_result["cds_count"],
        "gene_count": bakta_result["gene_count"],
        "coding_density": bakta_result["coding_density"],
        "ori_features": bakta_result["ori_features"],
        # From MOB-suite
        "mobility": mobsuite_result["mobility"],
        "replicon_type": mobsuite_result["replicon_type"],
        "relaxase_type": mobsuite_result["relaxase_type"],
        "predicted_mobility": mobsuite_result["predicted_mobility"],
        # From COPLA
        "predicted_host": copla_result["predicted_host"],
        "ptu": copla_result["ptu"],
        "host_range": copla_result["host_range"],
        # Raw results for debugging
        "_bakta": bakta_result,
        "_mobsuite": mobsuite_result,
        "_copla": copla_result,
    }

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"[{args.sample_id}] Natural scan complete")
    print(f"  AMR genes: {len(result['amr_genes'])}")
    print(f"  Mobility: {result['mobility']}")
    print(f"  Replicon type: {result['replicon_type']}")
    print(f"  Predicted host: {result['predicted_host']}")


if __name__ == "__main__":
    main()
