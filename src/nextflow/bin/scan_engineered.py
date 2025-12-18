#!/usr/bin/env python3
"""
scan_engineered.py - Track 1: Engineered plasmid detection for SPACE pipeline.

Uses PlasmidKit to detect synthetic origins of replication.
Simple classification logic: has_synthetic_ori -> Engineered
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import plasmidkit as pk


def scan_for_origins(file_path: Path) -> Dict:
    """
    Scan a file for synthetic origins using PlasmidKit.

    Returns dict with:
        - has_synthetic_ori: bool
        - origins: list of detected origin names
        - markers: list of detected resistance markers (bonus info)
    """
    try:
        # Load record
        record = pk.load_record(str(file_path))

        # Use only ori detector as specified in plan
        annotations = pk.annotate(record, detectors=['ori'])

        # Extract origins
        origins = []
        for ann in annotations:
            if ann.type in ('rep_origin', 'origin', 'ori'):
                origins.append({
                    "id": ann.id,
                    "type": ann.type,
                    "start": getattr(ann, 'start', None),
                    "end": getattr(ann, 'end', None),
                })

        # Also run marker detector for bonus info (useful for copy_number inference)
        try:
            marker_annotations = pk.annotate(record, detectors=['marker'])
            markers = [
                {"id": ann.id, "type": ann.type}
                for ann in marker_annotations
                if ann.type in ('marker', 'resistance', 'cds_resistance')
            ]
        except Exception:
            markers = []

        has_synthetic_ori = len(origins) > 0

        return {
            "has_synthetic_ori": has_synthetic_ori,
            "origins": origins,
            "origin_names": [o["id"] for o in origins],
            "markers": markers,
            "marker_names": [m["id"] for m in markers],
            "total_annotations": len(annotations),
            "error": None,
        }

    except Exception as e:
        return {
            "has_synthetic_ori": False,
            "origins": [],
            "origin_names": [],
            "markers": [],
            "marker_names": [],
            "total_annotations": 0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Scan plasmids for engineered signatures (synthetic origins)")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--sample-id", required=True, help="Sample ID")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Scan for origins
    result = scan_for_origins(input_path)
    result["sample_id"] = args.sample_id
    result["input_file"] = input_path.name

    # Write result
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    status = "ENGINEERED" if result["has_synthetic_ori"] else "NATURAL"
    origins_str = ", ".join(result["origin_names"]) if result["origin_names"] else "none"
    print(f"[{args.sample_id}] Classification: {status}")
    print(f"  Origins detected: {origins_str}")
    if result["error"]:
        print(f"  Warning: {result['error']}")


if __name__ == "__main__":
    main()
