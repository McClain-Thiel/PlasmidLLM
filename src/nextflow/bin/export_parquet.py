#!/usr/bin/env python3
"""
export_parquet.py - Export classified results to Parquet Golden Table for SPACE pipeline.

Creates unified Parquet output with schema:
- id, seq_hash, length (identity)
- classification, topology, copy_number, plasmid_type, host (primary columns)
- origins (list)
- features (JSON - all track results)
- metadata (JSON - remaining GenBank info)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_classified_jsons(input_dir: Path) -> List[Dict]:
    """Load all classified JSON files from directory."""
    records = []

    for json_file in input_dir.glob("*_classified.json"):
        try:
            with open(json_file, 'r') as f:
                record = json.load(f)
                records.append(record)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return records


def create_golden_table(records: List[Dict]) -> pd.DataFrame:
    """Create Golden Table DataFrame from classified records."""
    rows = []

    for record in records:
        row = {
            # Identity
            "id": record.get("id"),
            "seq_hash": record.get("seq_hash"),
            "length": record.get("length"),
            "sequence": record.get("sequence"),  # Include actual sequence

            # Primary columns
            "classification": record.get("classification"),
            "topology": record.get("topology"),
            "copy_number": record.get("copy_number"),
            "plasmid_type": record.get("plasmid_type"),
            "host": record.get("host"),

            # Origins as JSON array
            "origins": json.dumps(record.get("origins", [])),

            # Features as JSON object
            "features": json.dumps(record.get("features", {})),

            # Metadata as JSON object
            "metadata": json.dumps(record.get("metadata", {})),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure correct column order
    columns = [
        "id", "seq_hash", "length", "sequence",
        "classification", "topology", "copy_number", "plasmid_type", "host",
        "origins", "features", "metadata"
    ]

    # Only include columns that exist
    columns = [c for c in columns if c in df.columns]
    df = df[columns]

    return df


def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the Golden Table."""
    summary = {
        "total_records": len(df),
        "classification_counts": df["classification"].value_counts().to_dict() if "classification" in df.columns else {},
        "topology_counts": df["topology"].value_counts().to_dict() if "topology" in df.columns else {},
        "copy_number_counts": df["copy_number"].value_counts().to_dict() if "copy_number" in df.columns else {},
        "plasmid_type_counts": df["plasmid_type"].value_counts().to_dict() if "plasmid_type" in df.columns else {},
        "length_stats": {
            "min": int(df["length"].min()) if "length" in df.columns and len(df) > 0 else 0,
            "max": int(df["length"].max()) if "length" in df.columns and len(df) > 0 else 0,
            "mean": float(df["length"].mean()) if "length" in df.columns and len(df) > 0 else 0,
            "median": float(df["length"].median()) if "length" in df.columns and len(df) > 0 else 0,
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export classified results to Parquet")
    parser.add_argument("--input-dir", required=True, help="Directory containing classified JSON files")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--summary", required=True, help="Output summary JSON file")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    # Load all classified records
    records = load_classified_jsons(input_dir)
    print(f"Loaded {len(records)} classified records")

    if not records:
        # Create empty DataFrame with schema
        df = pd.DataFrame(columns=[
            "id", "seq_hash", "length", "sequence",
            "classification", "topology", "copy_number", "plasmid_type", "host",
            "origins", "features", "metadata"
        ])
    else:
        # Create Golden Table
        df = create_golden_table(records)

    # Write Parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Wrote {len(df)} records to {output_path}")

    # Generate and write summary
    summary = generate_summary(df)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    # Print summary
    print("\n=== Golden Table Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Classification: {summary['classification_counts']}")
    print(f"Topology: {summary['topology_counts']}")
    print(f"Copy number: {summary['copy_number_counts']}")
    if summary['length_stats']['mean'] > 0:
        print(f"Length: {summary['length_stats']['min']}-{summary['length_stats']['max']} bp "
              f"(mean: {summary['length_stats']['mean']:.0f})")


if __name__ == "__main__":
    main()
