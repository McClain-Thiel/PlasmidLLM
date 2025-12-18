import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict

import plasmidkit as pk

def scan_file(file_path: Path) -> Dict:
    """
    Scan a single file for engineered signatures.
    Returns a dict with classification and details.
    """
    try:
        # Load and annotate
        # We use 'ori' and 'marker' to detect engineered features.
        # If plasmidkit detects them, they are likely from its 'engineered-core' DB.
        record = pk.load_record(str(file_path))
        
        # Using ori and marker detectors
        # Note: If markers are found, it's a strong sign of being engineered (antibiotic resistance in a vector context)
        annotations = pk.annotate(record, detectors=['ori', 'marker'])
        
        # Extract features
        oris = [ann.id for ann in annotations if ann.type in ('rep_origin', 'origin', 'ori')]
        markers = [ann.id for ann in annotations if ann.type in ('marker', 'resistance', 'cds_resistance')]
        
        # Classification Logic
        # PRD: Engineered Dominance if Matches >1 Synthetic Origin OR contains Artificial Selection Markers
        
        is_engineered = False
        reason = []
        
        if oris:
            is_engineered = True
            reason.append(f"Origins: {', '.join(oris)}")
            
        if markers:
            is_engineered = True
            reason.append(f"Markers: {', '.join(markers)}")
            
        return {
            "filename": file_path.name,
            "classification": "Engineered" if is_engineered else "Natural", # Default to Natural if no engineered sigs found
            "evidence": "; ".join(reason),
            "feature_count": len(annotations),
            "origins": ";".join(oris),
            "markers": ";".join(markers)
        }
        
    except Exception as e:
        return {
            "filename": file_path.name,
            "classification": "Error",
            "evidence": str(e),
            "feature_count": 0,
            "origins": "",
            "markers": ""
        }

def main():
    parser = argparse.ArgumentParser(description="Scan plasmids for engineered signatures.")
    parser.add_argument("--input", required=True, help="Input FASTA/GBK file or directory")
    parser.add_argument("--output", required=True, help="Output CSV file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process.extend(input_path.glob("*.fasta"))
        files_to_process.extend(input_path.glob("*.fa"))
        files_to_process.extend(input_path.glob("*.gbk"))
        files_to_process.extend(input_path.glob("*.gb"))
    
    print(f"Scanning {len(files_to_process)} files...")
    
    results = []
    for f in files_to_process:
        res = scan_file(f)
        results.append(res)
        
    # Write to CSV
    keys = ["filename", "classification", "evidence", "feature_count", "origins", "markers"]
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    main()
