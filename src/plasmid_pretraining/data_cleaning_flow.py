"""Metaflow pipeline for plasmid data cleaning and pretraining prep."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from metaflow import FlowSpec, current, resources, step
from tqdm.auto import tqdm

from plasmid_pretraining.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaningFlow(FlowSpec):
    """
    Pipeline for cleaning and processing PlasmidScope data.
    
    Architecture:
    1. Extract: Parallel fan-out by input archive (.tar.gz).
    2. Shard: Group extracted files into batches.
    3. Process (per batch): Parse -> Annotate -> CreatePairs.
    4. Split: Aggregate and create final splits.
    
    Caching:
    - Checks existence of output directories/files before running steps.
    """

    @step
    def start(self):
        """Initialize and discover input archives."""
        print(f"Starting DataCleaningFlow with base_dir: {settings.base_dir}")
        self.input_dir = settings.input_dir
        
        # Discover .tar.gz files
        self.archives = [str(p) for p in self.input_dir.glob("*.tar.gz")]

        if not self.archives:
            print("No .tar.gz archives found in input_dir; skipping extraction.")
            self.next(self.shard_files)
            return

        print(f"Found {len(self.archives)} archives to extract.")

        # Fan-out 1: Extraction
        self.next(self.extract_archive, foreach="archives")

    @step
    def extract_archive(self):
        """Extract a single tar archive if not already extracted."""
        from plasmid_pretraining.parsers import extract_tar_file
        
        archive_path = Path(self.input)
        source_name = archive_path.name.replace(".gbk.tar.gz", "").replace(".tar.gz", "")
        
        # Cache Check
        # We define "extracted" as the source folder existing in raw_dir and having content
        output_dir = settings.raw_dir
        target_dir = output_dir / source_name
        
        if target_dir.exists() and any(target_dir.iterdir()):
             print(f"[{source_name}] Data appears to be already extracted in {target_dir}. Skipping.")
        else:
            print(f"[{source_name}] Extracting {archive_path}...")
            stats = extract_tar_file(archive_path, output_dir, source_name)
            print(f"[{source_name}] Extraction complete: {stats}")
            
        self.next(self.shard_files_join)

    @step
    def shard_files_join(self, inputs):
        """Join extraction tasks."""
        self.next(self.shard_files)

    @step
    def shard_files(self):
        """
        Scan all raw files and create batches for processing.
        This balances load across workers regardless of source size.
        """
        all_gbk_files = list(settings.raw_dir.glob("**/*.gbk"))
        print(f"Total files to process: {len(all_gbk_files)}")
        
        # Create batches
        batch_size = settings.batch_size
        # We will create a list of lists of file PLANS.
        # Passing full file contents is bad. Passing 1000 absolute paths strings is fine.
        
        self.batches = []
        current_batch = []
        for p in all_gbk_files:
            current_batch.append(str(p))
            if len(current_batch) >= batch_size:
                self.batches.append(current_batch)
                current_batch = []
        if current_batch:
            self.batches.append(current_batch)
            
        print(f"Created {len(self.batches)} batches (size ~{batch_size})")
        
        if not self.batches:
            print("No gbk files found after sharding; proceeding to split step.")
            self.batches = []
            self.next(self.split)
            return

        # Fan-out 2: Process Batches
        self.next(self.parse_batch, foreach="batches")

    @step
    def parse_batch(self):
        """
        Parse a batch of files. If final output already exists, skip downstream work.
        """
        from plasmid_pretraining.parsers import parse_genbank_file

        batch_files: List[str] = self.input
        import hashlib

        self.batch_files = batch_files
        self.batch_sig = hashlib.md5("".join(sorted(batch_files)).encode()).hexdigest()
        self.parsed_out = settings.parsed_dir / f"batch_{self.batch_sig}.parquet"
        self.annotated_out = settings.annotated_dir / f"batch_{self.batch_sig}.parquet"
        self.processed_out = settings.processed_dir / f"batch_{self.batch_sig}.parquet"

        if not settings.force_reprocess and self.processed_out.exists():
            logger.info(f"Batch {self.batch_sig} already processed; skipping parse/annotate/pairs.")
            try:
                df = pd.read_parquet(self.processed_out)
                self.result_info = {"pairs": len(df)}
            except Exception:
                self.result_info = {"pairs": 0}
            # continue through annotate -> pairs to satisfy static graph; downstream will short-circuit
            self.next(self.annotate_batch)
            return

        if self.parsed_out.exists():
            logger.info(f"Batch {self.batch_sig} parsed cache found; reusing.")
            self.next(self.annotate_batch)
            return

        start_time = time.perf_counter()
        parsed_records = []
        for gbk_path in tqdm(self.batch_files, desc=f"Parse {self.batch_sig}"):
            source_name = Path(gbk_path).parent.name
            record = parse_genbank_file(Path(gbk_path), source_name, {})
            if record:
                parsed_records.append(asdict(record))
        parse_time = time.perf_counter() - start_time

        self._save_parquet(parsed_records, self.parsed_out)
        print(f"Batch {self.batch_sig}: {len(parsed_records)} records parsed in {parse_time:.2f}s.")
        self.next(self.annotate_batch)

    @step
    def annotate_batch(self):
        """
        Annotate parsed records for a batch. Uses cached annotated output if present.
        """
        from plasmid_pretraining.annotators import annotate_record

        if not settings.force_reprocess and self.processed_out.exists():
            logger.info(f"Batch {self.batch_sig} already processed; skipping annotate/pairs.")
            try:
                df = pd.read_parquet(self.processed_out)
                self.result_info = {"pairs": len(df)}
            except Exception:
                self.result_info = {"pairs": 0}
            self.next(self.pairs_batch)
            return

        if self.annotated_out.exists():
            logger.info(f"Batch {self.batch_sig} annotated cache found; reusing.")
            self.next(self.pairs_batch)
            return

        df = pd.read_parquet(self.parsed_out)
        start_time = time.perf_counter()
        annotated_records = []
        for rec in tqdm(df.to_dict(orient="records"), desc=f"Annotate {self.batch_sig}"):
            ann_rec = annotate_record(rec)
            annotated_records.append(ann_rec)
        annotate_time = time.perf_counter() - start_time

        self._save_parquet(annotated_records, self.annotated_out)
        print(f"Batch {self.batch_sig}: {len(annotated_records)} records annotated in {annotate_time:.2f}s.")
        self.next(self.pairs_batch)

    @step
    def pairs_batch(self):
        """
        Create training pairs for a batch. Skips if already processed.
        """
        from plasmid_pretraining.training import process_records_to_pairs
        import json

        if not settings.force_reprocess and self.processed_out.exists():
            logger.info(f"Batch {self.batch_sig} already processed; skipping.")
            try:
                df = pd.read_parquet(self.processed_out)
                self.result_info = {"pairs": len(df)}
            except Exception:
                self.result_info = {"pairs": 0}
            self.next(self.join_batches)
            return

        df = pd.read_parquet(self.annotated_out)
        annotated_records = df.to_dict(orient="records")

        # Deserialize JSON-encoded complex columns
        complex_cols = [
            "features",
            "resistance_markers",
            "reporter_genes",
            "tags",
            "parse_warnings",
            "insert_regions",
            "backbone_regions",
            "metadata",
        ]
        start_time = time.perf_counter()
        for rec in annotated_records:
            for col in complex_cols:
                val = rec.get(col)
                if isinstance(val, str):
                    try:
                        rec[col] = json.loads(val)
                    except Exception:
                        # leave as-is if not JSON
                        pass

        pairs, stats = process_records_to_pairs(annotated_records, settings.sequence_ratio)
        pairs_time = time.perf_counter() - start_time

        # Save Final Batch Output
        self._save_parquet(pairs, self.processed_out)

        print(f"Batch {self.batch_sig}: {len(pairs)} pairs generated in {pairs_time:.2f}s.")
        print(f"  Stats: {stats}")

        self.result_info = {
            "pairs": len(pairs),
            "parsed": len(annotated_records),
            "stats": stats
        }

        self.next(self.join_batches)

    def _save_parquet(self, records: List[Dict], output_path: Path):
        """Helper to save list of dicts to parquet."""
        df = pd.DataFrame(records)
        complex_cols = [
            "features",
            "resistance_markers",
            "reporter_genes",
            "tags",
            "parse_warnings",
            "insert_regions",
            "backbone_regions",
            "metadata",
        ]
        for col in complex_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

        df.to_parquet(output_path, index=False)

    @step
    def join_batches(self, inputs):
        """Aggregate results from all batches."""
        total_pairs = sum(i.result_info.get("pairs", 0) for i in inputs)
        print(f"Total pairs generated across all batches: {total_pairs}")
        self.next(self.split)

    @step
    def split(self):
        """Merge data and split."""
        from plasmid_pretraining.splitters import stratified_split, save_jsonl
        import pandas as pd
        
        all_pairs = []
        # Gather all parquet files from processed_dir
        # Note: If we had cached runs, we might have files from previous runs? 
        # Yes, that's what "cache" implies. We pick up everything in processed_dir.
        
        processed_files = list(settings.processed_dir.glob("*.parquet"))
        print(f"Reading {len(processed_files)} processed batch files...")
        
        for pfile in processed_files:
            try:
                df = pd.read_parquet(pfile)
                batch_pairs = df.to_dict(orient="records")
                # Deserialize metadata
                for p in batch_pairs:
                    if 'metadata' in p and isinstance(p['metadata'], str):
                         try:
                            p['metadata'] = json.loads(p['metadata'])
                         except: pass
                all_pairs.extend(batch_pairs)
            except Exception as e:
                print(f"Failed to read {pfile}: {e}")

        print(f"Total records for splitting: {len(all_pairs)}")
        
        if not all_pairs:
            print("No data found.")
            self.next(self.end)
            return
            
        # Split
        train, val, test = stratified_split(
            all_pairs,
            train_ratio=settings.train_ratio,
            val_ratio=settings.val_ratio,
            test_ratio=settings.test_ratio,
            random_seed=settings.random_seed
        )
        
        # Save
        save_jsonl(train, settings.final_dir / "train.jsonl", simple=False)
        save_jsonl(val, settings.final_dir / "val.jsonl", simple=False)
        save_jsonl(test, settings.final_dir / "test.jsonl", simple=False)
        
        print(f"Saved splits to {settings.final_dir}")
        self.next(self.end)

    @step
    def end(self):
        print("Data Cleaning Flow Complete.")


if __name__ == "__main__":
    DataCleaningFlow()
