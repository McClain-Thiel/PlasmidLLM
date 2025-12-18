"""Dataset splitting and saving utilities."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def extract_stratification_key(pair: Dict) -> str:
    """Extract key for stratified splitting."""
    source = pair.get("source", "unknown")
    metadata = pair.get("metadata", {})
    plasmid_type = metadata.get("plasmid_type", "unknown") or "unknown"
    return f"{source}_{plasmid_type}"


def stratified_split(
    pairs: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets with stratification."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    groups = defaultdict(list)
    for pair in pairs:
        key = extract_stratification_key(pair)
        groups[key].append(pair)

    train_set, val_set, test_set = [], [], []

    for key, group_pairs in groups.items():
        n = len(group_pairs)
        if n < 3:
            train_set.extend(group_pairs)
            continue

        random.shuffle(group_pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if n >= 10:
            train_end = max(train_end, 1)
            val_end = max(val_end, train_end + 1)

        train_set.extend(group_pairs[:train_end])
        val_set.extend(group_pairs[train_end:val_end])
        test_set.extend(group_pairs[val_end:])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def pair_to_simple_format(pair: Dict) -> Dict:
    """Convert full pair to simple prompt-response format."""
    return {"prompt": pair["prompt"], "response": pair["response"]}


def pair_to_full_format(pair: Dict) -> Dict:
    """Keep full pair information."""
    return {
        "prompt": pair["prompt"],
        "response": pair["response"],
        "plasmid_id": pair.get("plasmid_id", ""),
        "source": pair.get("source", ""),
        "insert_mode": pair.get("insert_mode", ""),
        "sequence_length": pair.get("sequence_length", 0),
        "insert_length": pair.get("insert_length", 0),
        "backbone_length": pair.get("backbone_length", 0),
        "metadata": pair.get("metadata", {}),
    }


def save_jsonl(pairs: List[Dict], output_path: Path, simple: bool = True):
    """Save pairs as JSONL file."""
    with open(output_path, "w") as f:
        for pair in pairs:
            record = pair_to_simple_format(pair) if simple else pair_to_full_format(pair)
            f.write(json.dumps(record) + "\n")


def save_parquet(pairs: List[Dict], output_path: Path):
    """Save pairs as Parquet file."""
    records = []
    for pair in pairs:
        record = {
            "prompt": pair["prompt"],
            "response": pair["response"],
            "plasmid_id": pair.get("plasmid_id", ""),
            "source": pair.get("source", ""),
            "insert_mode": pair.get("insert_mode", ""),
            "sequence_length": pair.get("sequence_length", 0),
            "insert_length": pair.get("insert_length", 0),
            "backbone_length": pair.get("backbone_length", 0),
        }
        metadata = pair.get("metadata", {})
        record["plasmid_type"] = metadata.get("plasmid_type", "")
        record["resistance_markers"] = ",".join(metadata.get("resistance_markers", []))
        record["copy_number"] = metadata.get("copy_number", "")
        record["tags"] = ",".join(metadata.get("tags", []))
        record["host"] = metadata.get("host", "")
        record["topology"] = metadata.get("topology", "")
        record["annotation_confidence"] = metadata.get("annotation_confidence", 0)
        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)


def save_hf_format(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    test_pairs: List[Dict],
    output_dir: Path
):
    """Save in HuggingFace datasets format."""
    try:
        from datasets import Dataset, DatasetDict

        def pairs_to_hf(pairs):
            return {
                "prompt": [p["prompt"] for p in pairs],
                "response": [p["response"] for p in pairs],
                "plasmid_id": [p.get("plasmid_id", "") for p in pairs],
                "source": [p.get("source", "") for p in pairs],
            }

        dataset_dict = DatasetDict({
            "train": Dataset.from_dict(pairs_to_hf(train_pairs)),
            "validation": Dataset.from_dict(pairs_to_hf(val_pairs)),
            "test": Dataset.from_dict(pairs_to_hf(test_pairs)),
        })
        dataset_dict.save_to_disk(str(output_dir / "hf_dataset"))
    except ImportError:
        pass


def calculate_statistics(pairs: List[Dict], name: str) -> Dict[str, Any]:
    """Calculate statistics for a dataset split."""
    if not pairs:
        return {"name": name, "count": 0}

    stats = {"name": name, "count": len(pairs)}

    seq_lengths = [p.get("sequence_length", 0) for p in pairs]
    backbone_lengths = [p.get("backbone_length", 0) for p in pairs]
    insert_lengths = [p.get("insert_length", 0) for p in pairs if p.get("insert_length", 0) > 0]

    stats["sequence_length"] = {
        "min": int(min(seq_lengths)) if seq_lengths else 0,
        "max": int(max(seq_lengths)) if seq_lengths else 0,
        "mean": float(np.mean(seq_lengths)) if seq_lengths else 0,
        "median": float(np.median(seq_lengths)) if seq_lengths else 0,
    }
    stats["backbone_length"] = {
        "min": int(min(backbone_lengths)) if backbone_lengths else 0,
        "max": int(max(backbone_lengths)) if backbone_lengths else 0,
        "mean": float(np.mean(backbone_lengths)) if backbone_lengths else 0,
        "median": float(np.median(backbone_lengths)) if backbone_lengths else 0,
    }
    if insert_lengths:
        stats["insert_length"] = {
            "count_with_insert": len(insert_lengths),
            "min": int(min(insert_lengths)),
            "max": int(max(insert_lengths)),
            "mean": float(np.mean(insert_lengths)),
            "median": float(np.median(insert_lengths)),
        }

    source_counts = defaultdict(int)
    for p in pairs:
        source_counts[p.get("source", "unknown")] += 1
    stats["by_source"] = dict(source_counts)

    type_counts = defaultdict(int)
    for p in pairs:
        ptype = p.get("metadata", {}).get("plasmid_type", "unknown") or "unknown"
        type_counts[ptype] += 1
    stats["by_plasmid_type"] = dict(type_counts)

    mode_counts = defaultdict(int)
    for p in pairs:
        mode_counts[p.get("insert_mode", "unknown")] += 1
    stats["by_insert_mode"] = dict(mode_counts)

    return stats


def get_token_vocabulary() -> Dict[str, Any]:
    """Get the vocabulary of special tokens."""
    return {
        "tokens": {
            "TYPE": ["mammalian_expression", "bacterial_expression", "yeast_expression",
                     "insect_expression", "plant_expression", "lentiviral", "retroviral",
                     "adenoviral", "aav", "crispr", "cloning", "shuttle", "unknown"],
            "RES": ["ampicillin", "kanamycin", "chloramphenicol", "tetracycline",
                    "spectinomycin", "streptomycin", "gentamicin", "hygromycin",
                    "puromycin", "blasticidin", "zeocin", "erythromycin", "trimethoprim",
                    "multiple", "unknown"],
            "COPY": ["high", "medium", "low", "unknown"],
            "TAG": ["his", "flag", "myc", "ha", "gst", "mbp", "strep", "v5", "sumo",
                    "t7", "gfp_tag", "fc", "multiple", "none"],
            "HOST": ["e_coli", "mammalian", "yeast", "insect", "plant", "bacterial",
                     "multiple", "unknown"],
            "TOPO": ["circular", "linear"],
            "LEN": ["small", "medium", "large", "very_large", "mega"],
        },
        "special_tokens": ["<INSERT>", "<INSERT_1>", "<INSERT_2>", "<INSERT_3>"],
    }
