from __future__ import annotations

import hashlib
import json
from typing import Dict, Set, List

import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, types as T


def _load_vocab_specific_values(vocab_json_path: str) -> Dict[str, Set[str]]:
    """
    Reads token_vocabulary.json and returns category -> set(token_value) for specific tokens
    """
    local_path = vocab_json_path.replace("/Volumes", "/dbfs/Volumes")
    with open(local_path, "r") as f:
        vocab = json.load(f)

    out: Dict[str, Set[str]] = {}
    for _, info in vocab.items():
        cat = info["category"]
        if info["token_type"] != "specific":
            continue
        out.setdefault(cat, set()).add(info["token_value"])
    return out


def assign_tags_df(
    spark,
    features_df: DataFrame,
    vocab_json_path: str,
    token_cfg_path: str,
) -> DataFrame:
    cfg = yaml.safe_load(open(token_cfg_path, "r"))
    tagging_cfg = cfg["tagging"]
    gc_denom = int(tagging_cfg["include_gc_prob_denom"])
    amr_none_denom = int(tagging_cfg["include_amr_none_prob_denom"])

    vocab_by_cat = _load_vocab_specific_values(vocab_json_path)
    bc = spark.sparkContext.broadcast(vocab_by_cat)

    @F.udf(T.ArrayType(T.StringType()))
    def _assign(host_genus, size_bin, gc_bin, amr_genes, has_no_amr, replicon_types, mobility_category):
        tags: List[str] = []
        vocab = bc.value

        # deterministic hash seed
        seed = f"{host_genus}|{size_bin}|{mobility_category}"
        hv = int(hashlib.md5(seed.encode()).hexdigest(), 16)

        # HOST
        if host_genus and host_genus in vocab.get("HOST", set()):
            tags.append(f"<HOST:{host_genus}>")
        else:
            tags.append("<HOST:any>")

        # SIZE
        tags.append(f"<SIZE:{size_bin}>")

        # GC (deterministic include 1/gc_denom)
        if gc_bin and (hv % gc_denom == 0):
            tags.append(f"<GC:{gc_bin}>")

        # AMR
        if has_no_amr:
            if hv % amr_none_denom == 0:
                tags.append("<AMR:none>")
        else:
            if amr_genes:
                in_vocab = [g for g in amr_genes if g in vocab.get("AMR", set())]
                if not in_vocab:
                    tags.append("<AMR:any>")
                else:
                    for g in in_vocab:
                        tags.append(f"<AMR:{g}>")

        # REP (include primary if in vocab else any)
        if replicon_types:
            rep_in = [r.strip() for r in replicon_types if r and r.strip() in vocab.get("REP", set())]
            if rep_in:
                tags.append(f"<REP:{rep_in[0]}>")
            else:
                tags.append("<REP:any>")
        else:
            tags.append("<REP:any>")

        # MOB
        if mobility_category:
            tags.append(f"<MOB:{mobility_category}>")
        else:
            tags.append("<MOB:any>")

        return tags

    return (
        features_df
        .withColumn("base_tags", _assign(
            F.col("host_genus"),
            F.col("size_bin"),
            F.col("gc_bin"),
            F.col("amr_genes"),
            F.col("has_no_amr"),
            F.col("replicon_types"),
            F.col("mobility_category"),
        ))
        .withColumn("num_base_tags", F.size(F.col("base_tags")))
    )

