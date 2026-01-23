from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from pyspark.sql import SparkSession

from src.data_eng.settings import load_env_config
from src.data_eng.paths import TableNames
from src.data_eng import io_fasta, bronze_load, silver_build, gold_vocab, gold_features


def execute_sql_file(spark: SparkSession, sql_path: Path) -> None:
    text = sql_path.read_text()
    # naive split by semicolon; ignore comments and empties
    parts = [p.strip() for p in text.split(";")]
    for stmt in parts:
        if not stmt:
            continue
        # drop leading SQL single-line comments
        lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
        stmt_clean = "\n".join(lines).strip()
        if stmt_clean:
            spark.sql(stmt_clean)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_env_path = repo_root / "config" / "env.yaml"
    cfg_token_path = repo_root / "config" / "token_config.yaml"

    spark = (
        SparkSession.builder.appName("plasmid-data-eng-pipeline").getOrCreate()
    )

    env = load_env_config(str(cfg_env_path))
    tn = TableNames(env)

    print("[DE] Creating catalog/schemas (if needed)...")
    execute_sql_file(spark, repo_root / "sql" / "00_create_schemas.sql")

    # Bronze: FASTA ingestion
    fasta_dir = env.volumes.get("fasta_dir")
    if fasta_dir:
        print(f"[DE] Ingesting FASTA from {fasta_dir} ...")
        seq_df = io_fasta.ingest_fasta_dir(spark, fasta_dir)
        bronze_sequences_fq = tn.fq_bronze("bronze_sequences")
        bronze_load.write_table(seq_df, bronze_sequences_fq, mode="overwrite")
        print(f"[DE] Wrote bronze sequences: {bronze_sequences_fq}")
        bronze_load.validate_table_basic(spark, bronze_sequences_fq).show(truncate=False)
    else:
        print("[DE][WARN] volumes.fasta_dir not set; skipping FASTA ingestion.")

    # Silver: build curated tables (requires bronze inputs present)
    print("[DE] Building silver tables ...")
    silver_build.build_silver_tables(
        spark,
        bronze_nuccore_fq=tn.fq_bronze("bronze_nuccore"),
        bronze_taxonomy_fq=tn.fq_bronze("bronze_taxonomy"),
        bronze_amr_fq=tn.fq_bronze("bronze_amr"),
        bronze_typing_fq=tn.fq_bronze("bronze_typing"),
        silver_metadata_fq=tn.fq_silver("silver_plasmid_metadata"),
        silver_taxonomy_fq=tn.fq_silver("silver_plasmid_taxonomy"),
        silver_amr_fq=tn.fq_silver("silver_plasmid_amr"),
        silver_replicons_fq=tn.fq_silver("silver_plasmid_replicons"),
        silver_mobility_fq=tn.fq_silver("silver_plasmid_mobility"),
    )

    # Gold: vocabulary
    print("[DE] Building gold token frequencies + vocabulary ...")
    gold_vocab.build_token_frequencies(
        spark,
        silver_amr_fq=tn.fq_silver("silver_plasmid_amr"),
        silver_taxonomy_fq=tn.fq_silver("silver_plasmid_taxonomy"),
        silver_replicons_fq=tn.fq_silver("silver_plasmid_replicons"),
        gold_token_frequencies_fq=tn.fq_gold("gold_token_frequencies"),
        token_cfg_path=str(cfg_token_path),
    )

    gold_vocab.build_token_vocabulary(
        spark,
        gold_token_frequencies_fq=tn.fq_gold("gold_token_frequencies"),
        gold_token_vocab_fq=tn.fq_gold("gold_token_vocabulary"),
        token_cfg_path=str(cfg_token_path),
    )

    vocab_json_path = str(Path(env.volumes["gold_config_dir"]) / "token_vocabulary.json")
    gold_vocab.export_vocab_json(
        spark,
        gold_token_vocab_fq=tn.fq_gold("gold_token_vocabulary"),
        output_path=vocab_json_path,
    )
    print(f"[DE] Exported vocabulary JSON to {vocab_json_path}")

    # Gold: features
    print("[DE] Building gold features ...")
    gold_features.build_gold_features(
        spark,
        silver_metadata_fq=tn.fq_silver("silver_plasmid_metadata"),
        silver_taxonomy_fq=tn.fq_silver("silver_plasmid_taxonomy"),
        silver_amr_fq=tn.fq_silver("silver_plasmid_amr"),
        silver_replicons_fq=tn.fq_silver("silver_plasmid_replicons"),
        silver_mobility_fq=tn.fq_silver("silver_plasmid_mobility"),
        gold_features_fq=tn.fq_gold("gold_plasmid_features"),
        gold_features_filtered_fq=tn.fq_gold("gold_plasmid_features_filtered"),
    )

    print("[DE] Pipeline completed.")


if __name__ == "__main__":
    main()

