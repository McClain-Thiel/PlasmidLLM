from __future__ import annotations

import json
from typing import Dict, Tuple

import yaml
from pyspark.sql import DataFrame


def build_token_frequencies(
    spark,
    silver_amr_fq: str,
    silver_taxonomy_fq: str,
    silver_replicons_fq: str,
    gold_token_frequencies_fq: str,
    token_cfg_path: str,
) -> None:
    cfg = yaml.safe_load(open(token_cfg_path, "r"))
    thr = cfg["thresholds"]
    critical = cfg["critical_amr"]

    amr_thr = int(thr["amr_min_freq"])
    host_thr = int(thr["host_min_freq"])
    rep_thr = int(thr["rep_min_freq"])

    critical_list_sql = ",".join([f"'{x}'" for x in critical])

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {gold_token_frequencies_fq} AS

        -- AMR
        SELECT
          'AMR' AS category,
          gene_symbol AS token_value,
          COUNT(DISTINCT accession) AS frequency,
          CASE
            WHEN COUNT(DISTINCT accession) >= {amr_thr} THEN 'frequent'
            WHEN gene_symbol IN ({critical_list_sql}) THEN 'critical'
            ELSE 'rare'
          END AS inclusion_status
        FROM {silver_amr_fq}
        LATERAL VIEW EXPLODE(amr_genes) AS gene_symbol
        GROUP BY gene_symbol
        HAVING inclusion_status IN ('frequent','critical')

        UNION ALL

        -- HOST
        SELECT
          'HOST' AS category,
          genus AS token_value,
          COUNT(DISTINCT accession) AS frequency,
          'included' AS inclusion_status
        FROM {silver_taxonomy_fq}
        WHERE genus IS NOT NULL
        GROUP BY genus
        HAVING COUNT(DISTINCT accession) >= {host_thr}

        UNION ALL

        -- REP
        SELECT
          'REP' AS category,
          TRIM(replicon) AS token_value,
          COUNT(DISTINCT accession) AS frequency,
          'included' AS inclusion_status
        FROM {silver_replicons_fq}
        LATERAL VIEW EXPLODE(replicon_types) AS replicon
        WHERE replicon IS NOT NULL AND TRIM(replicon) != ''
        GROUP BY TRIM(replicon)
        HAVING COUNT(DISTINCT accession) >= {rep_thr}
        """
    )


def build_token_vocabulary(
    spark,
    gold_token_frequencies_fq: str,
    gold_token_vocab_fq: str,
    token_cfg_path: str,
) -> None:
    cfg = yaml.safe_load(open(token_cfg_path, "r"))
    meta = cfg["meta_tokens"]

    # build meta token rows in SQL
    meta_rows = "\nUNION ALL\n".join(
        [f"SELECT '{m['category']}' AS category, '{m['value']}' AS token_value, 'meta' AS token_type"
         for m in meta]
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {gold_token_vocab_fq} AS
        WITH meta_tokens AS (
          {meta_rows}
        ),
        meta_indexed AS (
          SELECT
            ROW_NUMBER() OVER (ORDER BY category, token_value) - 1 AS token_id,
            category,
            token_value,
            token_type,
            CAST(NULL AS BIGINT) AS frequency,
            CONCAT('<', category, ':', token_value, '>') AS token_string
          FROM meta_tokens
        ),
        specific_indexed AS (
          SELECT
            ROW_NUMBER() OVER (ORDER BY category, token_value) - 1 AS rn,
            category,
            token_value,
            'specific' AS token_type,
            frequency,
            CONCAT('<', category, ':', token_value, '>') AS token_string
          FROM {gold_token_frequencies_fq}
        ),
        offsets AS (
          SELECT (SELECT MAX(token_id) + 1 FROM meta_indexed) AS off
        )
        SELECT * FROM meta_indexed
        UNION ALL
        SELECT
          s.rn + o.off AS token_id,
          s.category,
          s.token_value,
          s.token_type,
          s.frequency,
          s.token_string
        FROM specific_indexed s
        CROSS JOIN offsets o
        """
    )


def export_vocab_json(spark, gold_token_vocab_fq: str, output_path: str) -> None:
    """
    output_path: /Volumes/.../token_vocabulary.json
    """
    df = spark.table(gold_token_vocab_fq).toPandas()
    vocab_dict = df.set_index("token_string").to_dict("index")

    local_path = output_path.replace("/Volumes", "/dbfs/Volumes")
    with open(local_path, "w") as f:
        json.dump(vocab_dict, f, indent=2)

