from __future__ import annotations

from pyspark.sql import functions as F


def build_silver_tables(
    spark,
    bronze_nuccore_fq: str,
    bronze_taxonomy_fq: str,
    bronze_amr_fq: str,
    bronze_typing_fq: str,
    silver_metadata_fq: str,
    silver_taxonomy_fq: str,
    silver_amr_fq: str,
    silver_replicons_fq: str,
    silver_mobility_fq: str,
) -> None:
    """
    NOTE: Column names in your bronze tables must match the SQL below.
    If your PLSDB export uses different names, change them here once.
    """

    # --- plasmid_metadata
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {silver_metadata_fq} AS
        SELECT
          n.NUCCORE_ACC AS accession,
          n.NUCCORE_Description AS description,
          n.NUCCORE_Length AS length,
          n.NUCCORE_GC AS gc_content,
          n.NUCCORE_Topology AS topology,
          n.NUCCORE_CreateDate AS create_date,
          n.NUCCORE_Completeness AS completeness,

          CASE
            WHEN n.NUCCORE_Length < 5000 THEN 'small'
            WHEN n.NUCCORE_Length < 50000 THEN 'medium'
            ELSE 'large'
          END AS size_bin,

          CASE
            WHEN n.NUCCORE_GC < 0.40 THEN 'low'
            WHEN n.NUCCORE_GC < 0.60 THEN 'mid'
            ELSE 'high'
          END AS gc_bin,

          CASE
            WHEN n.NUCCORE_Length < 1000 OR n.NUCCORE_Length > 500000 THEN true
            ELSE false
          END AS is_extreme_length,

          current_timestamp() AS processed_at
        FROM {bronze_nuccore_fq} n
        WHERE n.NUCCORE_Genome = 'plasmid'
          AND n.NUCCORE_Completeness = 'complete'
        """
    )

    # --- plasmid_taxonomy
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {silver_taxonomy_fq} AS
        SELECT
          n.NUCCORE_ACC AS accession,
          SPLIT(t.TAXONOMY_taxon_name, ' ')[0] AS genus,
          t.TAXONOMY_taxon_name AS species,
          t.TAXONOMY_phylum AS phylum,
          t.TAXONOMY_class AS class,
          t.TAXONOMY_order AS order_name,
          t.TAXONOMY_family AS family,
          t.TAXONOMY_superkingdom AS kingdom,
          t.TAXONOMY_taxon_lineage AS lineage,
          current_timestamp() AS processed_at
        FROM {bronze_nuccore_fq} n
        JOIN {bronze_taxonomy_fq} t
          ON n.TAXONOMY_UID = t.TAXONOMY_UID
        WHERE n.NUCCORE_Genome = 'plasmid'
          AND n.NUCCORE_Completeness = 'complete'
        """
    )

    # --- plasmid_amr (aggregated)
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {silver_amr_fq} AS
        SELECT
          NUCCORE_ACC AS accession,
          COLLECT_LIST(gene_symbol) AS amr_genes,
          COUNT(DISTINCT gene_symbol) AS num_amr_genes,
          COLLECT_SET(drug_class) AS drug_classes,
          COUNT(DISTINCT drug_class) AS num_drug_classes,
          AVG(sequence_identity) AS avg_identity,
          AVG(coverage_percentage) AS avg_coverage,
          current_timestamp() AS processed_at
        FROM {bronze_amr_fq}
        GROUP BY NUCCORE_ACC
        """
    )

    # --- plasmid_replicons
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {silver_replicons_fq} AS
        SELECT
          NUCCORE_ACC AS accession,
          SPLIT(REGEXP_REPLACE(`rep_type(s)`, '[()]', ''), ',') AS replicon_types,
          SIZE(SPLIT(REGEXP_REPLACE(`rep_type(s)`, '[()]', ''), ',')) AS num_replicons,
          current_timestamp() AS processed_at
        FROM {bronze_typing_fq}
        WHERE `rep_type(s)` IS NOT NULL
        """
    )

    # --- plasmid_mobility
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {silver_mobility_fq} AS
        SELECT
          NUCCORE_ACC AS accession,
          predicted_mobility AS mobility_type,
          CASE
            WHEN `relaxase_type(s)` IS NOT NULL
            THEN SPLIT(REGEXP_REPLACE(`relaxase_type(s)`, '[()]', ''), ',')
            ELSE ARRAY()
          END AS relaxase_types,
          CASE
            WHEN predicted_mobility = 'conjugative' THEN 'conjugative'
            WHEN predicted_mobility = 'mobilizable' THEN 'mobilizable'
            WHEN predicted_mobility = 'non-mobilizable' THEN 'non-mobilizable'
            ELSE 'unknown'
          END AS mobility_category,
          current_timestamp() AS processed_at
        FROM {bronze_typing_fq}
        """
    )

