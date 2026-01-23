from __future__ import annotations


def build_gold_features(
    spark,
    silver_metadata_fq: str,
    silver_taxonomy_fq: str,
    silver_amr_fq: str,
    silver_replicons_fq: str,
    silver_mobility_fq: str,
    gold_features_fq: str,
    gold_features_filtered_fq: str,
) -> None:
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {gold_features_fq} AS
        SELECT
          meta.accession,
          meta.description,

          meta.length,
          meta.gc_content,
          meta.topology,
          meta.size_bin,
          meta.gc_bin,
          meta.is_extreme_length,

          tax.genus AS host_genus,
          tax.species AS host_species,
          tax.phylum AS host_phylum,

          COALESCE(amr.amr_genes, ARRAY()) AS amr_genes,
          COALESCE(amr.num_amr_genes, 0) AS num_amr_genes,
          COALESCE(amr.drug_classes, ARRAY()) AS drug_classes,
          CASE WHEN amr.accession IS NULL THEN true ELSE false END AS has_no_amr,

          COALESCE(rep.replicon_types, ARRAY()) AS replicon_types,
          COALESCE(rep.num_replicons, 0) AS num_replicons,

          mob.mobility_category,
          mob.relaxase_types,

          current_timestamp() AS processed_at

        FROM {silver_metadata_fq} meta
        LEFT JOIN {silver_taxonomy_fq} tax
          ON meta.accession = tax.accession
        LEFT JOIN {silver_amr_fq} amr
          ON meta.accession = amr.accession
        LEFT JOIN {silver_replicons_fq} rep
          ON meta.accession = rep.accession
        LEFT JOIN {silver_mobility_fq} mob
          ON meta.accession = mob.accession
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {gold_features_filtered_fq} AS
        SELECT *
        FROM {gold_features_fq}
        WHERE is_extreme_length = false
          AND host_genus IS NOT NULL
          AND length BETWEEN 1000 AND 500000
        """
    )

