from __future__ import annotations

import hashlib
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, types as T


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def build_prompt_variants(tags_df: DataFrame, num_variants: int = 4) -> DataFrame:
    """
    Input: accession, base_tags (array<string>)
    Output: accession, prompt, prompt_variant_id, weight
    """
    # deterministic shuffle by hashing tag + accession + variant_id
    @F.udf(T.ArrayType(T.StringType()))
    def shuffle_tags(tags, accession, variant_id):
        if tags is None:
            return []
        decorated = []
        for t in tags:
            hv = _stable_int(f"{accession}|{variant_id}|{t}")
            decorated.append((hv, t))
        decorated.sort(key=lambda x: x[0])
        return [t for _, t in decorated]

    templates = [
        lambda tag_str: f"{tag_str}\nSEQUENCE:",
        lambda tag_str: f"Design a plasmid with: {tag_str}\nSequence:",
        lambda tag_str: f"Constraints: {tag_str}\nOutput DNA:",
        lambda tag_str: f"{tag_str}\nDNA:",
    ]

    # Make a small DataFrame of template ids
    spark = tags_df.sparkSession
    tmpl_df = spark.createDataFrame([(i,) for i in range(min(num_variants, len(templates)))], ["prompt_variant_id"])

    base = tags_df.select("accession", "base_tags").crossJoin(tmpl_df)

    # shuffle tags per variant
    base = base.withColumn(
        "tags_variant",
        shuffle_tags(F.col("base_tags"), F.col("accession"), F.col("prompt_variant_id"))
    )

    # optionally drop GC or AMR tags in some variants (deterministic)
    base = base.withColumn(
        "tags_variant",
        F.expr("""
          filter(tags_variant, t ->
            NOT (
              (prompt_variant_id = 1 AND t LIKE '<GC:%>') OR
              (prompt_variant_id = 2 AND t LIKE '<AMR:%>')
            )
          )
        """)
    )

    # join tags into string
    base = base.withColumn("tag_string", F.concat_ws(" ", F.col("tags_variant")))

    # template map (pure SQL CASE)
    base = base.withColumn(
        "prompt",
        F.expr("""
          CASE prompt_variant_id
            WHEN 0 THEN concat(tag_string, '\nSEQUENCE:')
            WHEN 1 THEN concat('Design a plasmid with: ', tag_string, '\nSequence:')
            WHEN 2 THEN concat('Constraints: ', tag_string, '\nOutput DNA:')
            ELSE concat(tag_string, '\nDNA:')
          END
        """)
    )

    return base.select(
        "accession",
        "prompt",
        "prompt_variant_id",
        F.lit(1.0).alias("weight"),
    )

