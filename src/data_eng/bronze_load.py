from __future__ import annotations

from typing import Optional, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def write_table(df: DataFrame, table_fq: str, mode: str = "overwrite") -> None:
    (
        df.write.mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(table_fq)
    )


def validate_table_basic(spark, table_fq: str) -> DataFrame:
    return spark.sql(
        f"""
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN is_valid_sequence = false THEN 1 ELSE 0 END) AS invalid,
          MIN(seq_length) AS min_len,
          percentile_approx(seq_length, 0.5) AS median_len,
          MAX(seq_length) AS max_len
        FROM {table_fq}
        """
    )


def load_delimited_to_bronze(
    spark,
    input_path: str,
    output_table_fq: str,
    fmt: str,
    options: Optional[Dict[str, str]] = None,
    mode: str = "overwrite",
) -> None:
    """
    Utility loader for nuccore/taxonomy/amr/typing if you have them in Volumes.
    fmt: "csv" | "parquet" | "json"
    """
    options = options or {}
    reader = spark.read.format(fmt)
    for k, v in options.items():
        reader = reader.option(k, v)

    df = reader.load(input_path)

    (
        df.write.mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(output_table_fq)
    )

