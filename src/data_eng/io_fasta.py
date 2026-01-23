from __future__ import annotations

from io import StringIO
from typing import Iterator, Tuple

import pandas as pd
from Bio import SeqIO
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, types as T


FASTA_ROW_SCHEMA = T.StructType(
    [
        T.StructField("nuccore_acc", T.StringType(), False),
        T.StructField("sequence", T.StringType(), False),
        T.StructField("seq_length", T.IntegerType(), False),
        T.StructField("header", T.StringType(), True),
        T.StructField("source_file", T.StringType(), True),
    ]
)


def _parse_fasta_binary_rows(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    pdf rows: path, content(bytes)
    yields rows: nuccore_acc, sequence, seq_length, header, source_file
    """
    for pdf in pdf_iter:
        out = []
        for _, r in pdf.iterrows():
            path = r["path"]
            content_bytes = r["content"]
            text = content_bytes.decode("utf-8", errors="replace")
            handle = StringIO(text)
            for rec in SeqIO.parse(handle, "fasta"):
                acc = rec.id.split()[0]
                seq = str(rec.seq).upper()
                out.append((acc, seq, len(seq), rec.description, path))
        yield pd.DataFrame(out, columns=[f.name for f in FASTA_ROW_SCHEMA.fields])


def ingest_fasta_dir(
    spark,
    fasta_dir: str,
    path_glob_filter: str = "sequences_*.fasta",
) -> DataFrame:
    """
    Reads all FASTA parts as binaryFile and parses on executors.
    """
    files_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", path_glob_filter)
        .load(fasta_dir)
        .select("path", "content")
    )

    seq_df = files_df.mapInPandas(_parse_fasta_binary_rows, schema=FASTA_ROW_SCHEMA)

    seq_df = (
        seq_df.withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("is_valid_sequence", F.col("sequence").rlike("^[ATCGN]+$"))
        .withColumn(
            "gc_content",
            (F.length(F.regexp_replace(F.col("sequence"), "[AT]", "")) / F.col("seq_length")).cast(
                "double"
            ),
        )
        .dropDuplicates(["nuccore_acc"])
    )
    return seq_df

