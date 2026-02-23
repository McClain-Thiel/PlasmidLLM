#!/usr/bin/env python3
"""Export all tables from Databricks addgene.default to S3 as parquet files."""

import json
import subprocess
import tempfile
import os
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq

WAREHOUSE_ID = "541a0be6f3f6333c"
PROFILE = "aws"
CATALOG = "addgene"
SCHEMA = "default"
S3_DEST = "s3://phd-research-storage-1758274488/databricks_export"
BATCH_SIZE = 10_000

# Tables with large text columns (sequences) need smaller batches to stay under 25MB inline limit
SMALL_BATCH_TABLES = {
    "full_sequences": 500,
    "partial_sequences": 500,
    "training_pairs": 500,
    "training_pairs_v4": 500,
    "training_bacterial_leq_16k": 500,
    "training_bacterial_leq_4k": 1000,
    "training_bacterial_leq_8k": 500,
    "training_combined_leq_16k": 500,
    "training_combined_leq_4k": 1000,
    "training_combined_leq_8k": 500,
    "training_mammalian_leq_16k": 500,
    "training_mammalian_leq_4k": 1000,
    "training_mammalian_leq_8k": 500,
}

TABLES = [
    "annotations",
    "full_sequences",
    "metadata",
    "motif_registry",
    "partial_sequences",
    "plasmidkit_annotations",
    "training_bacterial_leq_16k",
    "training_bacterial_leq_4k",
    "training_bacterial_leq_8k",
    "training_combined_leq_16k",
    "training_combined_leq_4k",
    "training_combined_leq_8k",
    "training_mammalian_leq_16k",
    "training_mammalian_leq_4k",
    "training_mammalian_leq_8k",
    "training_pairs",
    "training_pairs_v4",
]

TABLE_ROWS = {
    "annotations": 3856316,
    "full_sequences": 154149,
    "metadata": 167548,
    "motif_registry": 360,
    "partial_sequences": 131580,
    "plasmidkit_annotations": 2079935,
    "training_bacterial_leq_16k": 2995,
    "training_bacterial_leq_4k": 7122,
    "training_bacterial_leq_8k": 20004,
    "training_combined_leq_16k": 33769,
    "training_combined_leq_4k": 49235,
    "training_combined_leq_8k": 108343,
    "training_mammalian_leq_16k": 22722,
    "training_mammalian_leq_4k": 25131,
    "training_mammalian_leq_8k": 57562,
    "training_pairs": 108468,
    "training_pairs_v4": 108468,
}


def run_sql(statement, row_limit=BATCH_SIZE, retries=3):
    payload = json.dumps({
        "warehouse_id": WAREHOUSE_ID,
        "statement": statement,
        "wait_timeout": "50s",
        "row_limit": row_limit,
    })
    for attempt in range(retries):
        result = subprocess.run(
            ["databricks", "-p", PROFILE, "api", "post", "/api/2.0/sql/statements", "--json", payload],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}] SQL call failed, retrying in 5s...")
                time.sleep(5)
                continue
            raise RuntimeError(f"SQL failed after {retries} attempts: {result.stderr}")
        resp = json.loads(result.stdout)
        if resp["status"]["state"] == "SUCCEEDED":
            return resp
        if attempt < retries - 1:
            print(f"  [retry {attempt+1}] Query state={resp['status']['state']}, retrying in 5s...")
            time.sleep(5)
            continue
        return resp
    return resp


def fetch_chunk(statement_id, chunk_index, retries=3):
    for attempt in range(retries):
        result = subprocess.run(
            ["databricks", "-p", PROFILE, "api", "get",
             f"/api/2.0/sql/statements/{statement_id}/result/chunks/{chunk_index}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        if attempt < retries - 1:
            print(f"  [retry {attempt+1}] chunk fetch failed, retrying in 5s...")
            time.sleep(5)
            continue
        raise RuntimeError(f"Chunk fetch failed after {retries} attempts: {result.stderr}")


def get_schema(table):
    resp = run_sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.{table} LIMIT 0", row_limit=1)
    if resp["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(f"Schema query failed: {resp}")
    return resp["manifest"]["schema"]["columns"]


def type_map(dbx_type):
    t = dbx_type.upper()
    if t in ("STRING", "VARCHAR", "CHAR"):
        return pa.string()
    if t in ("INT", "INTEGER"):
        return pa.int32()
    if t in ("BIGINT", "LONG"):
        return pa.int64()
    if t in ("SMALLINT", "SHORT"):
        return pa.int16()
    if t in ("TINYINT", "BYTE"):
        return pa.int8()
    if t in ("FLOAT",):
        return pa.float32()
    if t in ("DOUBLE", "DECIMAL"):
        return pa.float64()
    if t in ("BOOLEAN",):
        return pa.bool_()
    if t in ("DATE",):
        return pa.date32()
    if t in ("TIMESTAMP", "TIMESTAMP_NTZ"):
        return pa.timestamp("us")
    if t in ("BINARY",):
        return pa.binary()
    return pa.string()


def rows_to_arrow(rows, columns, arrow_schema):
    arrow_columns = {}
    for i, col in enumerate(columns):
        raw = [row[i] for row in rows]
        t = col["type_name"].upper()
        if t in ("INT", "INTEGER", "BIGINT", "LONG", "SMALLINT", "SHORT", "TINYINT", "BYTE"):
            converted = [int(v) if v is not None else None for v in raw]
        elif t in ("FLOAT", "DOUBLE", "DECIMAL"):
            converted = [float(v) if v is not None else None for v in raw]
        elif t in ("BOOLEAN",):
            converted = [v == "true" if v is not None else None for v in raw]
        else:
            converted = raw
        arrow_columns[col["name"]] = converted
    return pa.table(arrow_columns, schema=arrow_schema)


def export_table(table):
    total_expected = TABLE_ROWS.get(table, 0)
    print(f"\n{'='*60}")
    print(f"Exporting: {CATALOG}.{SCHEMA}.{table} (~{total_expected:,} rows)")

    columns = get_schema(table)
    col_names = [c["name"] for c in columns]
    arrow_schema = pa.schema([(c["name"], type_map(c["type_name"])) for c in columns])
    batch_size = SMALL_BATCH_TABLES.get(table, BATCH_SIZE)
    print(f"  Columns ({len(col_names)}): {', '.join(col_names[:5])}{'...' if len(col_names) > 5 else ''}")
    print(f"  Batch size: {batch_size:,}")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name

    writer = None
    total_fetched = 0
    offset = 0
    t0 = time.time()

    try:
        while True:
            sql = f"SELECT * FROM {CATALOG}.{SCHEMA}.{table} LIMIT {batch_size} OFFSET {offset}"
            pct = (offset / total_expected * 100) if total_expected > 0 else 0
            print(f"  [{pct:5.1f}%] Fetching rows {offset:,}..{offset + batch_size:,}...", end=" ", flush=True)

            resp = run_sql(sql, row_limit=batch_size)

            if resp["status"]["state"] != "SUCCEEDED":
                raise RuntimeError(f"Query failed: {resp['status']}")

            rows = resp["result"]["data_array"]
            total_chunks = resp["manifest"]["total_chunk_count"]
            for ci in range(1, total_chunks):
                chunk_resp = fetch_chunk(resp["statement_id"], ci)
                rows.extend(chunk_resp["data_array"])

            batch_count = len(rows)
            total_fetched += batch_count
            elapsed = time.time() - t0
            rate = total_fetched / elapsed if elapsed > 0 else 0
            print(f"{batch_count:,} rows (total: {total_fetched:,}, {rate:.0f} rows/s)")

            batch_table = rows_to_arrow(rows, columns, arrow_schema)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, arrow_schema, compression="snappy")
            writer.write_table(batch_table)

            if batch_count < batch_size:
                break
            offset += batch_size

    finally:
        if writer:
            writer.close()

    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  Total: {total_fetched:,} rows, {size_mb:.1f} MB parquet, {elapsed:.0f}s")

    s3_path = f"{S3_DEST}/{table}.parquet"
    print(f"  Uploading to {s3_path}...")
    subprocess.run(["aws", "s3", "cp", tmp_path, s3_path], check=True)
    os.unlink(tmp_path)
    print(f"  Done!")


if __name__ == "__main__":
    start_from = sys.argv[1] if len(sys.argv) > 1 else None
    tables = TABLES
    if start_from:
        if start_from in TABLES:
            idx = TABLES.index(start_from)
            tables = TABLES[idx:]
            print(f"Resuming from table: {start_from}")
        else:
            print(f"Unknown table: {start_from}")
            sys.exit(1)

    total_rows = sum(TABLE_ROWS.get(t, 0) for t in tables)
    print(f"Exporting {len(tables)} tables ({total_rows:,} total rows) from {CATALOG}.{SCHEMA} -> {S3_DEST}/")

    for i, table in enumerate(tables):
        print(f"\n[Table {i+1}/{len(tables)}]", end="")
        export_table(table)

    print(f"\n{'='*60}")
    print("All tables exported successfully!")
