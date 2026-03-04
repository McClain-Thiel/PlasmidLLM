# Databricks notebook source
# MAGIC %md
# MAGIC # Building Training Pairs for PlasmidGPT
# MAGIC
# MAGIC **Goal:** Create prompt → full-sequence training pairs for pre-training a DNA language model on plasmid design.
# MAGIC
# MAGIC **Prompt format:**
# MAGIC ```
# MAGIC <BOS><VEC_MAMMALIAN><COPY_HIGH><AMR_KANAMYCIN><ORI_PBR322><PROM_CMV><REP_EGFP>...<SEQ>ATGCCC...<EOS>
# MAGIC ```
# MAGIC
# MAGIC **Data sources:**
# MAGIC - `annotations` — BLAST-based feature annotations per plasmid (3.8M rows)
# MAGIC - `metadata` — Addgene metadata per plasmid (167K rows)
# MAGIC - `full_sequences` — Complete plasmid DNA sequences (154K rows)
# MAGIC - `motif_registry` — Canonical motif sequences with token assignments
# MAGIC
# MAGIC **Token types:**
# MAGIC - **Hard tokens** (motif-backed, verifiable in post-training via BLAST): AMR, ORI, PROM, REPORTER, TAG, ELEM
# MAGIC - **Soft tokens** (metadata-derived, guide generation but not directly verifiable): VEC, BACKBONE, COPY, SPECIES
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. EDA
# MAGIC 2. Build token vocabulary (hard + soft)
# MAGIC 3. Assign tokens to plasmids
# MAGIC 4. Construct training pairs
# MAGIC 5. Filter, deduplicate, split
# MAGIC 6. Validation & stats
# MAGIC 7. Export token list for tokenizer
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup & Load Tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, coalesce, concat, concat_ws,
    collect_set, collect_list, countDistinct, count,
    explode, explode_outer, array, array_distinct, array_sort, array_union,
    get_json_object, from_json, schema_of_json, size,
    regexp_replace, upper, trim, lower,
    row_number, dense_rank, percent_rank,
    mean, stddev, min as spark_min, max as spark_max,
    percentile_approx, hash as spark_hash, abs as spark_abs
)
from pyspark.sql.window import Window
from pyspark.sql.types import *
import json


# COMMAND ----------

annotations = spark.read.table("addgene.default.annotations")
metadata = spark.read.table("addgene.default.metadata")
full_sequences = spark.read.table("addgene.default.full_sequences")
partial_seqs = spark.read.table("addgene.default.partial_sequences")
motifs = spark.read.table("addgene.default.motif_registry")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Exploratory Data Analysis
# MAGIC
# MAGIC Quick look at table sizes, schemas, and key distributions before building anything.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Table sizes and schemas

# COMMAND ----------

print("=== Table Counts ===")
for name, df in [("annotations", annotations), ("metadata", metadata), 
                  ("full_sequences", full_sequences), ("partial_seqs", partial_seqs),
                  ("motifs", motifs)]:
    print(f"{name}: {df.count():,} rows")


# COMMAND ----------

print("=== Full Sequences Schema ===")
full_sequences.printSchema()
print("\n=== Annotations Schema ===")
annotations.printSchema()
print("\n=== Motif Registry Schema ===")
motifs.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Sequence length distribution

# COMMAND ----------

# Sequence length stats from full_sequences
seq_stats = full_sequences.select(
    count("*").alias("total"),
    mean("sequence_length").alias("mean_len"),
    stddev("sequence_length").alias("std_len"),
    spark_min("sequence_length").alias("min_len"),
    spark_max("sequence_length").alias("max_len"),
    percentile_approx("sequence_length", 0.25).alias("p25"),
    percentile_approx("sequence_length", 0.50).alias("median"),
    percentile_approx("sequence_length", 0.75).alias("p75"),
    percentile_approx("sequence_length", 0.90).alias("p90"),
    percentile_approx("sequence_length", 0.95).alias("p95"),
    percentile_approx("sequence_length", 0.99).alias("p99"),
)
display(seq_stats)


# COMMAND ----------

# Length buckets for understanding what we lose at each cutoff
length_buckets = (
    full_sequences
    .withColumn("bucket", 
        when(col("sequence_length") <= 4096, "0-4k")
        .when(col("sequence_length") <= 8192, "4k-8k")
        .when(col("sequence_length") <= 16384, "8k-16k")
        .when(col("sequence_length") <= 32768, "16k-32k")
        .otherwise("32k+"))
    .groupBy("bucket")
    .agg(count("*").alias("count"))
    .orderBy("bucket")
)
display(length_buckets)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1c. Motif registry — category & token distribution

# COMMAND ----------

# How many unique tokens per category, and plasmid count ranges
motif_summary = (
    motifs
    .groupBy("category")
    .agg(
        countDistinct("token").alias("unique_tokens"),
        countDistinct("sseqid").alias("unique_sseqids"),
        spark_min("plasmid_count").alias("min_plasmid_count"),
        spark_max("plasmid_count").alias("max_plasmid_count"),
        mean("plasmid_count").alias("mean_plasmid_count"),
    )
    .orderBy(col("unique_tokens").desc())
)
display(motif_summary)


# COMMAND ----------

# All distinct tokens with their plasmid counts
# This is the raw material for the hard token vocabulary
motif_tokens = (
    motifs
    .select("token", "category", "plasmid_count")
    .distinct()
    .orderBy(col("plasmid_count").desc())
)
display(motif_tokens)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1d. Annotation quality distribution

# COMMAND ----------

# pident and percmatch distributions — these determine our quality thresholds
ann_quality = (
    annotations
    .select(
        mean("pident").alias("mean_pident"),
        percentile_approx("pident", [0.05, 0.25, 0.5, 0.75, 0.95]).alias("pident_pctiles"),
        mean("percmatch").alias("mean_percmatch"),
        percentile_approx("percmatch", [0.05, 0.25, 0.5, 0.75, 0.95]).alias("percmatch_pctiles"),
    )
)
display(ann_quality)


# COMMAND ----------

# How many annotations survive various quality thresholds
for pident_thresh in [90, 95, 98]:
    for percmatch_thresh in [70, 80, 90]:
        cnt = annotations.filter(
            (col("pident") >= pident_thresh) & (col("percmatch") >= percmatch_thresh)
        ).count()
        print(f"pident >= {pident_thresh}, percmatch >= {percmatch_thresh}: {cnt:,} annotations")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1e. Metadata — vector types, backbones, copy number

# COMMAND ----------

# Vector types (from cloning JSON) — this is the basis for VEC soft tokens
vector_type_counts = (
    metadata
    .withColumn("vector_types", get_json_object(col("cloning"), "$.vector_types"))
    .filter(col("vector_types").isNotNull())
    .groupBy("vector_types")
    .count()
    .orderBy(col("count").desc())
)
display(vector_type_counts)


# COMMAND ----------

# Backbone distribution — basis for BB soft tokens
backbone_counts = (
    metadata
    .withColumn("backbone", get_json_object(col("cloning"), "$.backbone"))
    .filter(col("backbone").isNotNull() & (col("backbone") != ""))
    .groupBy("backbone")
    .count()
    .orderBy(col("count").desc())
)
display(backbone_counts)


# COMMAND ----------

# Copy number distribution
copy_counts = (
    metadata
    .groupBy("plasmid_copy")
    .count()
    .orderBy(col("count").desc())
)
display(copy_counts)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1f. Insert species distribution

# COMMAND ----------

# Species from inserts JSON — requires UDF due to nested structure
# inserts looks like: [{"species": [[4932, "Saccharomyces cerevisiae"]], ...}]

from pyspark.sql.functions import udf

@udf(ArrayType(StringType()))
def extract_species(inserts_json):
    """Extract species names from the inserts JSON array."""
    if not inserts_json:
        return []
    try:
        inserts = json.loads(inserts_json)
        species = []
        for insert in inserts:
            for sp in insert.get("species", []):
                if isinstance(sp, list) and len(sp) >= 2:
                    species.append(sp[1])  # species name is second element
        return list(set(species))
    except:
        return []

species_df = (
    metadata
    .withColumn("species_list", extract_species(col("inserts")))
    .withColumn("species", explode_outer(col("species_list")))
    .filter(col("species").isNotNull())
    .groupBy("species")
    .count()
    .orderBy(col("count").desc())
)
display(species_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build Token Vocabulary
# MAGIC
# MAGIC Two types of tokens:
# MAGIC - **Hard tokens**: Derived from motif registry. Each has canonical DNA/protein sequences that can be verified in post-training (RL reward signal via BLAST alignment).
# MAGIC - **Soft tokens**: Derived from metadata. Guide generation but aren't directly sequence-verifiable.
# MAGIC
# MAGIC Cardinality strategy: Drop rare tokens entirely (no `_OTHER` fallbacks — if we can't name it specifically, it shouldn't constrain generation).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Hard Token Vocabulary
# MAGIC
# MAGIC Filter motif registry to tokens appearing in >= `MIN_PLASMID_COUNT` plasmids. 
# MAGIC These tokens already exist in the registry's `token` column.
# MAGIC

# COMMAND ----------

# ── CONFIG ──
MIN_PLASMID_COUNT = 50  # minimum plasmids a motif must appear in to get its own token
PIDENT_THRESHOLD = 95.0  # minimum percent identity for annotation match
PERCMATCH_THRESHOLD = 80.0  # minimum percent coverage for annotation match
TOP_N_BACKBONES = 20  # number of backbone tokens to keep
MAX_DUPLICATES_PER_PROMPT = 5  # max sequences per unique prompt


# COMMAND ----------

# Hard token vocabulary: distinct tokens from motif registry above frequency threshold
hard_token_vocab = (
    motifs
    .select("token", "category", "plasmid_count", "uuid")
    .distinct()
    .filter(col("plasmid_count") >= MIN_PLASMID_COUNT)
    .orderBy("category", col("plasmid_count").desc())
)

print(f"Hard token vocabulary size: {hard_token_vocab.count()}")
display(hard_token_vocab)


# COMMAND ----------

# Also build the sseqid -> token lookup for joining with annotations
# A single token (e.g. <AMR_AMPICILLIN>) can have multiple sseqid variants (AmpR, AmpR_(2), etc.)
sseqid_to_token = (
    motifs
    .filter(col("plasmid_count") >= MIN_PLASMID_COUNT)
    .select("sseqid", "token", "category", "uuid", "sequence")
    .distinct()
)

print(f"sseqid -> token lookup size: {sseqid_to_token.count()}")
display(sseqid_to_token.limit(20))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Soft Token — Vector Type (VEC)
# MAGIC
# MAGIC The `vector_types` field is an array embedded in the `cloning` JSON. It has hundreds of unique 
# MAGIC combinations, most with count=1. We canonicalize into ~12 high-level buckets via keyword matching.
# MAGIC
# MAGIC A plasmid can receive **multiple** VEC tokens (e.g., a lentiviral CRISPR mammalian expression vector).
# MAGIC

# COMMAND ----------

# VEC canonicalization map: keyword -> token
VEC_KEYWORD_MAP = {
    "Mammalian Expression": "<VEC_MAMMALIAN>",
    "Bacterial Expression": "<VEC_BACTERIAL>",
    "Lentiviral":          "<VEC_LENTIVIRAL>",
    "Retroviral":          "<VEC_RETROVIRAL>",
    "AAV":                 "<VEC_AAV>",
    "CRISPR":              "<VEC_CRISPR>",
    "Yeast Expression":    "<VEC_YEAST>",
    "Insect Expression":   "<VEC_INSECT>",
    "Plant Expression":    "<VEC_PLANT>",
    "Gateway":             "<VEC_GATEWAY>",
    "Entry":               "<VEC_GATEWAY>",
    "Destination":         "<VEC_GATEWAY>",
    "Reporter":            "<VEC_REPORTER>",
}

@udf(ArrayType(StringType()))
def assign_vec_tokens(cloning_json):
    """Parse vector_types from cloning JSON and map to canonical VEC tokens."""
    if not cloning_json:
        return []
    try:
        cloning = json.loads(cloning_json)
        vector_types = cloning.get("vector_types", [])
        if not vector_types:
            return []
        vt_str = " ".join(vector_types)
        tokens = set()
        for keyword, token in VEC_KEYWORD_MAP.items():
            if keyword.lower() in vt_str.lower():
                tokens.add(token)
        return sorted(tokens)
    except:
        return []
    
vec_check = (
    metadata
    .withColumn("vec_tokens", assign_vec_tokens(col("cloning")))
    .withColumn("vec_token", explode_outer(col("vec_tokens")))
    .filter(col("vec_token").isNotNull())
    .groupBy("vec_token")
    .count()
    .orderBy(col("count").desc())
)
display(vec_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. Soft Token — Backbone (BB)
# MAGIC
# MAGIC Take the top N backbones by frequency. Everything else gets no backbone token.
# MAGIC

# COMMAND ----------

# Get top N backbones
top_backbones_df = (
    metadata
    .withColumn("backbone", trim(get_json_object(col("cloning"), "$.backbone")))
    .filter(col("backbone").isNotNull() & (col("backbone") != ""))
    .groupBy("backbone")
    .count()
    .orderBy(col("count").desc())
    .limit(TOP_N_BACKBONES)
)

# Build the backbone -> token map
top_backbones = [row["backbone"] for row in top_backbones_df.collect()]

# Create token strings: sanitize backbone names for token format
def sanitize_token_name(name):
    """Convert a backbone name to a valid token string component."""
    return name.upper().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("/", "_").replace(".", "")


BB_TOKEN_MAP = {bb: f"<BB_{sanitize_token_name(bb)}>" for bb in top_backbones}

@udf(StringType())
def assign_bb_token(cloning_json):
    if not cloning_json:
        return None
    try:
        cloning = json.loads(cloning_json)
        backbone = cloning.get("backbone", "").strip()
        return BB_TOKEN_MAP.get(backbone)
    except:
        return None

# Verify
bb_check = (
    metadata
    .withColumn("bb_token", assign_bb_token(col("cloning")))
    .filter(col("bb_token").isNotNull())
    .groupBy("bb_token")
    .count()
    .orderBy(col("count").desc())
)
print(f"Plasmids with a backbone token: {bb_check.agg(F.sum('count')).collect()[0][0]:,}")
display(bb_check)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. Soft Token — Copy Number (COPY)
# MAGIC
# MAGIC Only two meaningful values: High Copy and Low Copy. Null/empty → no token.
# MAGIC

# COMMAND ----------

@udf(StringType())
def assign_copy_token(plasmid_copy):
    if not plasmid_copy or plasmid_copy.strip() == "":
        return None
    pc = plasmid_copy.strip().lower()
    if "high" in pc:
        return "<COPY_HIGH>"
    elif "low" in pc:
        return "<COPY_LOW>"
    return None

# Verify
copy_check = (
    metadata
    .withColumn("copy_token", assign_copy_token(col("plasmid_copy")))
    .groupBy("copy_token")
    .count()
    .orderBy(col("count").desc())
)
display(copy_check)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2e. Soft Token — Species (SP)
# MAGIC
# MAGIC Canonicalize insert species into ~6 high-frequency buckets. Rare species → no token.
# MAGIC

# COMMAND ----------

SPECIES_MAP = {
    "Homo sapiens": "<SP_HUMAN>",
    "Mus musculus": "<SP_MOUSE>",
    "Rattus norvegicus": "<SP_RAT>",
    "Escherichia coli": "<SP_ECOLI>",
    "Saccharomyces cerevisiae": "<SP_YEAST>",
    "Drosophila melanogaster": "<SP_DROSOPHILA>",
    "Danio rerio": "<SP_ZEBRAFISH>",
    "Caenorhabditis elegans": "<SP_CELEGANS>",
    "synthetic construct": "<SP_SYNTHETIC>",
    "Synthetic": "<SP_SYNTHETIC>",
}

@udf(ArrayType(StringType()))
def assign_species_tokens(inserts_json):
    if not inserts_json:
        return []
    try:
        inserts = json.loads(inserts_json)
        tokens = set()
        for insert in inserts:
            for sp in insert.get("species", []):
                if isinstance(sp, list) and len(sp) >= 2:
                    token = SPECIES_MAP.get(sp[1])
                    if token:
                        tokens.add(token)
        return sorted(tokens)
    except:
        return []
        return []

# Verify
sp_check = (
    metadata
    .withColumn("sp_tokens", assign_species_tokens(col("inserts")))
    .withColumn("sp_token", explode_outer(col("sp_tokens")))
    .filter(col("sp_token").isNotNull())
    .groupBy("sp_token")
    .count()
    .orderBy(col("count").desc())
)
display(sp_check)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2f. Unified Token Registry
# MAGIC
# MAGIC Combine all hard and soft tokens into a single reference table. This is the complete vocabulary
# MAGIC that the tokenizer needs to know about, plus structural tokens.
# MAGIC

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, BooleanType

# Hard tokens from motif registry
hard_tokens = (
    hard_token_vocab
    .select(
        col("token").alias("token_str"),
        col("category"),
        lit(True).alias("is_hard"),
        col("uuid").alias("motif_uuid"),
    )
    .distinct()
)

# Soft tokens — build manually from our maps
soft_token_rows = []

# VEC tokens
for token in sorted(set(VEC_KEYWORD_MAP.values())):
    soft_token_rows.append((token, "VEC", False, None))

# BB tokens
for token in sorted(BB_TOKEN_MAP.values()):
    soft_token_rows.append((token, "BB", False, None))

# COPY tokens
for token in ["<COPY_HIGH>", "<COPY_LOW>"]:
    soft_token_rows.append((token, "COPY", False, None))

# SPECIES tokens
for token in sorted(set(SPECIES_MAP.values())):
    soft_token_rows.append((token, "SP", False, None))

# Structural tokens
for token in ["<BOS>", "<EOS>", "<SEQ>", "<PAD>", "<UNK>"]:
    soft_token_rows.append((token, "STRUCTURAL", False, None))

soft_tokens = spark.createDataFrame(
    soft_token_rows,
    schema=StructType([
        StructField("token_str", StringType()),
        StructField("category", StringType()),
        StructField("is_hard", BooleanType()),
        StructField("motif_uuid", StringType()),
    ])
)

# Union
token_registry = hard_tokens.unionByName(soft_tokens)
print(f"Total token vocabulary: {token_registry.count()}")
display(token_registry.orderBy("category", "token_str"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Assign Tokens to Plasmids
# MAGIC
# MAGIC For each plasmid, determine which tokens belong in its prompt by:
# MAGIC 1. Matching annotations against the motif registry (hard tokens)
# MAGIC 2. Parsing metadata fields (soft tokens)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Hard Token Assignment via Annotations
# MAGIC
# MAGIC Join `annotations` to `motif_registry` on `sseqid`, applying quality thresholds.
# MAGIC Each match gives us a (plasmid_id, token) pair plus annotation metadata we'll preserve
# MAGIC for the RL reward lookup.
# MAGIC

# COMMAND ----------

# Join annotations to motif sseqid lookup, with quality filters
hard_token_assignments = (
    annotations
    .filter(
        (col("pident") >= PIDENT_THRESHOLD) &
        (col("percmatch") >= PERCMATCH_THRESHOLD) &
        (~col("fragment"))  # exclude fragment matches
    )
    .join(
        sseqid_to_token.select(
            col("sseqid").alias("motif_sseqid"),
            col("token").alias("token_str"),
            col("category").alias("token_category"),
            col("uuid").alias("motif_uuid"),
            col("sequence").alias("motif_canonical_seq"),
        ),
        annotations.sseqid == col("motif_sseqid"),
        "inner"
    )
    .select(
        col("plasmid_id"),
        col("token_str"),
        col("token_category"),
        col("motif_uuid"),
        col("motif_canonical_seq"),
        # Preserve annotation metadata for RL reward lookup
        col("pident"),
        col("percmatch"),
        col("qstart"),
        col("qend"),
        col("Feature"),
    )
)

# Deduplicate: if the same token appears multiple times for a plasmid 
# (e.g., two copies of the same promoter), keep the best match
w = Window.partitionBy("plasmid_id", "token_str").orderBy(col("pident").desc(), col("percmatch").desc())
hard_token_assignments = (
    hard_token_assignments
    .withColumn("rn", row_number().over(w))
    .filter(col("rn") == 1)
    .drop("rn")
)

print(f"Hard token assignments: {hard_token_assignments.count():,}")
print(f"Unique plasmids with hard tokens: {hard_token_assignments.select('plasmid_id').distinct().count():,}")


# COMMAND ----------

# Sanity check: distribution of hard tokens per plasmid
hard_per_plasmid = (
    hard_token_assignments
    .groupBy("plasmid_id")
    .agg(count("*").alias("n_hard_tokens"))
)
display(
    hard_per_plasmid
    .groupBy("n_hard_tokens")
    .count()
    .orderBy("n_hard_tokens")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Soft Token Assignment via Metadata
# MAGIC
# MAGIC Parse metadata JSON fields and assign VEC, BB, COPY, and SPECIES tokens.
# MAGIC

# COMMAND ----------

# Apply all soft token UDFs to metadata
soft_token_df = (
    metadata
    .select("id", "cloning", "plasmid_copy", "inserts")
    .withColumn("vec_tokens", assign_vec_tokens(col("cloning")))
    .withColumn("bb_token", assign_bb_token(col("cloning")))
    .withColumn("copy_token", assign_copy_token(col("plasmid_copy")))
    .withColumn("sp_tokens", assign_species_tokens(col("inserts")))
)

# Flatten into (plasmid_id, token_str) rows — one per token
# We need to handle the array columns (vec, species) and scalar columns (bb, copy) separately

# VEC tokens (array)
vec_rows = (
    soft_token_df.select(col("id").alias("plasmid_id"), explode("vec_tokens").alias("token_str"))
    .withColumn("token_category", lit("VEC"))
)

# BB token (scalar)
bb_rows = (
    soft_token_df.filter(col("bb_token").isNotNull())
    .select(col("id").alias("plasmid_id"), col("bb_token").alias("token_str"))
    .withColumn("token_category", lit("BB"))
)

# COPY token (scalar)
copy_rows = (
    soft_token_df.filter(col("copy_token").isNotNull())
    .select(col("id").alias("plasmid_id"), col("copy_token").alias("token_str"))
    .withColumn("token_category", lit("COPY"))
)

# SPECIES tokens (array)
sp_rows = (
    soft_token_df.select(col("id").alias("plasmid_id"), explode("sp_tokens").alias("token_str"))
    .withColumn("token_category", lit("SP"))
)

# Union all soft token assignments
soft_token_assignments = (
    vec_rows
    .unionByName(bb_rows)
    .unionByName(copy_rows)
    .unionByName(sp_rows)
)

print(f"Soft token assignments: {soft_token_assignments.count():,}")
print(f"Unique plasmids with soft tokens: {soft_token_assignments.select('plasmid_id').distinct().count():,}")


# COMMAND ----------

# Distribution of soft tokens per plasmid
soft_per_plasmid = (
    soft_token_assignments
    .groupBy("plasmid_id")
    .agg(count("*").alias("n_soft_tokens"))
)
display(
    soft_per_plasmid
    .groupBy("n_soft_tokens")
    .count()
    .orderBy("n_soft_tokens")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Merge All Tokens per Plasmid
# MAGIC
# MAGIC Combine hard and soft tokens, then aggregate into a sorted prompt-token list per plasmid.
# MAGIC
# MAGIC **Canonical token ordering** (model sees consistent order):
# MAGIC 1. VEC (vector type)
# MAGIC 2. SP (species)
# MAGIC 3. COPY (copy number)
# MAGIC 4. BB (backbone)
# MAGIC 5. AMR (resistance)
# MAGIC 6. ORI (origin of replication)
# MAGIC 7. PROM (promoter)
# MAGIC 8. ELEM (regulatory elements)
# MAGIC 9. REPORTER
# MAGIC 10. TAG
# MAGIC

# COMMAND ----------

# Define canonical category ordering
CATEGORY_ORDER = {
    "VEC": 1,
    "SP": 2,
    "COPY": 3,
    "BB": 4,
    "AMR": 5,
    "ORI": 6,
    "PROM": 7,
    "ELEM": 8,
    "REPORTER": 9,
    "TAG": 10,
}


# Union hard + soft token assignments into a common schema
all_token_assignments = (
    hard_token_assignments
    .select("plasmid_id", "token_str", "token_category")
    .unionByName(
        soft_token_assignments
        .select("plasmid_id", "token_str", "token_category")
    )
    .distinct()  # deduplicate exact (plasmid, token) pairs
)

print(f"Total token assignments: {all_token_assignments.count():,}")
print(f"Unique plasmids: {all_token_assignments.select('plasmid_id').distinct().count():,}")


# COMMAND ----------

# Aggregate tokens per plasmid with canonical ordering
# We use a UDF to sort tokens by category order, then alphabetically within category

@udf(ArrayType(StringType()))
def sort_tokens_canonical(tokens, categories):
    """Sort tokens by canonical category order, then alphabetically within category."""
    if not tokens or not categories:
        return []
    order_map = CATEGORY_ORDER
    paired = list(zip(tokens, categories))
    paired.sort(key=lambda x: (order_map.get(x[1], 99), x[0]))
    return [t for t, c in paired]

tokens_per_plasmid = (
    all_token_assignments
    .groupBy("plasmid_id")
    .agg(
        collect_list("token_str").alias("token_list"),
        collect_list("token_category").alias("category_list"),
        countDistinct("token_str").alias("n_tokens"),
    )
    .withColumn("sorted_tokens", sort_tokens_canonical(col("token_list"), col("category_list")))
    .drop("token_list", "category_list")
)

print(f"Plasmids with tokens: {tokens_per_plasmid.count():,}")
display(tokens_per_plasmid.orderBy(col("n_tokens").desc()).limit(20))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build Training Pairs
# MAGIC
# MAGIC Join token prompts with full sequences. Construct the final prompt string format:
# MAGIC ```
# MAGIC <BOS><TOKEN1><TOKEN2>...<SEQ>SEQUENCE<EOS>
# MAGIC ```
# MAGIC
# MAGIC Also preserve the hard-token reward metadata for post-training.
# MAGIC

# COMMAND ----------

# Build reward metadata: per-plasmid list of hard tokens with their canonical motif sequences
# This is used in post-training to verify generated sequences contain expected motifs
reward_metadata = (
    hard_token_assignments
    .groupBy("plasmid_id")
    .agg(
        collect_list(
            F.struct(
                col("token_str"),
                col("motif_uuid"),
                col("motif_canonical_seq"),
                col("qstart"),
                col("qend"),
                col("pident"),
                col("percmatch"),
            )
        ).alias("reward_motifs")
    )
)


# COMMAND ----------

# Join: tokens + sequences + reward metadata
training_pairs = (
    tokens_per_plasmid
    .join(
        full_sequences.select(
            col("plasmid_id"),
            col("sequence"),
            col("sequence_length"),
        ),
        on="plasmid_id",
        how="inner"
    )
    .join(reward_metadata, on="plasmid_id", how="left")
)

# Construct the prompt string
@udf(StringType())
def build_prompt(sorted_tokens):
    """Build the prompt prefix from sorted tokens."""
    if not sorted_tokens:
        return "<BOS><SEQ>"
    return "<BOS>" + "".join(sorted_tokens) + "<SEQ>"

training_pairs = (
    training_pairs
    .withColumn("prompt", build_prompt(col("sorted_tokens")))
    .withColumn("full_text", concat(col("prompt"), col("sequence"), lit("<EOS>")))
    .withColumn("prompt_length", F.length(col("prompt")))
    .withColumn("total_length", F.length(col("full_text")))
)

print(f"Training pairs before filtering: {training_pairs.count():,}")
display(training_pairs.select("plasmid_id", "prompt", "n_tokens", "sequence_length", "total_length").limit(10))


# COMMAND ----------

# Inspect a few examples end-to-end
display(
    training_pairs
    .select("prompt", "sequence_length", "n_tokens")
    .orderBy(col("n_tokens").desc())
    .limit(5)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Filter, Deduplicate, Split
# MAGIC
# MAGIC ### Filtering strategy:
# MAGIC - Tag plasmids with **0 hard tokens** (prompt is purely soft tokens — not useful for RL but can be used for pretraining)
# MAGIC - Apply sequence length cutoffs for different dataset variants
# MAGIC - Subsample near-duplicate prompts (same prompt → different sequences)
# MAGIC
# MAGIC ### Split strategy:
# MAGIC Tag-based filtering to create bacterial, mammalian, and combined datasets.
# MAGIC

# COMMAND ----------

# Tag hard-token presence and add prompt hash for analytics
training_tagged = (
    training_pairs
    .withColumn("has_hard_tokens",
        col("reward_motifs").isNotNull() & (size(col("reward_motifs")) >= 1)
    )
    .withColumn("prompt_hash", spark_hash(col("prompt")))
)

total = training_tagged.count()
n_hard = training_tagged.filter(col("has_hard_tokens")).count()
print(f"Total training pairs: {total:,}")
print(f"  With hard tokens (usable for RL): {n_hard:,} ({100*n_hard/total:.1f}%)")
print(f"  Soft-only (pretraining only):     {total - n_hard:,} ({100*(total-n_hard)/total:.1f}%)")

# COMMAND ----------

# Optional: cap examples per prompt to prevent extreme imbalance
MAX_PER_PROMPT = 50

# Check if capping is even needed
prompt_freq = (
    training_tagged
    .groupBy("prompt_hash")
    .agg(count("*").alias("n"))
    .orderBy(col("n").desc())
)
display(prompt_freq.limit(20))
print(f"Unique prompts: {prompt_freq.count():,}")
print(f"Max examples per prompt: {prompt_freq.agg(spark_max('n')).collect()[0][0]}")

# COMMAND ----------

# Apply cap
w = Window.partitionBy("prompt_hash").orderBy(col("n_tokens").desc(), col("sequence_length"))
training_capped = (
    training_tagged
    .withColumn("rn", row_number().over(w))
    .filter(col("rn") <= MAX_PER_PROMPT)
    .drop("rn")
)

print(f"Before cap: {training_tagged.count():,}")
print(f"After cap (max {MAX_PER_PROMPT} per prompt): {training_capped.count():,}")

# COMMAND ----------

# Length buckets + split tags
training_final = (
    training_capped
    .withColumn("len_bucket",
        when(col("sequence_length") <= 4096, "leq_4k")
        .when(col("sequence_length") <= 8192, "leq_8k")
        .when(col("sequence_length") <= 16384, "leq_16k")
        .when(col("sequence_length") <= 32768, "leq_32k")
        .otherwise("gt_32k")
    )
    .withColumn("splits", assign_splits(col("sorted_tokens")))
)

# Stats per length bucket
display(
    training_final
    .groupBy("len_bucket")
    .agg(
        count("*").alias("n_pairs"),
        countDistinct("prompt_hash").alias("n_unique_prompts"),
        F.sum(col("has_hard_tokens").cast("int")).alias("n_with_hard_tokens"),
        mean("n_tokens").alias("avg_tokens"),
        mean("sequence_length").alias("avg_seq_len"),
    )
    .orderBy("len_bucket")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset splits
# MAGIC
# MAGIC Create tagged views for bacterial, mammalian, and combined datasets.
# MAGIC A plasmid can appear in multiple splits (e.g., a shuttle vector).
# MAGIC

# COMMAND ----------

# Tag each training pair with its applicable splits
# We check if specific VEC tokens are present in the sorted_tokens array

@udf(ArrayType(StringType()))
def assign_splits(sorted_tokens):
    """Determine which dataset splits this plasmid belongs to."""
    if not sorted_tokens:
        return ["combined"]
    
    token_set = set(sorted_tokens)
    splits = ["combined"]  # everything goes in combined
    
    if "<VEC_BACTERIAL>" in token_set:
        splits.append("bacterial")
    if any(t in token_set for t in ["<VEC_MAMMALIAN>", "<VEC_LENTIVIRAL>", "<VEC_RETROVIRAL>", "<VEC_AAV>"]):
        splits.append("mammalian")
    if "<VEC_CRISPR>" in token_set:
        splits.append("crispr")
    if "<VEC_YEAST>" in token_set:
        splits.append("yeast")
    if "<VEC_INSECT>" in token_set:
        splits.append("insect")
    if "<VEC_PLANT>" in token_set:
        splits.append("plant")
    
    return splits

training_final = training_final.withColumn("splits", assign_splits(col("sorted_tokens")))

# Split counts
split_counts = (
    training_final
    .withColumn("split", explode("splits"))
    .groupBy("split")
    .agg(
        count("*").alias("n_pairs"),
        countDistinct("prompt_hash").alias("n_unique_prompts"),
    )
    .orderBy(col("n_pairs").desc())
)
display(split_counts)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Datasets
# MAGIC
# MAGIC Write the master table and filtered views to Delta. Each downstream training run can filter by 
# MAGIC `len_bucket` and `splits` as needed.
# MAGIC

# COMMAND ----------

# Select final columns for the training dataset
output_columns = [
    "plasmid_id",
    "prompt",
    "prompt_hash",
    "sorted_tokens",
    "n_tokens",
    "sequence",
    "sequence_length",
    "full_text",
    "total_length",
    "len_bucket",
    "splits",
    "reward_motifs",
]

# Write master training table
(
    training_final
    .select(output_columns)
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("addgene.default.training_pairs_v4")
)

print("Saved: addgene.default.training_pairs")


# COMMAND ----------

# Convenience: write per-split, per-length tables for the most useful combos
# These are just filtered views of the master table

for max_len, bucket_filter in [("8k", "leq_4k"), ("8k", "leq_8k"), ("16k", "leq_16k")]:
    for split_name in ["combined", "bacterial", "mammalian"]:
        table_name = f"addgene.default.training_{split_name}_{bucket_filter}"
        
        df = (
            training_final
            .select(output_columns)
            .filter(F.array_contains(col("splits"), split_name))
            .filter(col("len_bucket").isin(
                # Include all buckets up to the target length
                [b for b in ["leq_4k", "leq_8k", "leq_16k", "leq_32k"] 
                 if b <= bucket_filter]
            ))
        )
        
        cnt = df.count()
        if cnt > 0:
            df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
            print(f"Saved {table_name}: {cnt:,} pairs")
        else:
            print(f"Skipped {table_name}: 0 pairs")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validation & Statistics
# MAGIC
# MAGIC Verify the datasets look right before training.
# MAGIC

# COMMAND ----------

# Reload the master table
tp = spark.read.table("addgene.default.training_pairs")
print(f"Total training pairs: {tp.count():,}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a. Token coverage — what % of plasmids got each token type?

# COMMAND ----------

# Token type coverage
@udf(ArrayType(StringType()))
def extract_token_categories(sorted_tokens):
    """Extract the category prefix from each token."""
    if not sorted_tokens:
        return []
    categories = set()
    for t in sorted_tokens:
        # Token format: <CATEGORY_NAME> — extract up to first underscore after <
        inner = t.strip("<>")
        parts = inner.split("_", 1)
        if parts:
            categories.add(parts[0])
    return sorted(categories)

coverage = (
    tp
    .withColumn("token_cats", extract_token_categories(col("sorted_tokens")))
    .withColumn("cat", explode("token_cats"))
    .groupBy("cat")
    .agg(
        count("*").alias("n_plasmids"),
        (count("*") / tp.count() * 100).alias("pct_coverage"),
    )
    .orderBy(col("n_plasmids").desc())
)
display(coverage)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Most common prompts

# COMMAND ----------

prompt_freq = (
    tp
    .groupBy("prompt")
    .agg(count("*").alias("n_sequences"))
    .orderBy(col("n_sequences").desc())
)
display(prompt_freq.limit(30))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7c. Length distribution per split

# COMMAND ----------

for split_name in ["combined", "bacterial", "mammalian"]:
    split_df = tp.filter(F.array_contains(col("splits"), split_name))
    stats = split_df.select(
        lit(split_name).alias("split"),
        count("*").alias("n"),
        mean("sequence_length").alias("mean_len"),
        percentile_approx("sequence_length", 0.5).alias("median_len"),
        percentile_approx("sequence_length", 0.95).alias("p95_len"),
        mean("n_tokens").alias("mean_tokens"),
    )
    display(stats)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7d. Example training pairs — spot check

# COMMAND ----------

# Show a few complete examples
display(
    tp
    .filter(col("n_tokens") >= 3)  # pick examples with a decent number of tokens
    .select("prompt", "sequence_length", "n_tokens", "sorted_tokens", "splits")
    .limit(10)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Token List for Tokenizer
# MAGIC
# MAGIC Export the complete special token vocabulary. This is what needs to be added to the 
# MAGIC base DNA tokenizer (on top of A, T, G, C, N tokens).
# MAGIC
# MAGIC Format: one token per line, ready to feed into tokenizer `add_special_tokens()`.
# MAGIC

# COMMAND ----------

# Collect all unique tokens actually used in training data (not just the vocabulary)
used_tokens = (
    tp
    .withColumn("token", explode("sorted_tokens"))
    .select("token")
    .distinct()
    .orderBy("token")
)

# Also include structural tokens that may not appear in sorted_tokens
structural_tokens = ["<BOS>", "<EOS>", "<SEQ>", "<PAD>", "<UNK>"]

all_tokens = (
    used_tokens
    .unionByName(
        spark.createDataFrame([(t,) for t in structural_tokens], ["token"])
    )
    .distinct()
    .orderBy("token")
)

print(f"Total special tokens for tokenizer: {all_tokens.count()}")
all_tokens.show(200, truncate=False)


# COMMAND ----------

# Save token list as a simple text file and as a Delta table
token_list = [row["token"] for row in all_tokens.collect()]

# Save to DBFS as text
token_text = "\n".join(token_list)
dbutils.fs.put("/FileStore/plasmidgpt/special_tokens.txt", token_text, overwrite=True)
print(f"Saved {len(token_list)} tokens to /FileStore/plasmidgpt/special_tokens.txt")

# Also save as Delta for programmatic access
all_tokens.write.mode("overwrite").saveAsTable("addgene.default.special_tokens")
print("Saved: addgene.default.special_tokens")


# COMMAND ----------

# Print the full token list for reference
print("=== Special Token Vocabulary ===")
print(f"Total: {len(token_list)} tokens\n")
for i, t in enumerate(token_list, 1):
    print(f"  {i:3d}. {t}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Tables created:**
# MAGIC - `addgene.default.training_pairs` — Master table with all training pairs, tokens, reward metadata
# MAGIC - `addgene.default.training_combined_leq_8k` — Combined dataset, sequences ≤ 8kb
# MAGIC - `addgene.default.training_bacterial_leq_8k` — Bacterial only, ≤ 8kb
# MAGIC - `addgene.default.training_mammalian_leq_8k` — Mammalian only, ≤ 8kb
# MAGIC - `addgene.default.special_tokens` — Token vocabulary for tokenizer
# MAGIC
# MAGIC **Files created:**
# MAGIC - `/FileStore/plasmidgpt/special_tokens.txt` — Token list (one per line)
# MAGIC
# MAGIC **For post-training (RL):**
# MAGIC - Use the `reward_motifs` column which contains, per training pair, the list of hard tokens 
# MAGIC   with their canonical motif sequences. At inference time, BLAST/align the generated sequence 
# MAGIC   against these canonical sequences to compute the reward.
# MAGIC

# COMMAND ----------

