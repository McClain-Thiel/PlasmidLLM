#!/bin/bash
# Transfer model checkpoint and data files to Myriad.
#
# Run from your local machine (not g6-big or Myriad).
# Requires SSH access to both g6-big and Myriad.
#
# Usage:
#   bash scripts/myriad/transfer_data.sh
#
# Files transferred (~70MB total):
#   - checkpoint-15000/  (model weights, config, vocab, special tokens)
#   - training_pairs_v4.parquet
#   - motif_registry.parquet
set -euo pipefail

# --- Configuration ---
G6BIG_HOST="g6-big"   # adjust to your SSH config alias or user@host
MYRIAD_HOST="ucbt042@myriad.rc.ucl.ac.uk"

G6BIG_CHECKPOINT="/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"
G6BIG_TRAINING_PAIRS="/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"
G6BIG_MOTIF_REGISTRY="/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"

LOCAL_STAGING="/tmp/plasmid_myriad_staging"
MYRIAD_DEST="~/PlasmidLLM/myriad_data"

# --- Step 1: Pull from g6-big to local staging ---
echo "=== Step 1: Downloading from g6-big to local staging ==="
mkdir -p "$LOCAL_STAGING"

echo "  Checkpoint..."
rsync -avz --progress "$G6BIG_HOST:$G6BIG_CHECKPOINT" "$LOCAL_STAGING/"

echo "  Training pairs..."
rsync -avz --progress "$G6BIG_HOST:$G6BIG_TRAINING_PAIRS" "$LOCAL_STAGING/"

echo "  Motif registry..."
rsync -avz --progress "$G6BIG_HOST:$G6BIG_MOTIF_REGISTRY" "$LOCAL_STAGING/"

echo "Local staging contents:"
du -sh "$LOCAL_STAGING"/*

# --- Step 2: Push from local to Myriad ---
echo ""
echo "=== Step 2: Uploading from local to Myriad ==="

ssh "$MYRIAD_HOST" "mkdir -p $MYRIAD_DEST"

echo "  Checkpoint..."
rsync -avz --progress "$LOCAL_STAGING/checkpoint-15000" "$MYRIAD_HOST:$MYRIAD_DEST/"

echo "  Training pairs..."
rsync -avz --progress "$LOCAL_STAGING/training_pairs_v4.parquet" "$MYRIAD_HOST:$MYRIAD_DEST/"

echo "  Motif registry..."
rsync -avz --progress "$LOCAL_STAGING/motif_registry.parquet" "$MYRIAD_HOST:$MYRIAD_DEST/"

# --- Step 3: Verify ---
echo ""
echo "=== Step 3: Verifying files on Myriad ==="
ssh "$MYRIAD_HOST" "ls -lhR $MYRIAD_DEST"

echo ""
echo "=== Transfer complete ==="
echo "Clean up local staging with: rm -rf $LOCAL_STAGING"
