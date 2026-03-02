#!/bin/bash -l
# SGE job script for GRPO post-training on Myriad A100.
#
# Submit with:  qsub scripts/myriad/submit_grpo.sh
# Monitor with: qstat
# Logs at:      ~/PlasmidLLM/output/grpo_myriad_v1/grpo.$JOB_ID.{o,e}

#$ -N grpo
#$ -o /home/ucbt042/PlasmidLLM/output/grpo_myriad_v1/grpo.$JOB_ID.o
#$ -e /home/ucbt042/PlasmidLLM/output/grpo_myriad_v1/grpo.$JOB_ID.e

# Resources: 1 GPU (A100), 32 CPUs, 5G RAM per slot, 24h walltime
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 32
#$ -l mem=5G
#$ -l h_rt=24:00:00

# Run from repo root
#$ -wd /home/ucbt042/PlasmidLLM

set -euo pipefail

# Init module system (needed for non-login shells)
source /shared/ucl/apps/modules/current/init/bash

echo "=== Job $JOB_ID started at $(date) on $(hostname) ==="
echo "TMPDIR=$TMPDIR"
nvidia-smi

# --- Modules ---
module purge
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0
module load python/3.11.4-gnu-10.2.0

# --- Activate venv ---
source ~/plasmid_venv/bin/activate

# --- Load Databricks/MLflow credentials ---
if [[ -f ~/.databricks_env ]]; then
    source ~/.databricks_env
    echo "Loaded Databricks credentials"
else
    echo "WARNING: ~/.databricks_env not found — MLflow tracking will be disabled"
fi

# --- Stage data to $TMPDIR for fast local I/O ---
DATA_SRC=~/PlasmidLLM/myriad_data
DATA_DST=$TMPDIR/plasmid_data

echo "=== Staging data from $DATA_SRC to $DATA_DST ==="
mkdir -p "$DATA_DST"
cp -r "$DATA_SRC/checkpoint-15000" "$DATA_DST/"
cp "$DATA_SRC/training_pairs_v4.parquet" "$DATA_DST/"
cp "$DATA_SRC/motif_registry.parquet" "$DATA_DST/"
echo "Staged $(du -sh "$DATA_DST" | cut -f1) to $DATA_DST"

# --- Environment variables ---
export PLASMID_DATA_DIR="$DATA_DST"
export RAY_TMPDIR="$TMPDIR/ray"
mkdir -p "$RAY_TMPDIR"

# Ensure output dir exists
mkdir -p ~/PlasmidLLM/output/grpo_myriad_v1

# --- Run training ---
echo "=== Starting GRPO training at $(date) ==="
python scripts/train_ray.py configs/myriad_grpo.py

echo "=== Job $JOB_ID finished at $(date) ==="
