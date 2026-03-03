#!/bin/bash
# Setup script for p4 (8x A100-40GB) — installs deps, downloads data, configures W&B
set -euo pipefail

WORK_DIR="/opt/dlami/nvme/work/PlasmidLLM"
DATA_DIR="/opt/dlami/nvme/data"
VENV_DIR="/opt/dlami/nvme/venv"

echo "=== Setting up PlasmidLLM on p4 ==="

# 1. Create and activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(which python) ($(python --version))"

# 2. Install dependencies
echo "Installing PyTorch and dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate wandb pyarrow parasail-python
pip install -e "$WORK_DIR"

# 3. Download training data from S3
echo "Downloading training data from S3..."
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/training_pairs_v4.parquet" ]; then
    aws s3 cp s3://phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet "$DATA_DIR/"
    echo "Downloaded training_pairs_v4.parquet ($(du -h "$DATA_DIR/training_pairs_v4.parquet" | cut -f1))"
else
    echo "training_pairs_v4.parquet already exists"
fi

# Also download motif registry for eval
if [ ! -f "$DATA_DIR/motif_registry.parquet" ]; then
    aws s3 cp s3://phd-research-storage-1758274488/databricks_export/motif_registry.parquet "$DATA_DIR/"
fi

# 4. Configure W&B
echo "Configuring W&B..."
wandb login 4735d4f04c0bbfc6addc08ceec8e2a04a77b3d07

echo ""
echo "=== Setup complete ==="
echo "Activate venv: source $VENV_DIR/bin/activate"
echo "Run training:  python scripts/train_pretrain.py configs/p4_baseline.py"
