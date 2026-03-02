#!/bin/bash
# One-time environment setup for Myriad (run on login node).
#
# Usage:
#   ssh ucbt042@myriad.rc.ucl.ac.uk
#   bash ~/PlasmidLLM/scripts/myriad/setup_env.sh
#
# After running, verify with:
#   source ~/plasmid_venv/bin/activate
#   python -c "import torch; print(torch.cuda.is_available())"
#
# NOTE: Myriad's system cc is GCC 4.8.5 which is too old for numpy/pandas
# source builds. We set CC/CXX to GCC 10.2.0 and pin packages to versions
# with pre-built wheels where possible.
set -euo pipefail

# Init module system (needed for non-login shells)
source /shared/ucl/apps/modules/current/init/bash

echo "=== Loading modules ==="
module purge
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0
module load python/3.11.4-gnu-10.2.0

# Use GCC 10.2.0 as C/C++ compiler (system cc is 4.8.5)
export CC=/shared/ucl/apps/gcc/10.2.0/bin/gcc
export CXX=/shared/ucl/apps/gcc/10.2.0/bin/g++

echo "=== Creating venv at ~/plasmid_venv ==="
python3 -m venv ~/plasmid_venv
source ~/plasmid_venv/bin/activate

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing pre-built wheel dependencies ==="
pip install "numpy<2.3"
pip install pandas==2.2.3
pip install "pyarrow>=14.0,<17"
pip install transformers

echo "=== Installing MLflow (skinny client) and other deps ==="
pip install mlflow-skinny mlflow-tracing boto3 psutil

echo "=== Installing RL dependencies ==="
pip install "ray[default]>=2.9" "parasail>=1.3.0" "biopython>=1.80"

echo "=== Installing PlasmidLLM (editable, no deps) ==="
cd ~/PlasmidLLM
pip install --no-deps -e .

echo "=== Verifying install ==="
python -c "
import torch; print(f'torch {torch.__version__}')
import ray; print(f'ray {ray.__version__}')
import mlflow; print(f'mlflow {mlflow.__version__}')
import parasail; print('parasail OK')
import Bio; print('biopython OK')
import plasmid_llm; print('plasmid_llm OK')
print('=== ALL OK ===')
"

echo "=== Done ==="
