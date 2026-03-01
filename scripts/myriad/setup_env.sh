#!/bin/bash
# One-time environment setup for Myriad (run on login node).
#
# Usage:
#   ssh ucbt042@myriad.rc.ucl.ac.uk
#   bash scripts/myriad/setup_env.sh
#
# After running, verify with:
#   source ~/plasmid_venv/bin/activate
#   python -c "import torch; print(torch.cuda.is_available())"
#
# NOTE: Module names are placeholders — run `module avail` and adjust
# the gcc/cuda/python versions to match what's available on Myriad.
set -euo pipefail

echo "=== Loading modules ==="
module purge
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0
module load python/3.11.4-gnu-10.2.0

echo "=== Creating venv at ~/plasmid_venv ==="
python3 -m venv ~/plasmid_venv
source ~/plasmid_venv/bin/activate

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing transformers ==="
pip install transformers

echo "=== Installing PlasmidLLM with RL extras ==="
cd ~/PlasmidLLM
pip install -e ".[rl]"

echo "=== Verifying install ==="
python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import ray; print(f'ray {ray.__version__}')"
python -c "import parasail; print(f'parasail {parasail.__version__}')"

echo "=== Done ==="
