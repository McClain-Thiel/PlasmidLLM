#!/bin/bash
# Run RL curriculum experiments sequentially on g6-big.
# Each experiment is ~400 steps at ~16s/step = ~107 minutes.
# Total: ~5.3 hours for all 3.
#
# Usage: bash scripts/run_experiments.sh [a|b|c|all]

set -euo pipefail
cd /home/ubuntu/PlasmidLLM
PYTHON=/opt/dlami/nvme/plasmid_venv/bin/python
LOGDIR=/opt/dlami/nvme/output

run_exp() {
    local name=$1
    local config=$2
    local logfile="${LOGDIR}/${name}.log"

    echo "=== Starting experiment: ${name} ==="
    echo "  Config: ${config}"
    echo "  Log: ${logfile}"
    echo "  Time: $(date -u)"

    $PYTHON scripts/train_ppo.py "${config}" 2>&1 | tee "${logfile}"

    echo "=== Finished experiment: ${name} at $(date -u) ==="
    echo ""
}

TARGET="${1:-all}"

case "$TARGET" in
    a)
        run_exp "exp_a_soft_ceiling" "configs/exp_a_soft_ceiling.py"
        ;;
    b)
        run_exp "exp_b_presence_only" "configs/exp_b_presence_only.py"
        ;;
    c)
        run_exp "exp_c_two_phase" "configs/exp_c_two_phase.py"
        ;;
    all)
        # Run B first (fastest to show signal), then A, then C
        run_exp "exp_b_presence_only" "configs/exp_b_presence_only.py"
        run_exp "exp_a_soft_ceiling" "configs/exp_a_soft_ceiling.py"
        run_exp "exp_c_two_phase" "configs/exp_c_two_phase.py"
        ;;
    *)
        echo "Usage: $0 [a|b|c|all]"
        exit 1
        ;;
esac

echo "All requested experiments complete!"
