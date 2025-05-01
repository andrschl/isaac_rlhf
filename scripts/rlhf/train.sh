#!/usr/bin/env bash
set -euo pipefail

# 1) Load conda and activate your env
#    Adjust the path to `conda.sh` if your install is elsewhere
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

# 2) Sweep specs
SEEDS=$(seq 2 10)
ALGOS=(vanilla ts_last)

# 3) Launch train_rlhf.py for each combination
for seed in ${SEEDS}; do
  for algo in "${ALGOS[@]}"; do
    echo "=== Running seed=${seed}, algo=${algo} ==="
    python scripts/rlhf/train_rlhf.py \
      --base_seed ${seed} \
      --rlhf_algorithm ${algo}

    # force‑kill any stray Python processes and give the system a breather
    echo ">>> Killing leftover python processes..."
    pkill -9 python || true

    # sleep a bit before next run
    echo ">>> Sleeping for 10s before next run..."
    sleep 10
  done
done

echo "All runs completed."