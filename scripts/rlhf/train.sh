#!/usr/bin/env bash
set -euo pipefail

# 1) Load conda and activate your env
#    Adjust the path to `conda.sh` if your install is elsewhere
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

# 2) Sweep specs
SEEDS=$(seq 1 10)
ALGOS=(ts_last)
BETAS=(1e-3 1e-2 1e-1 1e0)

# 3) Launch train_rlhf.py for each combination
for seed in ${SEEDS}; do
  for algo in "${ALGOS[@]}"; do
    for beta in "${BETAS[@]}"; do
      # Run the training script with the current seed and algorithm
      echo "=== Running seed=${seed}, algo=${algo}, beta=${beta} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --rlhf_algorithm ${algo} \
        --beta ${beta} \
        --lazy

      # force‑kill any stray Python processes and give the system a breather
      echo ">>> Killing leftover python processes..."
      pkill -9 python || true

      # sleep a bit before next run
      echo ">>> Sleeping for 10s before next run..."
      sleep 10
    done
  done
done

for seed in ${SEEDS}; do
  for algo in "${ALGOS[@]}"; do
    for beta in "${BETAS[@]}"; do
      # Run the training script with the current seed and algorithm
      echo "=== Running seed=${seed}, algo=${algo}, beta=${beta} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --rlhf_algorithm ${algo} \
        --beta ${beta}

      # force‑kill any stray Python processes and give the system a breather
      echo ">>> Killing leftover python processes..."
      pkill -9 python || true

      # sleep a bit before next run
      echo ">>> Sleeping for 10s before next run..."
      sleep 10
    done
  done
done

echo "All runs completed."