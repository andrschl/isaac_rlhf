#!/usr/bin/env bash
set -euo pipefail

# 1) Load conda and activate your env
#    Adjust the path to `conda.sh` if your install is elsewhere
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

# TS loop
SEEDS=$(seq 1 3)
BETA1S=(1.0)
BETA2S=(1.0)
for seed in ${SEEDS}; do
  for beta1 in "${BETA1S[@]}"; do
    for beta2 in "${BETA2S[@]}"; do

      echo "=== Running ts seed=${seed}, beta1=${beta1}, beta2=${beta2} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --rlhf_algorithm ts_last \
        --resume

      # force-kill any stray Python processes
      # echo ">>> Killing leftover Python processes..."
      # pkill -9 python || true

      # sleep a bit before next run
      echo ">>> Sleeping for 10s before next run..."
      sleep 10

      echo "=== Running ts opt design seed=${seed}, beta1=${beta1}, beta2=${beta2} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --rlhf_algorithm ts_last \
        --lazy \
        --opt_design \
        --resume

      # force-kill any stray Python processes
      # echo ">>> Killing leftover Python processes..."
      # pkill -9 python || true

      # sleep a bit before next run
      echo ">>> Sleeping for 10s before next run..."
      sleep 10

      echo "=== Running ts lazy seed=${seed}, beta1=${beta1}, beta2=${beta2} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --rlhf_algorithm ts_last \
        --lazy \
        --resume

      # force-kill any stray Python processes
      # echo ">>> Killing leftover Python processes..."
      # pkill -9 python || true

      # sleep a bit before next run
      echo ">>> Sleeping for 10s before next run..."
      pkill -f train_rlhf.py || true
      sleep 10


    done
  done
done


for seed in ${SEEDS}; do
  # Run the training script with the current seed and algorithm
  echo "=== Running seed=${seed}, algo=vanilla ==="
  python scripts/rlhf/train_rlhf.py \
    --base_seed ${seed} \
    --rlhf_algorithm vanilla \
    --resume

  # force‑kill any stray Python processes and give the system a breather
  # echo ">>> Killing leftover python processes..."
  # pkill -9 python || true

  # sleep a bit before next run
  echo ">>> Sleeping for 10s before next run..."
  pkill -f train_rlhf.py || true
  sleep 10
done

for seed in ${SEEDS}; do
  # Run the training script with the current seed and algorithm
  echo "=== Running seed=${seed}, algo=rl==="
  python scripts/rlhf/train_rlhf.py \
    --base_seed ${seed} \
    --rlhf_algorithm rl

  # force‑kill any stray Python processes and give the system a breather
  # echo ">>> Killing leftover python processes..."
  # pkill -9 python || true

  # sleep a bit before next run
  echo ">>> Sleeping for 10s before next run..."
  pkill -f train_rlhf.py || true
  sleep 10
done


echo "All runs completed."