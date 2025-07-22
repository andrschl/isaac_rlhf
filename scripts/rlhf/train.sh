#!/usr/bin/env bash
set -euo pipefail

# 1) Load conda and activate your env
#    Adjust the path to `conda.sh` if your install is elsewhere
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

# # Run the training script with the current seed and algorithm
# echo "=== Running algo=rl ==="
# python scripts/rlhf/train_rlhf.py \
#   --rlhf_algorithm rl \

# # force‑kill any stray Python processes and give the system a breather
# # echo ">>> Killing leftover python processes..."
# # pkill -9 python || true

# # sleep a bit before next run
# echo ">>> Sleeping for 10s before next run..."
# sleep 20

# TS loop
SEEDS=$(seq 1 2)
BETA1S=(1e0 1e1)
BETA2S=(1e-1 1e0 1e1)
for seed in ${SEEDS}; do
  for beta1 in "${BETA1S[@]}"; do
    for beta2 in "${BETA2S[@]}"; do

      echo "=== Running ts seed=${seed}, beta1=${beta1}, beta2=${beta2} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --rlhf_algorithm ts_last

      echo ">>> Sleeping for 10s before next run..."
      sleep 20

      echo "=== Running ts seed=${seed}, beta1=${beta1}, beta2=${beta2} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --rlhf_algorithm ts_last \
        --lazy \

      echo ">>> Sleeping for 10s before next run..."
      sleep 20
      
      echo "=== Running ts seed=${seed}, beta1=${beta1}, beta2=${beta2} ==="
      python scripts/rlhf/train_rlhf.py \
        --base_seed ${seed} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --rlhf_algorithm ts_last \
        --lazy \
        --opt_design 

      echo ">>> Sleeping for 10s before next run..."
      sleep 20
    done
  done

  Run the training script with the current seed and algorithm
  echo "=== Running seed=${seed}, algo=vanilla ==="
  python scripts/rlhf/train_rlhf.py \
    --base_seed ${seed} \
    --rlhf_algorithm vanilla
  
  echo ">>> Sleeping for 10s before next run..."
  sleep 20
done


# for seed in ${SEEDS}; do
#   # Run the training script with the current seed and algorithm
#   echo "=== Running seed=${seed}, algo=vanilla ==="
#   python scripts/rlhf/train_rlhf.py \
#     --base_seed ${seed} \
#     --rlhf_algorithm vanilla \
#     --resume

#   # force‑kill any stray Python processes and give the system a breather
#   # echo ">>> Killing leftover python processes..."
#   # pkill -9 python || true

#   # sleep a bit before next run
#   echo ">>> Sleeping for 10s before next run..."
#   pkill -f train_rlhf.py || true
#   sleep 10
# done

# for seed in ${SEEDS}; do
#   # Run the training script with the current seed and algorithm
#   echo "=== Running seed=${seed}, algo=rl==="
#   python scripts/rlhf/train_rlhf.py \
#     --base_seed ${seed} \
#     --rlhf_algorithm rl

#   # force‑kill any stray Python processes and give the system a breather
#   # echo ">>> Killing leftover python processes..."
#   # pkill -9 python || true

#   # sleep a bit before next run
#   echo ">>> Sleeping for 10s before next run..."
#   pkill -f train_rlhf.py || true
#   sleep 10
# done


echo "All runs completed."