#!/usr/bin/env bash
set -euo pipefail

PROJECT=${SWEEP_PROJECT:-isaac_rlhf}
ENTITY=${SWEEP_ENTITY:-}
GROUP=${SWEEP_GROUP:-"rlhf_gridworld_$(date +%Y%m%d_%H%M%S)"}

SEED_START=${SEED_START:-15}
NUM_SEEDS=5
BETA1=0.05
BETA2=0.1

echo "Using wandb group: $GROUP"
export WANDB_RUN_GROUP="$GROUP"
[[ -n $ENTITY ]] && export WANDB_ENTITY="$ENTITY"
export WANDB_PROJECT="$PROJECT"

for ((offset = 0; offset < NUM_SEEDS; ++offset)); do
  seed=$((SEED_START + offset))

  # vanilla
  cmd=(python -u scripts/rlhf/train_rlhf.py --task gridworld --rlhf_algorithm vanilla --base_seed "$seed")
  echo "[vanilla] ${cmd[*]}"
  "${cmd[@]}"

  # ts_last with fixed betas
  cmd=(python -u scripts/rlhf/train_rlhf.py
        --task gridworld
        --rlhf_algorithm ts_last
        --base_seed "$seed"
        --beta1 "$BETA1"
        --beta2 "$BETA2")
  echo "[ts_last] ${cmd[*]}"
  "${cmd[@]}"
done

echo "All runs completed."
