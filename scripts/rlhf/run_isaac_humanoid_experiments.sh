#!/usr/bin/env bash
set -euo pipefail

PROJECT=${SWEEP_PROJECT:-isaac_rlhf}
ENTITY=${SWEEP_ENTITY:-}
GROUP=${SWEEP_GROUP:-"isaac_humanoid_experiments_$(date +%Y%m%d_%H%M%S)"}

SEEDS=(${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20})
PRESET=${PRESET:-isaac_humanoid}
BETA1=${BETA1:-0.01}
BETA2=${BETA2:-0.0016}

echo "Using wandb group: $GROUP"
export WANDB_RUN_GROUP="$GROUP"
[[ -n $ENTITY ]] && export WANDB_ENTITY="$ENTITY"
export WANDB_PROJECT="$PROJECT"

for seed in "${SEEDS[@]}"; do
  for method in ts_last ts_last_lazy ts_last_lazy_opt_design; do
    cmd=(
      python -u scripts/rlhf/train_rlhf.py
      --preset "$PRESET"
      --rlhf_algorithm ts_last
      --base_seed "$seed"
      --beta1 "$BETA1"
      --beta2 "$BETA2"
    )

    case $method in
      ts_last)
        ;;
      ts_last_lazy)
        cmd+=(--lazy)
        ;;
      ts_last_lazy_opt_design)
        cmd+=(--lazy --opt_design)
        ;;
      *)
        echo "Unknown method: $method" >&2
        exit 1
        ;;
    esac

    echo "[$method][$PRESET][seed=$seed] ${cmd[*]}"
    "${cmd[@]}"
  done
done

echo "Isaac Humanoid experiments completed."
