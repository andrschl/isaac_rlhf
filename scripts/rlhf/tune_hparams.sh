#!/usr/bin/env bash
set -euo pipefail

PROJECT=${SWEEP_PROJECT:-isaac_rlhf}
ENTITY=${SWEEP_ENTITY:-}
GROUP=${SWEEP_GROUP:-"rlhf_random_$(date +%Y%m%d_%H%M%S)"}

SEEDS=(${SEEDS:-1 2 3})
SAMPLES=${NUM_SAMPLES:-30}

BETA1_MIN=${BETA1_MIN:-0.001}
BETA1_MAX=${BETA1_MAX:-1.0}
BETA2_MIN=${BETA2_MIN:-0.001}
BETA2_MAX=${BETA2_MAX:-1.0}

TS_DOUBLE_RUNS=${TS_DOUBLE_RUNS:-2}
(( TS_DOUBLE_RUNS % 2 )) && TS_DOUBLE_RUNS=$((TS_DOUBLE_RUNS + 1))

echo "Using wandb group: $GROUP"
export WANDB_RUN_GROUP="$GROUP"
[[ -n $ENTITY ]] && export WANDB_ENTITY="$ENTITY"
export WANDB_PROJECT="$PROJECT"

float_rand() {
  python - "$1" "$2" <<'PY'
import random, math, sys
lo, hi = map(float, sys.argv[1:3])
value = 10 ** random.uniform(math.log10(lo), math.log10(hi))
print(f"{value:.6g}")
PY
}

for seed in "${SEEDS[@]}"; do
  # vanilla
  cmd=(python -u scripts/rlhf/train_rlhf.py --task gridworld --rlhf_algorithm vanilla --base_seed "$seed")
  echo "[vanilla] ${cmd[*]}"
  "${cmd[@]}"

  # ts_last / ts_double
  for ((i=0; i<SAMPLES; ++i)); do
    for method in ts_last ts_double; do
      beta1=$(float_rand "$BETA1_MIN" "$BETA1_MAX")
      beta2=$(float_rand "$BETA2_MIN" "$BETA2_MAX")
      cmd=(python -u scripts/rlhf/train_rlhf.py
            --task gridworld
            --rlhf_algorithm "$method"
            --base_seed "$seed"
            --beta1 "$beta1"
            --beta2 "$beta2")
      if [[ $method == "ts_double" ]]; then
        cmd+=(--num_rl_runs "$TS_DOUBLE_RUNS")
      fi
      echo "[$method] ${cmd[*]}"
      "${cmd[@]}"
    done
  done
done

echo "All runs completed."
