#!/bin/bash
# ============================================================
# COLA-SemCom mobility experiments (GPU)
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${COLA_PYTHON:-/home/qiankun/.conda/envs/RA_DI/bin/python}"
LOG_DIR="$ROOT_DIR/logs/uav"
mkdir -p "$LOG_DIR"

TIER="${TIER:-2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-COLA-SemCom}"
RUN_BASELINES="${RUN_BASELINES:-0}"
BASELINE_EPISODES="${BASELINE_EPISODES:-50}"

if [[ "$TIER" == "1" ]]; then
  MOBS=(none line)
  SEEDS=(1)
  NUM_STEPS="${NUM_STEPS:-400000}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
else
  MOBS=(none line drift)
  SEEDS=(1 2 3)
  NUM_STEPS="${NUM_STEPS:-2000000}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
fi

launch() {
  local exp_name="$1"
  shift
  local log_file="$LOG_DIR/${exp_name}.log"

  echo "Launch: $exp_name -> $log_file"
  nohup "$PYTHON" "$ROOT_DIR/main_uav.py" \
    --cuda --cuda_device "$CUDA_DEVICE" \
    --wandb_project "$WANDB_PROJECT" --wandb_offline \
    --num_steps "$NUM_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_devices 5 \
    --latent_dim 50 \
    --regular_alpha 0.5 \
    --regular_bar 0.25 \
    --consider_other \
    --use_avg \
    "$@" > "$log_file" 2>&1 < /dev/null &
  echo "  pid=$!"
}

echo "[run_uav_mobility] root=$ROOT_DIR"
echo "[run_uav_mobility] python=$PYTHON"
echo "[run_uav_mobility] tier=$TIER steps=$NUM_STEPS eval=$EVAL_INTERVAL"

auto_speed="${DEVICE_SPEED:-0.0}"
for mob in "${MOBS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    exp_name="cola_mob_${mob}_seed${seed}"
    launch "$exp_name" \
      --seed "$seed" \
      --device_mobility "$mob" \
      --device_speed "$auto_speed" \
      --exp_name "$exp_name"
  done
 done

if [[ "$RUN_BASELINES" == "1" ]]; then
  echo "Running baselines for each mobility..."
  for mob in "${MOBS[@]}"; do
    "$PYTHON" "$ROOT_DIR/baselines.py" --device_mobility "$mob" --n_episodes "$BASELINE_EPISODES"
  done
else
  echo "Skip baselines. To run them:"
  for mob in "${MOBS[@]}"; do
    echo "  $PYTHON $ROOT_DIR/baselines.py --device_mobility $mob --n_episodes $BASELINE_EPISODES"
  done
fi

echo "All mobility jobs launched. Logs: $LOG_DIR"
