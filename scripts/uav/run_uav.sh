#!/bin/bash
# ============================================================
# COLA-SemCom: static UAV paper experiments (GPU)
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${COLA_PYTHON:-/home/qiankun/.conda/envs/RA_DI/bin/python}"
LOG_DIR="$ROOT_DIR/logs/uav"
mkdir -p "$LOG_DIR"

NUM_STEPS="${NUM_STEPS:-2000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
LATENT_DIM="${LATENT_DIM:-50}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-COLA-SemCom}"
SEEDS="${SEEDS:-1 2 3}"

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
    --latent_dim "$LATENT_DIM" \
    "$@" > "$log_file" 2>&1 < /dev/null &
  echo "  pid=$!"
}

echo "[run_uav] root=$ROOT_DIR"
echo "[run_uav] python=$PYTHON"
echo "[run_uav] steps=$NUM_STEPS eval=$EVAL_INTERVAL seeds=($SEEDS)"

# 1) COLA full
for seed in $SEEDS; do
  launch "cola_full_s${seed}" \
    --seed "$seed" \
    --num_devices 5 \
    --regular_alpha 0.5 \
    --regular_bar 0.25 \
    --consider_other \
    --use_avg
 done

# 2) w/o COR
for seed in $SEEDS; do
  launch "no_cor_s${seed}" \
    --seed "$seed" \
    --num_devices 5 \
    --regular_alpha 0.0 \
    --regular_bar 0.25 \
    --use_avg
 done

# 3) w/o OADM
for seed in $SEEDS; do
  launch "no_oadm_s${seed}" \
    --seed "$seed" \
    --num_devices 5 \
    --regular_alpha 0.5 \
    --regular_bar 0.25 \
    --consider_other \
    --no-use_avg \
    --no-Policy_use_latent \
    --no-Critic_use_both
 done

# 4) COR sensitivity (seed=1)
for alpha in 0.01 0.1 0.5 1.0 5.0; do
  launch "alpha_${alpha}_s1" \
    --seed 1 \
    --num_devices 5 \
    --regular_alpha "$alpha" \
    --regular_bar 0.25 \
    --consider_other \
    --use_avg
 done

echo "All jobs launched. Check: $LOG_DIR"
