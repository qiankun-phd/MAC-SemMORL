#!/bin/bash
# ============================================================
# UAV paper experiment suite under line mobility
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${COLA_PYTHON:-/home/qiankun/.conda/envs/RA_DI/bin/python}"
LOG_DIR="$ROOT_DIR/logs/uav"
mkdir -p "$LOG_DIR"

NUM_STEPS="${NUM_STEPS:-200000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
MODEL_SAVED_STEP="${MODEL_SAVED_STEP:-100000}"
DEVICE_MOBILITY="${DEVICE_MOBILITY:-line}"
DEVICE_SPEED="${DEVICE_SPEED:-0.0}"
RUN_BASELINES="${RUN_BASELINES:-1}"
BASELINE_EPISODES="${BASELINE_EPISODES:-50}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-COLA-SemCom}"
SEEDS="${SEEDS:-1 2 3}"
WS_SEEDS="${WS_SEEDS:-1}"

launch() {
  local exp_name="$1"
  shift
  local log_file="$LOG_DIR/${exp_name}.log"

  echo "Launch: ${exp_name} -> ${log_file}"
  nohup "$PYTHON" "$ROOT_DIR/main_uav.py" \
    --cuda --cuda_device "$CUDA_DEVICE" \
    --wandb_project "$WANDB_PROJECT" --wandb_offline \
    --num_steps "$NUM_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --model_saved_step "$MODEL_SAVED_STEP" \
    --device_mobility "$DEVICE_MOBILITY" \
    --device_speed "$DEVICE_SPEED" \
    --exp_name "$exp_name" \
    "$@" > "$log_file" 2>&1 < /dev/null &
  echo "  pid=$!"
}

echo "[run_uav_line_paper] root=$ROOT_DIR"
echo "[run_uav_line_paper] python=$PYTHON"
echo "[run_uav_line_paper] mobility=$DEVICE_MOBILITY steps=$NUM_STEPS eval=$EVAL_INTERVAL"
echo "[run_uav_line_paper] seeds=($SEEDS) ws_seeds=($WS_SEEDS)"

echo "=== 1) COLA ==="
for seed in $SEEDS; do
  launch "cola_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.5 \
    --regular_bar 0.25 \
    --consider_other \
    --use_avg
 done

echo "=== 2) w/o COR ==="
for seed in $SEEDS; do
  launch "no_cor_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.0 \
    --regular_bar 0.25 \
    --use_avg
 done

echo "=== 3) w/o OADM ==="
for seed in $SEEDS; do
  launch "no_oadm_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.5 \
    --regular_bar 0.25 \
    --consider_other \
    --no-use_avg \
    --no-Policy_use_latent \
    --no-Critic_use_both
 done

echo "=== 4) Envelope SAC ==="
for seed in $SEEDS; do
  launch "envelope_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.0 \
    --regular_bar 0.25 \
    --no-consider_other \
    --no-use_avg \
    --no-Policy_use_latent \
    --no-Critic_use_both
 done

echo "=== 5) Weighted-sum SAC ==="
for seed in $WS_SEEDS; do
  launch "ws_fidelity_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.0 \
    --regular_bar 0.25 \
    --no-consider_other \
    --no-use_avg \
    --no-Policy_use_latent \
    --no-Critic_use_both \
    --fixed_weight 0.4 0.2 0.2 0.2

done

for seed in $WS_SEEDS; do
  launch "ws_balanced_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.0 \
    --regular_bar 0.25 \
    --no-consider_other \
    --no-use_avg \
    --no-Policy_use_latent \
    --no-Critic_use_both \
    --fixed_weight 0.25 0.25 0.25 0.25

done

for seed in $WS_SEEDS; do
  launch "ws_energy_line_s${seed}" \
    --seed "$seed" \
    --regular_alpha 0.0 \
    --regular_bar 0.25 \
    --no-consider_other \
    --no-use_avg \
    --no-Policy_use_latent \
    --no-Critic_use_both \
    --fixed_weight 0.2 0.2 0.4 0.2

done

if [[ "$RUN_BASELINES" == "1" ]]; then
  echo "=== 6) Heuristic baselines (${DEVICE_MOBILITY}) ==="
  "$PYTHON" "$ROOT_DIR/baselines.py" \
    --device_mobility "$DEVICE_MOBILITY" \
    --device_speed "$DEVICE_SPEED" \
    --n_episodes "$BASELINE_EPISODES"
else
  echo "Skip baselines (RUN_BASELINES=$RUN_BASELINES)"
fi

echo "Line-paper suite launched. Logs: $LOG_DIR"
