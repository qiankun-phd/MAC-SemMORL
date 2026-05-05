# scripts/uav

Main launchers for UAV semantic communication experiments.

## Scripts

- `run_uav.sh`: static UAV suite (COLA, no-COR, no-OADM, alpha sensitivity)
- `run_uav_mobility.sh`: mobility suite (`none/line/drift`)
- `run_uav_line_paper.sh`: line-mobility paper suite (COLA, ablations, Envelope, WS baselines)

## Common overrides

Use environment variables:

- `COLA_PYTHON`: python executable (default `/home/qiankun/.conda/envs/RA_DI/bin/python`)
- `CUDA_DEVICE`: GPU id (default `0`)
- `NUM_STEPS`: training steps
- `EVAL_INTERVAL`: evaluation interval
- `WANDB_PROJECT`: wandb project name

## Examples

```bash
# Static suite
COLA_PYTHON=/home/qiankun/.conda/envs/RA_DI/bin/python \
NUM_STEPS=400000 EVAL_INTERVAL=20000 \
bash scripts/uav/run_uav.sh

# Mobility quick tier
TIER=1 RUN_BASELINES=1 bash scripts/uav/run_uav_mobility.sh

# Line-paper suite
DEVICE_MOBILITY=line NUM_STEPS=200000 RUN_BASELINES=1 \
bash scripts/uav/run_uav_line_paper.sh
```

## Output

All logs are written to:

- `logs/uav/*.log`
