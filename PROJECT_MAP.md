# PROJECT_MAP.md

## Project Purpose

This project adapts COLA-style multi-objective reinforcement learning to UAV semantic communication.

## Core Files

- `main_uav.py`: UAV-SemCom training entry.
- `main.py`: Original benchmark entry.
- `agent.py`: COLA/SAC learning pipeline.
- `model.py`: policy, critic, latent encoder models.
- `baselines.py`: heuristic baselines for communication comparison.

## Environment

- `environments/uav_semcom_env.py`: UAV semantic communication environment.
- `environments/__init__.py`: gym environment registration.

## Experiment & Analysis

- `scripts/uav/run_uav.sh`: static UAV paper experiments.
- `scripts/uav/run_uav_mobility.sh`: mobility experiments.
- `scripts/uav/run_uav_line_paper.sh`: line-mobility paper suite.
- `scripts/uav/README.md`: quick usage and overrides for UAV launchers.
- `scripts/bench/run.sh`: original benchmark launcher.
- `scripts/analysis/run_analyze.sh`: quick analysis helper.
- `experiments/analysis/*`: diagnostics and quick analysis tools.

## Output Locations

- `logs/`: training logs and summaries.
- `figures/`: generated plots.
- `paper/`: paper source and generated figures.
- `results/`: exported summary artifacts.

## Suggested Working Flow

1. Run experiments from `scripts/uav/`.
2. Collect metrics from `logs/`.
3. Generate figures via `plot_results*.py`.
4. Summarize results using `final_results*.py`.
