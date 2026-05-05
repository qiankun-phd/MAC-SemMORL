# STRUCTURE.md

## Standardized Layout (UAV-SemCom + COLA)

```text
COLA/
|-- agent.py
|-- model.py
|-- main.py
|-- main_uav.py
|-- baselines.py
|-- environments/
|-- experiments/
|   |-- analysis/
|   |   |-- analyze_results.py
|   |   |-- calibrate_rewards.py
|   |   |-- check_progress.py
|   |   |-- quick_stats.py
|   |   |-- reward_diagnosis.py
|   |   `-- snr_diagnosis.py
|   |-- compare_mobility_cola_vs_baselines.py
|   |-- mobility_ablation_eval.py
|   `-- plot_uav_trajectory.py
|-- logs/
|-- paper/
|-- figures/
|-- plot_results.py
|-- final_results.py
|-- plot_results_line.py
|-- final_results_line.py
|-- scripts/
|   |-- uav/
|   |   |-- run_uav.sh
|   |   |-- run_uav_mobility.sh
|   |   |-- run_uav_line_paper.sh
|   |   `-- README.md
|   |-- bench/
|   |   `-- run.sh
|   `-- analysis/
|       `-- run_analyze.sh
|-- docs/
|   `-- STRUCTURE.md
`-- PROJECT_MAP.md
```

## Notes

- Main UAV entry is `main_uav.py`.
- Main algorithm implementation is in `agent.py` and `model.py`.
- Preferred run scripts are under `scripts/`.
- All UAV logs should be written under `logs/uav/`.
