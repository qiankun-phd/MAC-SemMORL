---
name: Experiment
about: Track a specific experiment run (training, ablation, baseline comparison).
title: "[Exp] "
labels: experiment
assignees: ''
---

## Hypothesis
What are we testing?

## Setup
- Method / variant: 
- Environment config: (M UAVs, K devices, mobility model, etc.)
- Seeds: (e.g., 4, 5, 6, 7, 8, 9 — 6 seeds)
- Hyperparameters: (or link to config file)
- Hardware: (GPU model, count)

## Expected Wall-Clock
__ hours / __ GPU-hours

## Success Criteria
- Metric: (e.g., HV ≥ 2.6e13)
- Comparison baseline: (e.g., conference SemMORL HV = 2.54e13)

## Result Location
- Raw outputs: `results/expXXX/`
- Plots: `results/expXXX/figures/`

## Outcome (fill after run)
- [ ] Hypothesis confirmed
- [ ] Hypothesis rejected
- [ ] Inconclusive — needs follow-up

Notes:
