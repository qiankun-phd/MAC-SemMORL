# MAC-SemMORL

**Multi-Agent Constrained SemMORL** — Journal extension of the SemMORL conference work, targeting IEEE Transactions on Wireless Communications (TWC).

## Overview

This repository hosts the journal extension of the GLOBECOM 2026 conference paper *SemMORL: A Conflict-Aware MORL Framework for UAV-enabled Semantic IoT Resource Allocation*.

The extension lifts the framework from single-UAV unconstrained MORL to:
- **Multi-Agent**: M cooperatively-serving UAVs with CTDE or federated-latent training.
- **Constrained**: hard reliability/energy/service-rate constraints via Lagrangian methods.
- **Theory-backed**: contraction proof for the COR-augmented Bellman backup, plus Pareto-regret bound.
- **Broader empirical scope**: 4 mobility models, 3 new SOTA baselines (C-MORL, PSL-MORL, MO-PPO), real channel traces (DeepMIMO).

## Status

- Conference paper (single-UAV SemMORL): submitted to GLOBECOM 2026 MWN, deadline 2026-04-01.
- Journal extension (this repo): planning phase. Coding begins after conference notification (~2026-08).

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for phase milestones.

## Documents

| Doc | Purpose |
|-----|---------|
| [`docs/PLAN.md`](docs/PLAN.md) | Full extension plan (Tier 1/2/3 work, schedule, references). |
| [`docs/SKETCHES.md`](docs/SKETCHES.md) | First-cut technical sketches: multi-UAV MOMDP, Theorem 1, baseline porting. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phase 0/1/2/3 milestones and deadlines. |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decision records (ADRs). |

## Origin Repository

Conference codebase and paper source live in [`COLA_v2`](https://github.com/qiankun-phd/COLA_v2) (private). This repo branches off from the conference algorithm baseline.

## License

MIT — see [`LICENSE`](LICENSE).
