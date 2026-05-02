# MAC-SemMORL Roadmap

## Phase 0 — Pre-Conference Preparation (2026-05 to 2026-08)

Light-weight setup work compatible with the conference paper deadline. No major coding commits.

- [ ] **P0.1** Lock conference paper at `COLA_v2/paper/main.tex`, submit to GLOBECOM 2026 MWN by 2026-04-01.
- [ ] **P0.2** Reserve GPU budget (estimate 6 GPUs × 6 weeks for Phase 1 multi-UAV training).
- [ ] **P0.3** Verify DeepMIMO license + replay infrastructure access.
- [ ] **P0.4** Survey C-MORL and PSL-MORL public repos; bookmark for porting in Phase 2.
- [ ] **P0.5** Multi-UAV environment refactor design doc (`docs/DESIGN-multi-uav.md`).

## Phase 1 — Multi-UAV + Constrained (2026-08 to 2026-12)

Triggered by GLOBECOM acceptance/rejection notification.

- [ ] **P1.1** Multi-UAV MOMDP implementation (Task C.1) — refactor `uav_semcom_env.py`, agent.py, training loop.
- [ ] **P1.2** CTDE training loop (Option A, default).
- [ ] **P1.3** Federated-latent training loop (Option B, fallback).
- [ ] **P1.4** Constrained MORL formulation (Task C.2) — Lagrangian dual update step.
- [ ] **P1.5** Coordination experiments: M = 2, 4, 5 UAVs, K = 5, 10, 20 devices.

## Phase 2 — Theory + Baselines + Real Channel (2027-01 to 2027-03)

- [ ] **P2.1** Theorem 1 contraction proof draft (Task C.3).
- [ ] **P2.2** Lemma 1 + Proposition 1 supporting analysis.
- [ ] **P2.3** C-MORL baseline porting (3.5 weeks).
- [ ] **P2.4** PSL-MORL baseline porting (3 weeks).
- [ ] **P2.5** MO-PPO + Pareto Q-Learning + Pareto-PG baseline porting.
- [ ] **P2.6** DeepMIMO real-channel experiment.
- [ ] **P2.7** Mobility-model expansion: Random Waypoint + Levy + Group.

## Phase 3 — Writing + Submission (2027-03 to 2027-09)

- [ ] **P3.1** Journal manuscript draft (12–15 pages, IEEE TWC template).
- [ ] **P3.2** Internal review + revision rounds.
- [ ] **P3.3** Submit to TWC by 2027-03.
- [ ] **P3.4** First-round reviewer response (anticipated 2027-09).

## Critical-Path Decision Points

| ID | Decision | Deadline |
|----|----------|----------|
| D1 | CTDE vs Federated Latent | 2026-06-15 (after M=2 CTDE pilot) |
| D2 | Theorem 1 — full contraction or expected-contraction variant | 2026-08-01 |
| D3 | C-MORL — discrete vs continuous preferences | 2026-09-15 |

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Multi-UAV training compute > budget | Medium | Reduce M from 5 → 3; share OADM encoder. |
| Theorem 1 fails contraction at γ=0.995 | High | Fall back to expected-contraction or tighter ρ̄ (see SKETCHES.md §2.3). |
| C-MORL repo not publicly released by 2026-09 | Medium | Re-implement Stage-1/2 from paper algorithm box. |
| DeepMIMO scenario coverage insufficient | Low | Use synthetic urban-macro fallback. |
