# Architecture Decision Records (ADR)

Append-only log of design decisions. Each entry: date, context, decision, alternatives considered, consequences.

---

## ADR-0001 — Target venue: TWC over TCom

**Date**: 2026-05-02
**Status**: Accepted

### Context
SemMORL conference paper targets GLOBECOM 2026 MWN. Journal extension needs venue selection between IEEE TWC and IEEE TCom.

### Decision
Target IEEE TWC.

### Alternatives Considered
- IEEE TCom: stronger match for cross-layer/protocol work; weaker match for UAV/SemCom topic; demands deeper PHY/MAC theory.
- IEEE JSAC special issue: opportunistic, conditional on a relevant CFP.

### Consequences
- Reviewer pool will be wireless + AI/RL crossover, familiar with our stack.
- Theory bar acceptable: convergence + complexity proofs sufficient.
- Multi-UAV + constrained extensions align well with TWC trends.

See `docs/PLAN.md` §A for full comparison.

---

## ADR-0002 — Project name: MAC-SemMORL

**Date**: 2026-05-02
**Status**: Accepted

### Context
Need a short name for the journal extension repo that captures both major upgrades vs the conference paper.

### Decision
**MAC-SemMORL** (Multi-Agent Constrained SemMORL).

### Alternatives Considered
- SemMORL-Pro — too generic, doesn't signal upgrade.
- MA-SemMORL — captures multi-UAV but not constraints.
- C-SemMORL — captures constraints but not multi-UAV.

### Consequences
- "MAC" prefix is recognizable convention (cf. C-MORL, MA-PPO).
- Repo name `mac-semmorl` (kebab-case).

---

## ADR-0003 — Multi-UAV training: CTDE first, Federated-Latent fallback

**Date**: 2026-05-02
**Status**: Pending validation (decision deadline 2026-06-15)

### Context
Two coordination architectures are viable for multi-UAV MORL: CTDE (simple, leverages OADM) vs Federated Latent (scalable, novel).

### Decision
Default to **CTDE for initial M=2 experiments**. Switch to Federated Latent if wall-clock or reward variance becomes prohibitive at M ≥ 4.

### Alternatives Considered
- Federated Latent from day one: better scalability but extra synchronization heuristic and convergence risk.
- Fully decentralized (no shared encoder): worst sample efficiency.

### Consequences
- Lower initial engineering risk.
- Locks us into a fallback path; budget +2 weeks if switch is needed.

See `docs/SKETCHES.md` §1.3 for technical detail.

---

## ADR-0004 — Phase 1 multi-UAV training: GPU reservation and fallback

**Date**: 2026-05-02
**Status**: Accepted

### Context
Phase 1 (2026-08 to 2026-12) requires long-running multi-UAV MARL training (M = 2, 4, 5) and will be blocked by compute availability. We need to confirm GPU availability before the GLOBECOM notification (2026-08-01) and document a fallback plan if quota slips.

### Decision
- Reserve **~6 GPUs for ~6 weeks** for the Phase 1 pilot + sweep.
- Prefer running long jobs on **`qiankun@172.28.23.182`** (long-training server) as the primary allocation target.
- Use the following wall-clock estimate model (revise after pilot): take the conference single-UAV baseline wall-clock and scale approximately with the number of UAVs and coordination overhead.

### Wall-Clock Estimates (Relative)
Let `T1` be the measured single-UAV (M=1) wall-clock for one full training run at the conference baseline setting.

- **M=2**: ~`2.4 × T1` (≈2× compute + ~20% coordination/comm overhead)
- **M=4**: ~`5.2 × T1` (≈4× compute + ~30% overhead)
- **M=5**: ~`6.8 × T1` (≈5× compute + ~35% overhead)

Notes:
- These are planning estimates for scheduling only; actual scaling depends on CTDE implementation details, environment step throughput, and GPU utilization.
- After the M=2 pilot, re-fit the overhead term and update this ADR (append an addendum) and `docs/ROADMAP.md` P0.2 if needed.

### Reservation Confirmation (In Writing)
Record the booking confirmation reference here once received:

- **Channel**: (email / Slack)
- **Date confirmed**: (YYYY-MM-DD)
- **Confirmed by**: (name / handle)
- **Allocation window**: (YYYY-MM-DD → YYYY-MM-DD)
- **Resources**: (~6 GPUs, model, RAM if specified)
- **Notes**: (queue/priority constraints, preemption policy)

### Alternatives Considered
- On-demand / best-effort usage: high risk of missing the Phase 1 schedule.
- Fewer GPUs (e.g., 2–4): increases time-to-results; compresses the window for M=4/5 sweeps.

### Fallback Plan (If Quota Slips)
- Reduce target swarm size from **M=5 → M=3** for the Phase 1 scalability section (still report M=2 as baseline).
- Share the **OADM latent encoder** more aggressively (single shared encoder across UAVs, less frequent synchronization) to reduce compute and improve sample efficiency.
- Narrow the hyperparameter sweep (fewer seeds / fewer preference vectors) and prioritize the ablation set that supports CTDE vs Federated-Latent decision D1.

### Consequences
- Compute allocation becomes an explicit project dependency for Phase 1 execution.
- The Phase 1 experiment matrix must be revisited after the M=2 pilot to confirm scaling.

See `docs/PLAN.md` §H and `docs/ROADMAP.md` P0.2.

---

## ADR-0005 — Multi-UAV refactor: per-UAV federated CTDE (DESIGN §4 path)

**Date**: 2026-05-05
**Status**: Accepted

### Context
Two viable architectures for the multi-UAV upgrade exist. Joint-action CTDE keeps the existing `SacAgent` untouched (one fat policy outputs `M(4+K)`-dim joint action) at the cost of zero deployment-side decentralisation and limited TWC novelty. Per-UAV federated CTDE follows `docs/DESIGN-multi-uav.md` §4 (M actors, M critics, shared OADM encoder, optional FedAvg sync) — more code, more novelty.

### Decision
Take the per-UAV federated path. Concretely:

1. **New env class** `MultiUAVSemComEnv` in `environments/uav_semcom_multi_env.py`, registered as `UAV-SemCom-Multi-v0`. Keeps the original `UAVSemComEnv` untouched for backward compatibility with conference reproductions.
2. **Joint state, joint action interface** at the env layer — flat vectors so it remains a clean `gym.Env`. Agent splits per-UAV slices using `env.num_uavs` attribute.
3. **New agent class** `MultiAgentSemMORL` in a follow-up PR — wraps M actors + 1 shared OADM encoder + 1 centralized critic (CTDE phase). Optional FedAvg encoder sync added later if M ≥ 4 scaling fails.
4. **Phase 1.1 pilot**: M=2 with shared encoder, no FedAvg, centralised critic (CTDE Option A from DESIGN §2.1).
5. **Phase 1.2 fallback**: switch to FedAvg (Option B from DESIGN §2.2) if Decision D1 (deadline 2026-06-15) gate fails.

### Alternatives Considered
- **Joint-action CTDE**: zero agent.py change but no deployment-side decentralisation, weaker TWC novelty story, and the joint action dim grows as `M(4+K)` which becomes unwieldy at M=5/K=20 (210 dims). Rejected for the journal version; could be revisited as a baseline.

### Consequences
- ~350 lines of new env code + ~400 lines of new agent code expected.
- Phase 1 pilot timing target stays 4–6 weeks per `docs/PLAN.md` §C.1.
- Action space differs from the conference single-UAV case by per-UAV scheduling logits `x_{m,k}` — these are added only when `num_uavs ≥ 2` so the conference numerical reproduction path is unaffected.

See `docs/DESIGN-multi-uav.md` §1, §2, §4 for the full formulation.

---
