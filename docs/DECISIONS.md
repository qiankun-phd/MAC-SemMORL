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

## ADR-0004 — Phase 1 multi-UAV GPU reservation (6 GPUs × 6 weeks)

**Date**: 2026-05-02  
**Status**: Pending confirmation (booking required by 2026-08-01)

### Context
Phase 1 multi-UAV training (Task C.1) must run before GLOBECOM notification. The plan calls for a 6 GPU × 6 week window and scaling from the conference single-UAV baseline. We also need a clear fallback if quota slips.

### Decision
1. **Reservation request**: block **~6 GPUs × 6 weeks** on the long-training server `qiankun@172.28.23.182`; revise after the M=2 pilot (D1 in `docs/ROADMAP.md`).
2. **Priority allocation**: request priority access on the same host for Phase 1 runs; dedicate a stable GPU block for multi-UAV experiments.
3. **Per-method wall-clock estimates** based on the conference single-UAV baseline time **T₁** (per-method, per-seed):

   | UAVs (M) | Estimated wall-clock | Assumption |
   |----------|----------------------|------------|
   | 2 | **2.2 × T₁** | linear scaling × M with 10% CTDE overhead |
   | 4 | **4.4 × T₁** | same overhead |
   | 5 | **5.5 × T₁** | same overhead |

4. **Fallback if quota slips**: cap experiments at **M=3** and share the OADM encoder across UAVs more aggressively (single shared encoder + fewer seeds).

### Alternatives Considered
- Request 4 GPUs × 8 weeks: lower concurrency, risks missing multi-seed runs before notification.
- Delay multi-UAV runs to Phase 2: conflicts with GLOBECOM decision window.

### Consequences
- Training throughput is bounded by the 6-GPU block; adjust seeds or M if the pilot over-runs.
- Confirmation must be logged (email/Slack) once booking is accepted; update this ADR with the reference.

---
