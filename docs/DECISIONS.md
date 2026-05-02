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

---

## ADR-0005 — Real channel trace: DeepMIMO license check + replay shim

**Date**: 2026-05-02
**Status**: Pending validation

### Context
Task C.4.5 requires at least one experiment replaying realistic channel traces.
DeepMIMO is a common choice for reproducible ray-tracing-based channels, but we
must confirm licensing and verify that we can feed samples into our environment
PHY pipeline (pathloss/SNR/throughput).

### Decision
- Implement a minimal replay shim at `src/channel/deepmimo_replay.py`.
- Defer final DeepMIMO adoption until its license is reviewed and recorded.

### Alternatives Considered
- 3GPP TR 38.901 UMa/UMi synthetic channel generation as a license-safe fallback.
- Other open UAV channel datasets (limited availability/coverage).

### Consequences
- We can prototype the replay plumbing immediately using exported `.npz` channel
  tensors, while keeping the dataset choice reversible.
- A follow-up is required to:
  1) paste the DeepMIMO license text/URL here,
  2) record whether academic use is allowed,
  3) cite the exact scenario(s) selected (UAV-friendly, elevated TX).
- The Phase 1 experiment matrix must be revisited after the M=2 pilot to confirm scaling.

See `docs/PLAN.md` §H and `docs/ROADMAP.md` P0.2.

---
