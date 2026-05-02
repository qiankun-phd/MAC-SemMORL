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
