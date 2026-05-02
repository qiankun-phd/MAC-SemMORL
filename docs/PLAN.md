# SemMORL Journal Extension Plan (TWC / TCom Target)

**Date**: 2026-05-02
**Source paper**: SemMORL conference paper (`paper/main.tex`, target GLOBECOM 2026 MWN)
**Target journal**: IEEE Transactions on Wireless Communications (preferred) or IEEE Transactions on Communications
**Estimated workload**: 6–9 months single-person full-time

---

## A. Why TWC Over TCom

| Factor | TWC | TCom |
|--------|-----|------|
| Topic fit | UAV / SemCom / channel models — perfect | Cross-layer, protocol — partial |
| Theory bar | Convergence/complexity acceptable | Requires closed-form or regret bound |
| Acceptance rate | ~25–30% | ~25–30% |
| Reviewer pool | Wireless + AI/RL crossover, familiar with our stack | Network/protocol + theory, may demand deeper PHY/MAC |
| Match to extension plan | Multi-UAV + constrained + real channel align well | Same baseline, but PHY incrementals less natural |

**Decision**: target TWC.

---

## B. Literature Gaps (vs Current Conference Paper)

Based on 2024–2026 web survey:

1. **Multi-UAV MARL is mainstream** in TWC (e.g., TWC 2022 multi-UAV MEC offloading, TWC 2023 swarm-vs-jamming). Single-UAV is below the bar.
2. **Constrained MORL is the new SOTA** (C-MORL ICLR 2025, PSL-MORL AAAI 2025). Soft preference alone is conference-level; hard constraints + Lagrangian/dual methods are journal-level.
3. **Safe DRL for wireless** is emerging (arXiv:2507.08653 Peak AoI guarantees). TWC reviewers expect at least one safety/reliability constraint.
4. **Convergence/complexity proofs** appear in nearly every accepted TWC RL paper. Pure-empirical paper has low acceptance odds.
5. **Real channel traces** (e.g., DeepMIMO) increasingly expected. At least one experiment with realistic channel data raises credibility.

---

## C. Tier 1 — Required for TWC Bar

### C.1 Multi-UAV Extension (~30% of total work)

**Current**: 1 UAV serving K IoT devices, single-agent MOMDP.

**Upgrade**:
- M UAVs (typically M = 2–5) cooperatively serve K devices.
- Multi-agent MOMDP: state, action, reward extended to all UAVs.
- Shared OADM latent encoder (federated or centralized aggregator).
- Inter-UAV handover decisions, coverage overlap avoidance, collision avoidance.
- New section in System Model: Multi-UAV Coordination Mechanism.
- New algorithm box: Decentralized SemMORL with parameter sharing.
- Per-UAV partial observability (each UAV sees only its assigned region + neighbour summaries).

**Time**: 6–8 weeks.

### C.2 Constrained Pareto Formulation (~20%)

**Current**: 4 unconstrained objectives via preference vector.

**Upgrade**:
- Add hard constraints, e.g., `P[A_k(t) > A_max] ≤ ε` (reliability), `Σ E(t) ≤ E_total` (energy budget), `min_k ρ_svc(t) ≥ ρ_min` (min service rate).
- Lagrangian / barrier / projection-based satisfaction.
- Cite C-MORL 2025 as related work; position SemMORL as extending their constrained framework with semantic-aware OADM/COR.
- Algorithm extension: dual variable update step inside the training loop.

**Time**: 4–6 weeks.

### C.3 Theoretical Analysis (~15%)

**Current**: no convergence proof.

**Upgrade — at minimum 1–2 pages of formal analysis**:
- **Theorem 1** (contraction): Under linear function approximation, the COR-augmented Bellman backup `T_COR Q = T Q + α·max(ρ̄ − ρ, 0)·ΔQ` is a γ-contraction in the ℓ_2 weighted norm, where γ is the discount factor.
- **Lemma 1** (COR loss bound): When `ρ < ρ̄`, the COR penalty satisfies `L_COR ≤ α(ρ̄ − ρ)·D²` where `D = sup‖Q‖`.
- **Proposition 1** (Pareto regret): With T training steps, the Pareto regret of SemMORL is bounded by `O(T^{−1/2})` plus a constant term induced by α·(ρ̄ − ρ̄_min).
- **Complexity statement**: per-step cost `O(d_z + |λ|·d)`, scaling linearly in K.

**Time**: 3–4 weeks (drafting + verification).

### C.4 Expanded Experiments (~25%)

**Current**: 1 mobility model, 4 RL + 3 WS + 4 heuristic = 11 methods.

**Upgrade**:

#### C.4.1 Mobility models
- Line (current) + Random Waypoint + Levy walk + Group mobility (4 instead of 1).
- Per-mobility Pareto comparisons across all RL methods.

#### C.4.2 More baselines
- C-MORL (ICLR 2025) — constrained MORL SOTA.
- PSL-MORL (AAAI 2025) — Pareto Set Learning via hypernetwork.
- MO-PPO — preference-conditioned PPO.
- Pareto Q-Learning — classic discrete benchmark.
- GMAC (LLM-assisted multi-agent comm) — for SemCom baseline if applicable.

#### C.4.3 Scalability
- K = 5 / 10 / 20 / 50 device counts.
- Plot Pareto coverage vs K to show scalability.

#### C.4.4 Robustness
- Channel estimation error: perturb PL by ±2 dB and ±5 dB.
- Preference perturbation: train under uniform λ, test with skewed λ.
- Traffic distribution shift: train under one mix, test with another.

#### C.4.5 Real channel trace
- DeepMIMO or open UAV channel datasets.
- At least one experiment replaying real measurements as channel ground truth.

**Time**: 5–7 weeks.

---

## D. Tier 2 — Strongly Recommended (Boosts Acceptance Odds)

### D.1 Deployment Cost Analysis (~5%)

- FLOPs and inference latency on Jetson Nano / Raspberry Pi 4 / Jetson Orin Nano.
- Energy per inference (mJ) for actor.
- Side-by-side with traditional optimizers (CVX / IPOPT scalarized solver) on wall-clock time.

**Time**: 2 weeks.

### D.2 Online Adaptation / Meta-MORL (~5%)

- After offline deployment, agent encounters new traffic mix.
- Add online fine-tuning module using last-K transitions.
- Show convergence within minutes (vs full re-train hours).
- Position as a deployment-ready feature.

**Time**: 2–3 weeks.

---

## E. Tier 3 — Nice-to-Have

### E.1 Interpretability Case Study
- Visualize learned trajectory, power, compression decisions for distinct preferences.
- Qualitative analysis of emergent behaviors.

### E.2 Real Semantic Encoder/Decoder
- Replace parametric fidelity model with DeepSC or similar for some traffic types.
- Calibrate parametric model against real DeepSC outputs.

### E.3 Adversarial Preference Attack
- Malicious user forges preference vector to hijack policy.
- Defense via preference validation or robust training.

---

## F. Submission Schedule

```
2026-04-01   GLOBECOM 2026 MWN conference deadline (paper main.tex submitted)
2026-05–07   Start C.1 (multi-UAV) and C.4 (more mobility / baselines) in parallel
2026-08-01   GLOBECOM acceptance/rejection notification
2026-08–10   Continue C.2 (constrained) and C.3 (theory)
2026-11–12   Run new experiments, regenerate figures, draft journal version
2026-12 — 2027-03   Writing, internal review, polish
2027-03      Submit to TWC
2027-09 ~    First-round review feedback (TWC typically 5–7 months)
```

Total clock-time from now to submission: **~10–12 months**, of which **~6–9 months** is concentrated effort.

---

## G. Workload Breakdown

| Task | % of total | Estimated weeks (single FT) |
|------|-----------|-----------------------------|
| C.1 Multi-UAV system + training | 30% | 6–8 |
| C.2 Constrained MORL | 20% | 4–6 |
| C.3 Theory (1–2 pages + proofs) | 15% | 3–4 |
| C.4 Multi-mobility + baselines + real channel | 25% | 5–7 |
| D.x Deployment + online adaptation | 10% | 2–3 |
| Writing + revision (6pp → 12–15pp) | — | 4–6 |
| **Total** | **100%** | **24–34 weeks** |

---

## H. Critical Path & Risk

1. **Multi-UAV training compute** — adding M UAVs roughly multiplies training time by M; budget for 6 GPUs over 6 weeks for C.1.
2. **Theoretical analysis** — if formal proof is hard, fall back to empirical-Pareto-regret trends with confidence intervals; still acceptable with strong empirical results.
3. **Real channel trace integration** — DeepMIMO has UAV-friendly scenarios; verify license and replay infrastructure early.
4. **C-MORL / PSL-MORL baselines** — code may need adaptation for our environment; budget 2 weeks for porting per baseline.

---

## I. Action Items (Next 4 Weeks Before GLOBECOM Submission)

The conference paper is the priority; keep journal extension in design phase only:

- [ ] Lock conference paper at `main.tex`, finalize Apr 1 submission.
- [ ] Sketch multi-UAV MOMDP formulation (1-page draft).
- [x] Identify which C-MORL / PSL-MORL repos to port.
- [ ] Reserve GPU budget and verify DeepMIMO or alternative channel dataset access.
- [ ] Decide single FT vs split between team members.

---

## J. References Used in Plan Synthesis

- C-MORL: arXiv:2410.02236 (ICLR 2025) — constrained Pareto front discovery. Repo: https://github.com/RuohLiuq/C-MORL @ 67473b5.
- PSL-MORL: arXiv:2501.06773 (AAAI 2025) — Pareto Set Learning via hypernetwork.
- Safe DRL for PAoI: arXiv:2507.08653 (2025) — peak-AoI safety constraints.
- Multi-Agent RL UAV Swarm vs Jamming: IEEE TWC 2023, doi 10.1109/TWC.2023.3268082.
- Multi-Agent DRL UAV-MEC Offloading: IEEE TWC 2022, doi 10.1109/TWC.2022.3153316.
- Liu 2024 UAV-SemMEC (cited in conference paper) — single-UAV PPO baseline.
