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

## ADR-0006 — Fleet-energy reward normalisation (per-UAV bound, fleet sum numerator)

**Date**: 2026-05-06
**Status**: Accepted

### Context
The first M=2 pilot at 1M steps showed sum reward within 0.14% of the M=1 conference baseline, which was suspicious. Running `scripts/diagnose_pilot.py` on the final checkpoint with 10 episodes at uniform preference produced:

| metric                    | M=2 pilot         | conference M=1 (Table II) |
|---------------------------|-------------------|---------------------------|
| weighted_avg_fidelity     | 0.8583 ± 0.026    | 0.88                      |
| mean_aosi                 | 1.0273 ± 0.039    | 1.06                      |
| **energy_per_episode_kJ** | **45.32 ± 1.17**  | **21.96**                 |
| jain_fairness             | 0.9677 ± 0.019    | 0.98                      |
| service_rate              | 0.9895 ± 0.020    | 0.99                      |

Both UAVs were operating (energy is correctly ≈ 2× single-UAV), yet the agent's energy reward `r3` looked unchanged from the M=1 baseline. The root cause was the original normalisation in `MultiUAVSemComEnv.__init__`:

```python
self.max_energy = (...) * slot_duration * num_uavs   # ← scaled by M
self.min_energy = self.hover_power * slot_duration * num_uavs
```

`r3` is computed as `(max_energy − e_total − coll_pen) / (max_energy − min_energy)`. With both bounds scaled by `M` and `e_total ≈ M × single_actual`, every factor of `M` cancels out — the normalised ratio is invariant to the swarm size, so the multi-objective agent has **no incentive to coordinate UAVs to reduce fleet energy**. This silently neutralised one of the four objectives the journal extension is supposed to optimise.

### Decision
Drop the `* num_uavs` factor from both bounds. Keep `max_energy` / `min_energy` as **per-UAV** worst / best case for one slot; let `e_total` (the fleet sum across all M UAVs in the slot) flow through the numerator unscaled.

Effect on the normalised reward:
- **M=1**: ratio identical to the conference paper (single-UAV baseline preserved bit-exactly).
- **M=2 each UAV at full effort**: numerator ≈ `(per_uav_max − 2·per_uav_actual)` — drives `r3` deeply negative, penalising duplicated work.
- **M=2 each UAV at half effort (load-balanced)**: numerator ≈ `(per_uav_max − per_uav_actual_full)` — matches M=1 full-load reward, so coordination is rewarded relative to brute-force duplication.

### Alternatives Considered
- **Keep fleet-level bounds, raise `regular_alpha` to compensate**: adjusts the optimiser, not the objective. Doesn't fix the underlying signal.
- **Add a separate "fleet energy" objective r5**: changes the MOMDP from 4-D to 5-D, breaks comparability with the conference paper Pareto front, and asks the OADM encoder to handle a new dimension late in the schedule. Rejected.
- **Normalise per-UAV inside the agent (move the fix to `multi_agent.py`)**: leaks reward design into the agent and makes the env semantics inconsistent with the single-UAV class. Rejected.

### Consequences
- Existing M=2 pilot checkpoint (`COLA-SemCom-seed1_dev5`) is invalidated — the policy was trained against the buggy reward and must not be carried forward into Phase 1 results.
- Re-run M=2 pilot at 1M steps with a distinct `--exp_name` (e.g. `pilot-M2-fixenergy-seed1`) so it doesn't overwrite the conference single-UAV reproduction.
- `analyze_results.py` baselines for M=4 / M=5 sweeps must use post-fix runs only — flag any pre-fix checkpoint with a `legacy-energy-norm` tag if retained for ablation.
- The journal experimental section gains a small "ablation on energy normalisation" sidebar — the buggy form is a concrete justification for why fleet-sum vs per-UAV-bound matters and supports the multi-UAV novelty story for TWC.
- Reward scale for `r3` shifts: under-coordinated fleets see negative `r3` values where the old form returned ≥ 0. Logging dashboards expecting `r3 ≥ 0` need their y-axis range reviewed.

See `scripts/diagnose_pilot.py` (PR #23) for the diagnostic that surfaced this and `environments/uav_semcom_multi_env.py:158` for the fixed code.

---

## ADR-0008 — Drop the `r_energy ≥ 0.5` floor in the multi-UAV env

**Date**: 2026-05-07
**Status**: Accepted (follow-up to ADR-0006)
**Supersedes**: the floor clause in ADR-0006's effect description

### Context
After ADR-0006 unscaled the energy bounds, the M=2 pilot at 1M steps was diagnosed against `scripts/diagnose_pilot.py`:

| metric                    | pre-ADR-0006 (buggy)  | post-ADR-0006 (this) | conference M=1 |
|---------------------------|-----------------------|----------------------|----------------|
| weighted_avg_fidelity     | 0.858 ± 0.026         | 0.803 ± 0.090        | 0.88           |
| mean_aosi                 | 1.027 ± 0.039         | **4.017 ± 8.80**     | 1.06           |
| **energy_per_episode_kJ** | **45.32 ± 1.17**      | **77.98 ± 17.33**    | 21.96          |
| jain_fairness             | 0.968                 | 0.948                | 0.98           |
| service_rate              | 0.990                 | 0.945                | 0.99           |

Energy got *worse*, not better, after the fix that was supposed to make it visible. The AoSI standard deviation was 8× the post-fix mean — the policy was diverging.

Root cause: the multi-UAV step function inherited a `r_energy = max(0.5, r_energy)` floor from the single-UAV class. In the single-UAV regime, `e_total ≤ max_energy` always (one UAV cannot exceed its own per-step max), so the unfloored ratio is in `[0, 1]` and the floor is redundant — `r_energy ∈ [0.5, 4.5]`.

After ADR-0006, the multi-UAV bounds are per-UAV but `e_total` is a *fleet sum* across all M UAVs. For any swarm running both UAVs at non-trivial effort, `e_total > max_energy` and the unfloored ratio is negative. The 0.5 floor then **clamps `r_energy` to a constant 0.5 across the entire operating regime**, eliminating the gradient on the energy objective. The agent loses the energy signal entirely, so its policy on objective 3 wanders — driving up energy *and* AoSI variance.

### Decision
Remove the `r_energy = max(0.5, r_energy)` floor from `MultiUAVSemComEnv.step()`. Let `r_energy` go negative when the fleet exceeds the per-UAV worst-case energy bound. Single-UAV class keeps the floor untouched (it never fires there).

Effective range after this fix:
- **M=1** (`UAVSemComEnv`, untouched): `r_energy ∈ [0.5, 4.5]`, range 4.0.
- **M=2**: `r_energy ∈ [-3.5, 4.5]`, range 8.0.
- **M=4**: `r_energy ∈ [-11.5, 4.5]`, range 16.0.

The lower bound `0.5 − 4(M−1)` is the worst case (every UAV at full effort, single-UAV idea of max). In practice the policy will gravitate well above this floor since other reward components (`r_1`, `r_2`, `r_4`) remain in `[0.5, 4.5]` and preference weighting balances the trade-off.

### Alternatives Considered
- **Rescale by num_uavs in the multiplier** (`* 4.0 / num_uavs`): preserves the magnitude balance with other r_i but partially re-introduces the M-cancellation issue ADR-0006 was meant to fix. Rejected — same fundamental tension as the buggy original form.
- **Floor at a deeper bound** (e.g., `max(-4*(M-1), r_energy)`): bounds the magnitude but adds a hyperparameter to tune. Rejected for now in favour of the simpler "no floor" form; can revisit in PR-D of Issue #6 if the negative tail destabilises Q-learning.
- **Add an offset to keep `r_energy ≥ 0`** (e.g., `r_energy + 4*(M-1)`): preserves sign convention but breaks comparability with the conference single-UAV scale. Rejected.
- **Move the energy term into the constrained-MORL formulation only** (PR-26's Lagrangian path with `c_2`): correct long-term answer, but the unconstrained Pareto baselines (used as comparison points in the journal experiments section) still need a non-degenerate energy reward signal. Need both.

### Consequences
- **Existing M=2 fix-energy checkpoint (`pilot-M2-fixenergy-seed1`) is invalidated**. The numbers in the diagnosis table above become the "after ADR-0006, before ADR-0008" pre-fix data point, useful for the paper's reward-design ablation.
- Re-run the M=2 pilot at 1M steps with a fresh `--exp_name` (e.g., `pilot-M2-fixfloor-seed1`).
- Multi-UAV `r_3` is now signed; logging dashboards and CSV writers must expect `r_3 < 0` rows. The conference single-UAV path is bit-identical.
- The journal's reward-design ablation now spans **three** points instead of two: buggy-original (`max_energy *= num_uavs`, floor at 0.5), ADR-0006-only (no `* num_uavs`, floor still at 0.5 — the dead-gradient case documented above), and ADR-0008 (current — no floor). The dead-gradient case is itself a useful negative result for the paper.
- Theorem 1 in `paper/theorem1.tex` needs a one-line update: `‖r‖_∞ ≤ 4.5` should become `‖r‖_∞ ≤ 4.5 + max(0, 4(M−1))` to bound the contraction-coefficient analysis under the new range. Lands with PR-D of Issue #6.

See `environments/uav_semcom_multi_env.py:337` for the change and the diagnostic table above for the data.

---

## ADR-0009 — Soft floor `max(-4.0, r_energy)` for M ≥ 3 stability

**Date**: 2026-05-08
**Status**: Accepted (follow-up to ADR-0008)

### Context
After ADR-0008 removed the catastrophic 0.5 floor, the M=2 fix-floor pilot at 1M steps converged cleanly (HV 442B, energy 50 ± 4 kJ, AoSI 1.22 ± 0.5). The first M=4 fix-floor pilot at the same protocol was qualitatively different — the diagnostic on `policy_final.pkl`:

| metric                    | M=4 pilot (no floor) | M=2 fix-floor | conference M=1 |
|---------------------------|----------------------|---------------|----------------|
| weighted_avg_fidelity     | **0.51 ± 0.09**      | 0.83          | 0.88           |
| **mean_aosi**             | **29.3 ± 17.9**      | 1.22          | 1.06           |
| **energy_per_episode_kJ** | **197.7 ± 19.5**     | 50.2          | 22.0           |
| jain_fairness             | 0.90                 | 0.96          | 0.98           |
| **service_rate**          | **0.64 ± 0.26**      | 0.97          | 0.99           |

HV at the final eval was **0.0** — every preference sample produced a Pareto point dominated by the [0,0,0,0] reference because `r3 < 0` was driving the entire 4-vector below the reference. AoSI was 27× the conference baseline with a standard deviation of 18 — policy was systematically starving subsets of devices. Energy 9× M=1 indicates all four UAVs were operating near full power with no coordination.

Root cause: under ADR-0008's unbounded form, the multi-UAV r_energy range is `[0.5 − 4(M−1), 4.5]`. For M=2 that's `[−3.5, 4.5]` (size 8.0), comparable to the other rewards' [0.5, 4.5] (size 4.0). For M=4 it's `[−11.5, 4.5]` (size 16.0) — **the energy reward can dominate the other three components by 4× in magnitude**. A few bad slots per episode drive the policy gradient hard toward avoiding any energy use, but the optimisation also has to balance fidelity / freshness / fairness, so the agent oscillates and ultimately converges on a degenerate "spam all UAVs" attractor.

### Decision
Add a soft floor `r_energy = max(-4.0, r_energy)` after the unfloored computation in `MultiUAVSemComEnv.step`. Single-UAV class is untouched.

Effective per-step ranges:

| M | range size | dominates other rewards by? |
|---|-----------|----------------------------|
| 1 | [0.5, 4.5]   = 4.0  | bit-identical to conference (single-UAV class untouched) |
| 2 | [−3.5, 4.5]  = 8.0  | 2× — bit-identical to ADR-0008 (lower bound never hits −4.0) |
| 3 | [−4.0, 4.5]  = 8.5  | 2× (was 12.5 / 3.1× before this fix) |
| 4 | [−4.0, 4.5]  = 8.5  | 2× (was 16.0 / 4× before this fix) |
| 5 | [−4.0, 4.5]  = 8.5  | 2× (was 20.0 / 5× before this fix) |

Why `−4.0` specifically:
- Symmetric in magnitude with the upper bound +4.5 (close to but not exactly equal — 4.5 is set to keep `r ≥ 0` for ideal-coordination M=1 case, which would now correspond to `r = 0.5` for M=1 and saturating −4.0 for very-uncoordinated multi-UAV).
- M=2 lower bound under ADR-0008 is `−3.5`, strictly above `−4.0`, so M=2 sees zero behavioural change. The fix-floor M=2 1M checkpoint remains valid.
- Per-step penalty −4.0 still gives the agent a strong gradient signal toward coordination (4× the per-step penalty in M=1) without the destabilising 4× dominance over other objectives.

### Alternatives Considered (re-evaluated from the M=4 pilot data)
- **B. Multiplier rescale `* 4.0 / num_uavs`**: bounds magnitude perfectly (range stays [0.5, 4.5] regardless of M) but partially re-introduces ADR-0006's M-cancellation in *magnitude* — the gradient toward coordination shrinks proportionally. Not a clean fix.
- **D. 5M-step training**: 25h on the 3080, only 1 GPU available. The instability is in the reward shape, not training duration; more steps would amplify, not fix, the divergent attractor.
- **E. Curriculum from M=2 warm-start**: actor and critic shapes are M-dependent (action dim grows linearly), so warm-start needs an explicit weight-transfer protocol. Possible but heavy. Re-evaluate after seeing if the soft floor alone is enough.
- **C. Scale HV reference point with M**: pure measurement fix; would not have helped the M=4 pilot since the underlying policy is broken (AoSI 27×, energy 9×). Reject as standalone fix.

### Consequences
- M=2 fix-floor checkpoint (`pilot-M2-fixfloor-seed1`) **remains valid** — its training curve never touched the `r_energy < −3.5` region, so the soft floor is dead code for M=2.
- M=4 fix-floor checkpoint (`pilot-M4-fixfloor-seed1`) **is invalidated** — re-run with `--exp_name pilot-M4-softfloor-seed1` (~5h on long-training server).
- Reward-design ablation in the paper now has **four** data points instead of three:
  1. buggy-original (max_energy × M, floor 0.5) — M-cancellation
  2. ADR-0006 only (no × M, floor 0.5) — dead gradient
  3. ADR-0008 only (no × M, no floor) — divergent at M ≥ 4
  4. ADR-0009 (no × M, floor at −4) — current; expect stable across M
- Theorem 1's `r_max(M)` bound becomes `max(4.5, 4.0)` for M ≥ 3 (i.e., 4.5) — strictly tighter than the post-ADR-0008 form. Update `paper/theorem1.tex` once PR #28 lands so the bound matches the production form.
- All M=4 / M=5 baseline runs in the journal experiments matrix (DESIGN-baselines.md §2.4) must use post-ADR-0009 code.

See `environments/uav_semcom_multi_env.py:337-356` for the change and the M=4 diagnostic table above for the failure data that motivated the fix.

---
