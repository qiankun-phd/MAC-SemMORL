# Design Document: Constrained MORL Formulation

**Status**: Draft — pending reviewer approval
**Date**: 2026-05-06
**Author**: @claude (drafted from PLAN.md §C.2, ROADMAP.md P1.4, Issue #6)
**Decision deadline**: 2026-06-01 (constraint-handling default — see §4)
**Roadmap item**: P1.4 → enables the journal "constrained Pareto" novelty

---

## 0. Purpose and Scope

This document locks the formulation, algorithm, and code-change list for the Constrained MORL upgrade (Task C.2) before Phase 1.4 coding starts. It is the authoritative reference for:

- The three hard constraints to be enforced.
- The three candidate constraint-handling schemes and the criteria for choosing between them.
- The exact Lagrangian dual update procedure used in the training loop.
- The phased PR plan that breaks Issue #6 into reviewable chunks.
- A pre-mortem on the most likely failure modes.

Companion documents: `docs/PLAN.md` §C.2, `docs/ROADMAP.md` P1.4, `docs/DESIGN-multi-uav.md` (state/action/reward layout this builds on).

---

## 1. The Three Hard Constraints

The journal extension upgrades the conference 4-objective unconstrained MOMDP

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{T-1} \mathbf{w}^\top \mathbf{r}(s_t, a_t)\right]$$

into a *constrained* Pareto problem

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{T-1} \mathbf{w}^\top \mathbf{r}(s_t, a_t)\right]
\quad \text{s.t.} \quad \mathbf{c}_i(\pi) \le 0,\ i \in \{1,2,3\}.$$

The three constraints are picked to map directly onto operator-side SLA requirements that semantic communication papers typically waive away.

### 1.1 Reliability (per-device AoSI tail)

$$c_1: \quad \Pr_{t}\!\left[\max_k A_k(t) > A_{\max}\right] \le \varepsilon.$$

- $A_k(t)$ is the AoSI for device $k$ at slot $t$ (the same $A_k$ used in $r_2$).
- $A_{\max}$: hard tail bound. Default $A_{\max} = 3.0$ (≈ 3× the conference mean of 1.06).
- $\varepsilon$: tail-violation budget. Default $\varepsilon = 0.05$ (5% of slots may exceed).
- **Operational meaning**: even when objectives trade against AoSI, no single device may go silent for too long with high probability — concretely, the worst-case device's age stays within an SLA tail.

### 1.2 Per-episode fleet energy budget

$$c_2: \quad \sum_{t=0}^{T-1} E_{\text{fleet}}(t) \le E_{\text{total}}.$$

- $E_{\text{fleet}}(t) = \sum_{m=1}^M E_m(t)$ — the same fleet sum that drives $r_3$ after the ADR-0006 fix.
- $E_{\text{total}}$: per-episode energy budget. Default $E_{\text{total}} = 30\,\text{kJ}$ for $M=1$ (≈36% above conference mean of 22 kJ), and $E_{\text{total}} = M \cdot 25\,\text{kJ}$ for $M \ge 2$ (linear in $M$, but tighter than uncoordinated 2× scaling — encourages load balancing).
- **Operational meaning**: a UAV swarm in the field has a battery cap. The training agent must not produce policies that only *win on energy* by burning more of it.

### 1.3 Minimum service rate (per-slot worst device)

$$c_3: \quad \min_{k} \rho_{\text{svc},k}(t) \ge \rho_{\min} \quad \forall t.$$

- $\rho_{\text{svc},k}(t)$: average service indicator for device $k$ in a sliding window of $W$ slots (default $W = 20$).
- $\rho_{\min}$: minimum coverage. Default $\rho_{\min} = 0.7$ (each device must be served at least 70% of slots within any 20-slot window).
- **Operational meaning**: prevents pathological policies that systematically abandon low-priority devices. Soft fairness ($r_4$, Jain) is a population-level objective; this constraint is a *minimum coverage floor*.

### 1.4 Why these three (not five, not one)

| Considered | Status | Reason |
|------------|--------|--------|
| AoSI tail | Adopted | Reliability SLA is the strongest operator ask; tail violations are what newspapers report |
| Energy budget | Adopted | Battery is the inescapable physical limit; ADR-0006 made fleet energy meaningful |
| Min service rate | Adopted | Counters the "starve low-priority device" failure mode of weighted-sum MORL |
| Spectral mask | Rejected | Already implicit in `total_bandwidth` allocation |
| Collision | Rejected | Already a soft penalty in `r_3` (`d_min` in `MultiUAVSemComEnv`) |
| QoS jitter | Rejected | Tail bound on AoSI subsumes this for journal scope |

Three constraints is the minimum count that produces a non-trivial constrained-Pareto frontier (one constraint reduces to a Lagrangian-penalised single-objective problem, which is uninteresting).

---

## 2. Three Candidate Schemes

Issue #6 acceptance criterion calls for comparing three constraint-handling schemes on a pilot config. The comparison is itself the journal experimental contribution.

### 2.1 Lagrangian dual (primary, default-on)

Define the Lagrangian
$$\mathcal{L}(\pi, \boldsymbol{\lambda}) = \mathbb{E}\!\left[\sum_t \mathbf{w}^\top \mathbf{r}(s_t,a_t) - \sum_i \lambda_i\,c_i(\pi)\right],\quad \lambda_i \ge 0.$$

- **Primal step**: train the policy on a *shaped reward* $\tilde{r}(s,a) = \mathbf{w}^\top \mathbf{r}(s,a) - \boldsymbol{\lambda}^\top \mathbf{c}(s,a)$.
- **Dual step**: $\lambda_i \leftarrow \max(0, \lambda_i + \alpha_\lambda \cdot \hat{c}_i)$, where $\hat{c}_i$ is an exponential moving average of recent constraint values.
- **Pros**: simple to implement, scales to multi-objective natively, well-studied (PPO-Lagrangian, C-MORL, RCPO).
- **Cons**: dual variables can oscillate; $\alpha_\lambda$ tuning matters; converges to a *saddle point*, not a strict minimum.

### 2.2 Logarithmic barrier (alternative, ablation)

Add a penalty $-\sum_i \mu \log(-c_i)$ to the reward when $c_i < 0$ (constraint slack).

- **Pros**: enforces *strict* feasibility throughout training; no dual oscillation.
- **Cons**: needs an interior-feasible initialisation (hard to guarantee for AoSI-tail); $\mu$ schedule must shrink over training; numerical instability when $c_i \to 0^-$.

### 2.3 Action-space projection (alternative, ablation)

After the policy emits action $a$, project it onto the feasible set $\{a : \mathbf{c}(s, a) \le 0\}$.

- **Pros**: guaranteed per-step feasibility; deterministic.
- **Cons**: projection itself may be non-trivial (the AoSI-tail constraint depends on full state trajectory, not just $(s, a)$); breaks gradient flow; only natural for "simple-shape" constraints (e.g. energy can be projected by capping power).

### 2.4 Default selection — Lagrangian

Take the Lagrangian path as the default-on configuration. Barrier and projection ship as `--constraint_handler {lagrangian,barrier,projection}` flags for the comparison study (acceptance criterion 4 of Issue #6).

Justification:
- Lagrangian is the only one of the three that handles all three constraints uniformly (AoSI-tail is trajectory-level, not action-level, which projection cannot enforce).
- It composes cleanly with the existing COR-augmented Bellman backup (Theorem 1 in `paper/theorem1.tex`) — the shaped reward is just a different inner product.
- C-MORL (NeurIPS 2025), our positioning anchor, also uses Lagrangian dual.

---

## 3. Lagrangian Update Algorithm

### 3.1 Notation

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $\boldsymbol{\lambda} \in \mathbb{R}^3_{\ge 0}$ | Dual variables, one per constraint | init $0$ |
| $\alpha_\lambda$ | Dual learning rate | $10^{-3}$ |
| $\beta$ | EMA decay for $\hat{c}_i$ | $0.95$ |
| $N_{\text{dual}}$ | Dual-update cadence (env steps) | $1000$ |
| $\lambda_{\max}$ | Dual cap (prevents runaway) | $100.0$ |

### 3.2 Pseudocode

```
# At each env step inside agent.evluate():
c_step = [
    max(0.0, max_aosi_step - A_max),                     # c_1 violation magnitude
    max(0.0, info["energy"] - E_total / max_episode_steps),  # c_2 violation magnitude
    max(0.0, rho_min - service_rate_window),                  # c_3 violation magnitude
]
ema_c = beta * ema_c + (1 - beta) * c_step                # exponential moving avg

# Reward shaping at append-to-buffer time:
shaped_reward = reward - lambdas @ c_step                 # reward is 4-D, lambdas is 3-D
                                                          # c_step is 3-D, broadcast to 4-D pref
                                                          # by uniform spread: (lambda · c) / 4
                                                          # is subtracted from each reward component

# Dual update every N_dual steps:
if self.steps % N_dual == 0:
    lambdas = clip(lambdas + alpha_lambda * ema_c, 0.0, lambda_max)
    log_to_wandb(lambdas, ema_c)
```

### 3.3 Subtle points

1. **Reward is 4-dimensional, $\lambda$ is 3-dimensional.** The shaped reward must remain 4-D so the COR loss and Q-network shapes stay unchanged. Convention: subtract $\lambda^\top \mathbf{c} / 4$ from every component (uniform attribution). Justified because constraints are *not* per-objective; they're cross-cutting.

2. **EMA, not raw violation.** Dual gradients on raw $c_i$ values are extremely noisy (a single bad slot dominates). EMA smooths this with negligible bias for stationary policies and self-corrects for non-stationary ones.

3. **Cap $\lambda_{\max}$.** Without a cap, an early policy that violates constraints heavily can drive $\lambda_i \to \infty$, after which the primal step ignores the original objective entirely. $\lambda_{\max} = 100$ is safe given the per-step reward magnitude of $\sim 1$.

4. **AoSI-tail constraint $c_1$ is *probabilistic*.** The "max AoSI exceeds $A_{\max}$" indicator is binary per slot; the EMA over slots gives an unbiased estimate of the violation rate $\Pr[\cdot]$. So at convergence $\hat{c}_1 = $ violation rate, and the dual update drives this towards $\varepsilon$ (since $c_1 = $ rate $- \varepsilon$ — see the rewritten form below).

   To be precise, rewrite $c_1$ in the algorithm as
   $$\tilde{c}_1(s,a) = \mathbb{1}\!\left[\max_k A_k > A_{\max}\right] - \varepsilon,$$
   so the EMA target is the violation *rate* and the dual converges to enforce it.

---

## 4. CLI / Config / API Surface

Single source of truth: `main_uav.py` adds the following flags. All default values keep `use_lagrangian=False`, so existing pilots remain bit-identical.

```python
# --- Constrained MORL (Issue #6) ---
parser.add_argument("--use_lagrangian", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--constraint_handler", type=str, default="lagrangian",
                    choices=["lagrangian", "barrier", "projection"])
parser.add_argument("--A_max", type=float, default=3.0)
parser.add_argument("--epsilon_aosi", type=float, default=0.05)
parser.add_argument("--E_total_kJ", type=float, default=30.0,
                    help="Per-episode fleet energy budget in kJ. Scaled by num_uavs internally for M>=2.")
parser.add_argument("--rho_min", type=float, default=0.7)
parser.add_argument("--service_window", type=int, default=20)
parser.add_argument("--lambda_lr", type=float, default=1e-3)
parser.add_argument("--lambda_max", type=float, default=100.0)
parser.add_argument("--lambda_init", type=float, nargs=3, default=[0.0, 0.0, 0.0])
parser.add_argument("--dual_update_every", type=int, default=1000)
parser.add_argument("--ema_decay", type=float, default=0.95)
```

### Config dict additions

These are passed through to `SacAgent` / `MultiAgentSemMORL` via the existing `configs` dict:

```python
"use_lagrangian": args.use_lagrangian,
"constraint_handler": args.constraint_handler,
"constraint_thresholds": dict(A_max=..., epsilon_aosi=..., E_total_kJ=...,
                              rho_min=..., service_window=...),
"lambda_lr": args.lambda_lr,
"lambda_max": args.lambda_max,
"lambda_init": args.lambda_init,
"dual_update_every": args.dual_update_every,
"ema_decay": args.ema_decay,
```

### Agent attribute additions (`agent.py`)

```python
self.use_lagrangian: bool
self.constraint_handler: str
self.lambdas: torch.Tensor       # shape (3,), buffer
self.ema_costs: torch.Tensor     # shape (3,), buffer
self.svc_window: deque[int]      # length service_window, per-device service indicators
self.A_max, self.epsilon_aosi, self.E_total_per_step, self.rho_min: float
```

### Touch list

| File | Change | New lines (est) |
|------|--------|-----------------|
| `main_uav.py` | 11 new CLI flags + config plumbing | ~30 |
| `agent.py` | Lagrangian state, cost computation, reward shaping in `evluate`, dual update in training loop, wandb logging | ~150 |
| `multi_agent.py` | Inherit Lagrangian state; ensure cost computation uses fleet sum | ~30 |
| `environments/uav_semcom_env.py` | Expose `max_aosi` in `info` dict (currently only `mean_aosi`) | ~5 |
| `environments/uav_semcom_multi_env.py` | Same | ~5 |
| `docs/DECISIONS.md` | ADR-0007 documenting handler choice | ~50 |
| `tests/test_lagrangian_smoke.py` | Smoke test: run 5K steps with `--use_lagrangian`, verify no NaN | ~80 |

Total: ~350 LOC, comfortably within the 4-6 week C.2 budget.

---

## 5. Phased PR Plan

Issue #6 ships as four sequential PRs.

| PR | Branch | Scope | Time | Gates |
|----|--------|-------|------|-------|
| **A (this PR)** | `claude/issue-6-design-constrained` | This design doc | day 1 | reviewer approval before any code |
| **B** | `claude/issue-6-lagrangian-impl` | Lagrangian only: §3.2 algorithm + CLI flags + smoke test + ADR-0007 | week 1-2 | smoke test passes; default-off doesn't change baseline pilots |
| **C** | `claude/issue-6-barrier-projection` | Barrier and projection variants behind the `--constraint_handler` flag | week 3-4 | each handler runs without crash; documented limitations match §2.2/§2.3 |
| **D** | `claude/issue-6-comparison-study` | Run all three handlers on M=2 pilot config, produce constraint-violation-vs-HV figure for paper | week 5-6 | violation rate ≤ ε at convergence; HV degradation ≤ 10% (acceptance criteria of Issue #6) |

Each PR is mergeable in isolation; PR-D depends on PR-B + PR-C completing. PR-A (this) does not block PR-B implementation in a side branch — but the implementation reviewer should reference this doc.

---

## 6. Pre-Mortem

Three failure modes most likely to derail the implementation.

### 6.1 Dual variables stuck at zero (false-feasibility)

**Symptom**: $\lambda_i$ stays at 0 throughout training; constraints look "satisfied" but only because the policy never tries to violate them — and never tries to optimise toward the boundary either, leaving objective value on the table.

**Cause**: warm-up phase produces a conservative policy with low constraint violation, EMA stays low, dual update doesn't grow $\lambda$, primal has no penalty signal.

**Mitigation**:
- Initialise $\lambda_i = 0.1$ (small, non-zero) with `--lambda_init 0.1 0.1 0.1`.
- Skip dual updates during the first `start_steps = 10000` env steps (already the warmup).
- Log $\hat{c}_i$ to wandb so reviewer can see if violations are actually happening.

### 6.2 Dual variables runaway (constraints dominate)

**Symptom**: $\lambda_i \to \lambda_{\max}$; objectives hover near random-policy values; HV collapses.

**Cause**: a constraint is unsatisfiable under the current MOMDP, or $\alpha_\lambda$ is too high, or the cost magnitude is much larger than the reward magnitude.

**Mitigation**:
- Hard cap at `lambda_max = 100`.
- Sanity check at design time: estimate $\sup_\pi \hat{c}_i$ for an unconstrained policy; if any constraint is unsatisfiable (e.g., $E_{\text{total}}$ set lower than the energy needed to keep UAVs hovering for $T$ slots), abort and re-tune thresholds.
- Pre-pilot: 100K-step run at `--use_lagrangian` and `--lambda_lr 0` (frozen duals) to record what the natural $\hat{c}_i$ values are.

### 6.3 Reward-shape leakage breaks the COR Bellman analysis

**Symptom**: Theorem 1's contraction modulus $\gamma + 2\beta$ no longer holds because $\tilde{r}$ is not the same as $\mathbf{w}^\top \mathbf{r}$.

**Cause**: the COR analysis treats the reward as fixed; injecting $-\boldsymbol{\lambda}^\top \mathbf{c}$ adds a non-stationary term.

**Mitigation**:
- Treat the dual update as *outer-loop*: $\lambda$ is held fixed across primal updates, so within a primal phase $\tilde{r}$ is stationary and Theorem 1 applies as-is.
- Add a one-paragraph extension to `paper/theorem1.tex` noting that the contraction holds under the augmented reward $\tilde{r}$ for fixed $\lambda$, and the outer dual update is a separate gradient ascent that converges by standard Lagrangian theory (Bertsekas Ch. 6).
- This extension is in PR-D, not PR-B — implementation can proceed without it as long as the design captures the issue.

---

## 7. Open Questions for Reviewer

Resolve before merging this PR.

1. **Default thresholds**: are $A_{\max} = 3.0$, $\varepsilon = 0.05$, $E_{\text{total}} = 30$ kJ, $\rho_{\min} = 0.7$ the right starting points? They're 30-50% looser than conference observed values, which is the C-MORL convention but on the loose side for "hard SLA".

2. **Multi-UAV scaling of $E_{\text{total}}$**: should the budget scale linearly in $M$ (current proposal: $M \cdot 25$ kJ for $M \ge 2$) or stay constant (force tighter coordination at larger $M$)? Linear is the operator-realistic choice; constant is the more challenging research story.

3. **Dual update cadence**: $N_{\text{dual}} = 1000$ env steps means ~5 updates per episode. Is that too frequent (oscillation risk) or too rare (slow convergence)? C-MORL uses one update per episode; PPO-Lagrangian uses one per rollout.

4. **Constraint vs reward conflict**: $c_3$ (min service rate) has overlap with $r_4$ (Jain fairness). Reviewer may want to drop $c_3$ to avoid double-incentivising fairness. Counter-argument: Jain is *population-level* and rewards balance, while $c_3$ is a *floor* and prevents systematic exclusion of any single device.

---

## 8. Acceptance for This Design PR

This document is approved for implementation when:

- [ ] Reviewer confirms the three constraints are the right set (§1.4).
- [ ] Reviewer agrees Lagrangian is the default and barrier/projection are ablations (§2.4).
- [ ] Reviewer approves default thresholds (§7 Q1) or proposes alternates.
- [ ] No objections to the four-PR phasing (§5).

After approval, PR-B (Lagrangian implementation) opens against `main`.
