# Design Document: Multi-UAV Environment Refactor

**Status**: Draft — pending reviewer approval  
**Date**: 2026-05-02  
**Author**: @copilot (auto-generated from SKETCHES.md §1, PLAN.md §C.1, ROADMAP.md P0.5)  
**Decision deadline**: 2026-06-15 (CTDE vs Federated Latent default — see §2)  
**Roadmap item**: P0.5 → enables P1.1, P1.2, P1.3  

---

## 0. Purpose and Scope

This document locks the API and data flow for the multi-UAV environment refactor (Task C.1) before Phase 1 coding starts. It is the authoritative reference for:

- The upgraded MOMDP formulation (state, action, reward).
- The two coordination architectures and the criteria for choosing between them.
- The hard constraints on scheduling and safety.
- The exact code-change list that engineers will implement in Phase 1.
- A pre-mortem identifying the three most likely failure modes.

Companion documents: `docs/SKETCHES.md` §1 (first-cut formulation), `docs/PLAN.md` §C.1 (workload), `docs/ROADMAP.md` Phase 1.

---

## 1. State / Action / Reward Upgrades

### 1.1 Notation Upgrade (Single-UAV → Multi-UAV)

| Symbol | Conference (single UAV) | Journal (multi-UAV, $M$ UAVs) |
|--------|--------------------------|-------------------------------|
| Number of UAVs | 1 | $M$, indexed $m \in \{1,\dots,M\}$ |
| Position | $\mathbf{q}(t) \in \mathbb{R}^3$ | $\mathbf{q}_m(t) \in \mathbb{R}^3$ |
| Velocity | $\mathbf{v}(t) \in \mathbb{R}^3$ | $\mathbf{v}_m(t) \in \mathbb{R}^3$ |
| Acceleration action | $\boldsymbol{\nu}(t)$ | $\boldsymbol{\nu}_m(t)$ |
| Transmit power | $p(t)$ | $p_m(t)$ |
| Compression ratios | $\boldsymbol{\eta}(t)$ | $\boldsymbol{\eta}_m(t)$ |
| Service binary | implicit | $x_{m,k}(t) \in \{0,1\}$ — UAV $m$ serves device $k$ at slot $t$ |
| Scheduling constraint | implicit | $\sum_{m=1}^M x_{m,k}(t) \leq 1$ for all $k$ (no double-service) |
| Energy per slot | $E(t)$ | $E_m(t)$ |

Typical experimental settings: $M \in \{2, 4, 5\}$, $K \in \{5, 10, 20\}$.

---

### 1.2 Joint MOMDP Formulation

The multi-UAV system is formulated as a cooperative multi-agent MOMDP:

$$\mathcal{M} = \big(\mathcal{S},\; \mathcal{A},\; P,\; \mathbf{r},\; \gamma\big)$$

#### 1.2.1 State Space

The joint state at time $t$ is:

$$\mathbf{s}(t) = \Big[\underbrace{\{\mathbf{q}_m, \mathbf{v}_m\}_{m=1}^M}_{\text{UAV kinematics}},\; \underbrace{\{\Delta\mathbf{q}_{m,k},\, A_k,\, g_{m,k}\}_{m,k}}_{\text{UAV–device geometry + channel}},\; \underbrace{\{\tau_k\}_{k=1}^K}_{\text{AoSI per device}},\; \underbrace{t/T}_{\text{normalised time}}\Big]$$

**Dimension**: $|\mathbf{s}| = 6M + 3MK + 2K + 1$.

| Block | Description | Dimension |
|-------|-------------|-----------|
| $\{\mathbf{q}_m, \mathbf{v}_m\}$ | 3-D position + velocity per UAV | $6M$ |
| $\Delta\mathbf{q}_{m,k}$ | horizontal offset from UAV $m$ to device $k$ | $2MK$ |
| $A_k$ | semantic traffic load / data size | $K$ |
| $g_{m,k}$ | channel gain UAV $m$ → device $k$ | $MK$ |
| $\tau_k$ | age-of-semantic-information at device $k$ | $K$ |
| $t/T$ | episode progress | $1$ |

**Partial observability**: In the decentralized execution phase each UAV $m$ observes only its own kinematic state, the $K$ channels from its own transmitters, and a summary vector $\bar{\mathbf{h}}_{-m}$ from neighbouring UAVs (dimension $M-1$ per neighbour summary). The joint state is used only by the centralized critic during training.

#### 1.2.2 Action Space

The joint action at slot $t$ is the concatenation of per-UAV actions:

$$\mathbf{a}(t) = \big[\{\boldsymbol{\nu}_m,\, p_m,\, \boldsymbol{\eta}_m,\, \mathbf{x}_m\}_{m=1}^M\big]$$

| Component | Type | Dimension per UAV | Notes |
|-----------|------|-------------------|-------|
| $\boldsymbol{\nu}_m$ | Continuous | 3 | 3-D acceleration (clipped to $[\nu_{\min}, \nu_{\max}]$) |
| $p_m$ | Continuous | 1 | Transmit power (clipped to $[0, P_{\max}]$) |
| $\boldsymbol{\eta}_m$ | Continuous | $K$ | Per-device compression ratio $\in [0,1]$ |
| $\mathbf{x}_m$ | Binary / Soft | $K$ | Scheduling vector; $x_{m,k} \in \{0,1\}$ |

Per-UAV action dimension: $4 + K$.  Total joint action dimension: $M(4 + K)$.

**Scheduling enforcement**: The no-double-service constraint $\sum_m x_{m,k} \leq 1$ is enforced by applying a device-level softmax over UAVs in the actor output head (see §3.1). During training, hard constraint violations are penalised (see §3.2).

#### 1.2.3 Reward Vector

The reward remains 4-dimensional $\mathbf{r}(t) = [r_1, r_2, r_3, r_4]^\top$ but each component is now aggregated across all UAVs and devices:

| Index | Name | Single-UAV (conference) | Multi-UAV (journal) |
|-------|------|--------------------------|----------------------|
| $r_1$ | Semantic fidelity | $\sum_k x_k S_k$ | $\sum_m \sum_k x_{m,k} S_k$ — weighted sum over served $(m,k)$ pairs |
| $r_2$ | AoSI reduction | $-\sum_k \tau_k$ | $-\sum_k \tau_k$ — each device's AoSI evolves at most once regardless of how many UAVs are in range (no double-service) |
| $r_3$ | Energy efficiency | $-E(t)$ | $-\sum_m E_m(t)$ — sum of all UAV energy expenditure |
| $r_4$ | Fairness (Jain index) | Jain over device fidelity | Jain index over device-level fidelity $\{S_k\}_k$ (unchanged formula, extended device set) |

A scalar reward for scalarization (used in MORL preference weighting) is:

$$r_{\text{scalar}}(t) = \boldsymbol{\lambda}^\top \mathbf{r}(t), \quad \boldsymbol{\lambda} \in \Delta^3$$

---

## 2. Coordination Architecture

Two architectural options are documented here. The default for Phase 1 is **Option A (CTDE)**. Option B (Federated Latent) is the fallback if Option A fails the scalability gate at $M \geq 4$.

### 2.1 Option A — Centralized Training, Decentralized Execution (CTDE)

**Architecture overview**:

```
Training time:
  Joint state s(t) → Shared OADM encoder f_φ → latent z(t)
                                                    │
                          ┌─────────────────────────┤
                          │              (Centralized critic)
                          ▼                V_ψ(z, λ)
  Scalarized returns ─────┘

  Per-UAV actors π_θ_m:
    local state s_m(t) + z(t) → action a_m(t)

Execution time (deployed):
  Each UAV m runs π_θ_m with its own local state only.
  (Shared encoder weights broadcast once; critic discarded.)
```

**Formal description**:

- All $M$ actors $\{\pi_{\theta_m}\}$ and critics $\{V_{\psi_m}\}$ share a single OADM encoder $f_\phi$.
- During training, the centralized critic $V_\psi$ receives the full joint state $\mathbf{s}(t)$ via the shared encoder.
- At deployment, each UAV $m$ uses its local actor with its own observation $\mathbf{o}_m(t) \subset \mathbf{s}(t)$.
- Parameter update: gradient flows through the shared encoder during centralized critic updates.

**Pros**:
- Simple extension of existing SemMORL single-agent architecture.
- Leverages existing OADM encoder without architectural change.
- Well-understood convergence properties (CTDE is MARL standard).

**Cons**:
- Centralized critic input grows as $O(MK)$ — potential bottleneck at $M = 5, K = 20$.
- Joint state requires synchronization of all UAV observations each step during training.

**Scalability gate** (Decision D1, deadline 2026-06-15):  
Let $T_1$ denote the measured wall-clock time for one full training run at the single-UAV ($M=1$) conference baseline setting. If the M=2 CTDE pilot shows > 3× $T_1$ or > 30% increase in reward variance across seeds compared to M=1, switch to Option B.

### 2.2 Option B — Federated Latent

**Architecture overview**:

```
Each UAV m has its own actor-critic pair (π_θ_m, V_ψ_m).
Each UAV m has its own OADM encoder f_φ_m.

Every K_sync steps:
  φ_avg = (1/M) Σ_m φ_m          (FedAvg on encoder weights)
  All f_φ_m ← f_φ_avg             (broadcast)

Local updates between syncs: standard SAC/PPO with local observations.
```

**Formal description**:

- Each UAV $m$ maintains its own actor-critic $(π_{\theta_m}, V_{\psi_m})$ and encoder $f_{\phi_m}$.
- Encoder weights are synchronized via FedAvg every $K_{\text{sync}}$ environment steps (tunable; default $K_{\text{sync}} = 100$).
- Actor and critic weights are **not** shared (each UAV learns a personalized policy).
- No centralized critic — each UAV bootstraps from its own local value function.

**Pros**:
- Per-UAV compute and memory scale linearly in $M$ (no joint-state centralized critic).
- Communication overhead is $O(|\phi|)$ per sync step, independent of $K$.
- Enables privacy-preserving deployment (no raw observations shared).

**Cons**:
- Synchronization heuristic ($K_{\text{sync}}$) requires tuning; convergence guarantee is non-trivial.
- Without centralized critic, coordination relies solely on the shared encoder — risk of locally-optimal decentralized policies.

**Trigger for activation**: Option B becomes the default if the CTDE scalability gate fails at $M \geq 4$ (see Decision D1). Budget +2 weeks for engineering the sync infrastructure.

### 2.3 Decision Template — CTDE vs Federated Latent

| Criterion | CTDE (Option A) | Federated Latent (Option B) |
|-----------|----------------|------------------------------|
| Wall-clock at M=4 | Acceptable if < 6× $T_1$ (single-UAV baseline) | Always better |
| Reward variance | Acceptable if Δ ≤ 30% vs M=1 | May be worse (no global critic) |
| Engineering risk | Low (reuses existing arch) | Medium (+sync infra) |
| TWC novelty claim | Standard CTDE — cite prior work | Novel federated latent — new contribution |
| Fallback cost | N/A | +2 engineering weeks |

**Recommended default**: Option A (CTDE) through Phase 1.  
**Decision deadline**: 2026-06-15. Record outcome in `docs/DECISIONS.md` as ADR-0003 addendum.

---

## 3. Constraints

### 3.1 No-Double-Service Constraint

**Constraint**: Each IoT device $k$ may be served by at most one UAV per time slot:

$$\sum_{m=1}^{M} x_{m,k}(t) \leq 1 \quad \forall k \in \{1,\dots,K\},\; \forall t$$

**Enforcement mechanism** (actor output head):

```
For each device k:
  raw_logits[m, k]  ← linear layer output for UAV m, device k
  x[:, k] = softmax(raw_logits[:, k])  over UAVs m=1..M
```

This produces a probability distribution over which UAV (or no UAV) serves device $k$. At execution, argmax is taken for the hard binary assignment. During training, the soft probability is used directly (Gumbel-softmax for gradient flow through discrete choice).

**Alternative** (soft penalty): Add constraint violation penalty $\lambda_{\text{sched}} \cdot \max(0, \sum_m x_{m,k} - 1)^2$ to reward. Use only as a debugging fallback if the softmax head causes gradient instability.

### 3.2 Collision Avoidance Constraint

**Constraint**: All pairs of UAVs must maintain a minimum separation distance $d_{\min}$:

$$\|\mathbf{q}_m(t) - \mathbf{q}_{m'}(t)\| \geq d_{\min} \quad \forall m \neq m',\; \forall t$$

**Default enforcement** (soft penalty in reward):

$$r_{\text{collision}}(t) = -w_c \sum_{m \neq m'} \max\!\big(0,\; d_{\min} - \|\mathbf{q}_m - \mathbf{q}_{m'}\|\big)^2$$

where $w_c$ is a tunable penalty weight (default $w_c = 1.0$). This penalty is added to $r_3$ (energy/cost component) or treated as a fifth auxiliary reward component (not included in the preference vector).

**Hard barrier alternative**: Implement a projection step in `step()` that clips the next UAV position to maintain $d_{\min}$. More conservative but easier to guarantee safety. Decision on which to use deferred to Phase 1 engineering (milestone P1.1).

**Recommended**: Soft penalty for initial experiments (simpler, differentiable). Switch to hard barrier if collision events > 5% of timesteps in M=2 pilot.

### 3.3 Inter-UAV Interference (Optional)

If UAVs operate on the same frequency sub-band simultaneously, device $k$ receives interference from neighbouring UAV transmissions. The received SINR at device $k$ from UAV $m$ is:

$$\text{SINR}_{m,k}(t) = \frac{p_m g_{m,k}}{\sigma^2 + \sum_{m' \neq m} p_{m'} g_{m',k} \cdot \mathbf{1}[x_{m',k'}=1, k' \text{ nearby}]}$$

**Implementation decision**: Model interference as optional environment flag `use_interference: bool` (default `False`). Enable only for the full-journal ablation (Phase 2 experiments). Not required for Phase 1.

---

## 4. Detailed Code-Change List

The following changes are required to implement the multi-UAV environment. They are ordered by dependency (earlier items must be completed before later ones).

### 4.1 `environments/uav_semcom_env.py`

Priority: **P1** (blocking all others).

| Change | Description |
|--------|-------------|
| Constructor: add `num_uavs: int = 1` parameter | Accept $M > 1$; allocate per-UAV state/action buffers |
| `reset()` | Initialize positions $\{\mathbf{q}_m(0)\}$ with minimum separation $\geq d_{\min}$; return joint state |
| `step(actions)` | Accept `actions` as list of $M$ per-UAV action arrays; broadcast device states; apply no-double-service softmax; compute joint reward vector $\mathbf{r}(t)$; apply collision penalty |
| `_compute_reward()` | Aggregate $r_1$–$r_4$ across UAVs per §1.2.3; add collision penalty term |
| `_check_constraints()` | Return dict of constraint violations (collision, double-service) for logging |
| `observation_space` | Return per-UAV observation space (partial obs) AND joint observation space |
| `action_space` | Return joint action space of dimension $M(4+K)$ |
| `_get_obs(m)` | Return partial observation for UAV $m$: own kinematics + assigned device channels + neighbour summary |

**Backward compatibility**: when `num_uavs=1`, the new `step()` must behave identically to the old single-UAV interface (pass `actions=[action]` or `actions=action`).

### 4.2 `agent.py`

Priority: **P1** (after env refactor).

| Change | Description |
|--------|-------------|
| Add `MultiAgentSemMORL` class | Wraps $M$ `SemMORL` actor-critic instances with a single shared OADM encoder $f_\phi$ |
| `MultiAgentSemMORL.__init__()` | Accept `num_uavs`, `obs_dim`, `action_dim`, `shared_encoder: bool = True` |
| `MultiAgentSemMORL.act(obs_list, lam)` | Accept list of $M$ per-UAV observations; return list of $M$ actions |
| `MultiAgentSemMORL.update(batch)` | Run centralized critic update (CTDE Option A) using joint state; per-UAV actor updates; shared encoder gradient |
| Option B hook: `sync_encoders()` | FedAvg step — average $\phi_m$ across all UAVs; broadcast result |
| Export `SingleAgentSemMORL` alias | Preserve backward compatibility with existing single-UAV training scripts |

### 4.3 `train.py`

Priority: **P2** (after agent.py).

| Change | Description |
|--------|-------------|
| CLI: add `--num-uavs M` argument | Pass $M$ to environment and agent constructors |
| Inner loop: iterate over UAVs | Collect per-UAV observations; pass list to `MultiAgentSemMORL.act()` |
| Replay buffer | Store joint transitions $([\mathbf{o}_m], [\mathbf{a}_m], \mathbf{r}, [\mathbf{o}'_m])$ |
| Logging | Log per-UAV reward components, collision rate, double-service violations per episode |
| CTDE vs Federated switch | `--coordination ctde|federated` flag; call `sync_encoders()` every `K_sync` steps when `federated` |
| Outer loop | Unchanged (preference vector sampling and episode management remain as-is) |

### 4.4 `plot_results_line.py`

Priority: **P3** (after training).

| Change | Description |
|--------|-------------|
| Load per-UAV reward traces | Parse new `npz` keys `reward_m{m}` for $m \in \{1,\dots,M\}$ |
| Per-UAV Pareto plots | Add subplot per UAV or overlay with distinct line styles |
| Aggregate-per-method plots | Existing plots unchanged; add `M=2,4,5` comparison plot |
| Collision / constraint summary | New figure: collision rate and double-service violations vs training step |

### 4.5 Configuration / Hyperparameters

Add `configs/multi_uav_default.yaml` (new file):

```yaml
# Multi-UAV default configuration
environment:
  num_uavs: 2            # M
  num_devices: 10        # K
  d_min: 50.0            # minimum UAV separation (meters)
  use_interference: false
  collision_penalty_weight: 1.0   # w_c

agent:
  coordination: ctde      # ctde | federated
  shared_encoder: true
  k_sync: 100             # FedAvg sync period (federated only)

training:
  num_episodes: 5000
  preference_grid_size: 56
  seeds: [0, 1, 2, 3, 4, 5]
```

---

## 5. Pre-Mortem: Likely Failure Modes

Before Phase 1 coding begins, we identify the three most probable failure modes and mitigation strategies.

### 5.1 State Explosion

**Failure mode**: The joint state dimension $6M + 3MK + 2K + 1$ grows as $O(MK)$. At $M=5, K=20$ the joint state has dimension $6(5) + 3(5)(20) + 2(20) + 1 = 30 + 300 + 40 + 1 = 371$. The centralized critic input is 371-dimensional. Training may require significantly longer to converge, and the replay buffer memory footprint grows proportionally.

**Symptoms**: Training loss stalls; critic variance is high; sample efficiency worse than linear in $M$.

**Mitigations**:
1. **Attention-pooling over devices**: replace the device-specific block $\{\Delta\mathbf{q}_{m,k}, A_k, g_{m,k}\}$ with a cross-attention readout (UAV query, device key/value) — reduces effective per-UAV input to $O(d_{\text{attn}})$ regardless of $K$.
2. **Neighbourhood masking**: UAV $m$ only includes devices within its communication radius in its local state; devices outside are masked to zero.
3. **Reduce $K$ for pilot**: run $K=5$ first; if training converges, scale to $K=10, 20$.
4. **Smaller OADM latent dimension** $d_z$: reduce from default to $d_z/2$ for multi-UAV experiments if memory is the binding constraint.

**Trigger**: activate mitigation if critic loss does not decrease after 1000 episodes in the M=2, K=10 pilot.

### 5.2 Reward Sparsity

**Failure mode**: With $M$ UAVs competing for $K$ devices and the no-double-service constraint, many scheduling assignments will be "wasted" — UAV $m$ selects a device already served by UAV $m'$. The effective service rate per UAV drops, leading to sparse non-zero rewards per agent per step. Agents may learn trivially to always fly close to densely populated areas and ignore the fairness objective.

**Symptoms**: $r_4$ (Jain index) collapses to low values early in training and never recovers; per-UAV $r_1$ variance is very high across seeds; scheduling $\mathbf{x}_m$ degenerates (one UAV wins all devices).

**Mitigations**:
1. **Reward shaping with coverage bonus**: add a small bonus $r_{\text{cov}} = \sum_k \mathbf{1}[\exists m: x_{m,k}=1]$ to encourage spreading service across devices.
2. **Counterfactual reward**: each UAV $m$ receives its **marginal contribution** to the joint reward (i.e., $r(\mathbf{a}) - r(\mathbf{a}_{-m}, \bar{a}_m)$ where $\bar{a}_m$ is a default action) to reduce credit assignment noise.
3. **Curriculum scheduling**: start with $M=K$ (one UAV per device, trivially non-sparse) and gradually reduce $M$ or increase $K$ over training.
4. **Tighter preference sampling**: over-weight $\lambda_4$ (fairness) during early training to prevent collapse.

**Trigger**: activate mitigation if Jain index $r_4 < 0.5$ at episode 500 in the M=2, K=10 pilot.

### 5.3 Scheduling Deadlock

**Failure mode**: With strict no-double-service enforcement (Gumbel-softmax over UAVs) and greedy actor policies, the system may reach a cyclic assignment: UAV 1 always selects device set $\mathcal{K}_1$, UAV 2 always selects $\mathcal{K}_2$, and neither adapts to device mobility or traffic load changes. This is a deterministic policy trap where exploration collapses.

**Symptoms**: Entropy of the scheduling distribution $H(\mathbf{x}_m)$ falls to near zero after ~500 episodes; curriculum learning shows improvement initially but plateaus; trajectory visualization shows UAVs locked onto fixed spatial partitions.

**Mitigations**:
1. **Entropy regularization**: add $-\beta_H H(\pi_\theta)$ to the actor loss (standard SAC entropy bonus), specifically targeting the scheduling sub-head.
2. **Coordination penalty for static partitioning**: detect if inter-UAV assignment overlap index $\rho_{\text{overlap}} = \mathbb{E}[\sum_k x_{1,k} x_{2,k}]$ is consistently zero (never serving same device) — both over-service and under-service are failure modes. Add penalty for Gini coefficient of device-level service frequency deviating from uniform.
3. **Temperature annealing in Gumbel-softmax**: start with high temperature $\tau=2.0$ (uniform-like), anneal to $\tau=0.1$ over training to allow soft exploration early and crisp assignments later.
4. **Random preference injection**: during training, occasionally inject $\boldsymbol{\lambda} = \mathbf{e}_4$ (pure fairness preference) to force the agent to explore fair schedules.

**Trigger**: activate mitigation if scheduling entropy $H(\mathbf{x}_m) < 0.3$ bits at episode 300 in any seed.

---

## 6. Open Questions and Decisions Required

| ID | Question | Owner | Deadline | Notes |
|----|----------|-------|----------|-------|
| D1 | CTDE vs Federated Latent default for Phase 1 | Lead engineer + advisor | **2026-06-15** | Run M=2 CTDE pilot; evaluate wall-clock and reward variance against gate criteria in §2.3 |
| D2 | Soft penalty vs hard barrier for collision avoidance | Lead engineer | 2026-08-15 | Assess collision rate in M=2 pilot |
| D3 | Include interference model in Phase 1 experiments | Advisor | 2026-08-15 | Default: disable; enable only in ablation |
| D4 | Counterfactual reward vs joint reward for credit assignment | Lead engineer | 2026-08-15 | Decision driven by reward sparsity observation (§5.2) |
| D5 | Attention pooling for state representation | Lead engineer | 2026-09-01 | Implement only if state explosion triggers (§5.1) |

Record each decision as an ADR entry in `docs/DECISIONS.md` when resolved.

---

## 7. Acceptance Criteria

This design is considered complete and ready for Phase 1 implementation when:

- [x] `docs/DESIGN-multi-uav.md` merged to main.
- [ ] At least 1 reviewer (advisor / collaborator) has approved the MOMDP formulation in §1.2 and the constraint specifications in §3.
- [ ] Decision D1 (CTDE vs Federated Latent) resolved and recorded in `docs/DECISIONS.md` by 2026-06-15.
- [ ] Phase 1 milestone P1.1 backlog created in the project tracker with tasks corresponding to §4 code-change list.

---

## 8. References

- `docs/SKETCHES.md` §1 — original formulation sketch (MOMDP, coordination, constraints, code changes).
- `docs/PLAN.md` §C.1 — workload and time estimates for multi-UAV extension.
- `docs/ROADMAP.md` Phase 1, Decision Points D1 — CTDE vs Federated deadline.
- `docs/DECISIONS.md` ADR-0003 — initial CTDE-first decision; ADR-0004 — GPU reservation plan.
- SKETCHES.md §1.3 — technical detail on Option A and Option B architectures.
- IEEE TWC 2022, doi:10.1109/TWC.2022.3153316 — multi-agent DRL UAV-MEC (CTDE precedent).
- IEEE TWC 2023, doi:10.1109/TWC.2023.3268082 — multi-agent RL UAV swarm (CTDE at scale).
