# SemMORL Journal Extension — Implementation Sketches

**Date**: 2026-05-02
**Companion to**: `JOURNAL-EXTENSION-PLAN-2026-05-02.md`
**Purpose**: provide first-cut technical drafts for the three highest-leverage Tier-1 tasks so that engineering effort can begin before the formal journal write-up phase.

---

## 1. Multi-UAV MOMDP Formulation Sketch (Task C.1)

### 1.1 Notation Upgrade

| Symbol | Conference (single UAV) | Journal (multi-UAV) |
|--------|--------------------------|----------------------|
| Number of UAVs | 1 | $M$, indexed by $m \in \{1,\dots,M\}$ |
| Position | $\mathbf{q}(t)$ | $\mathbf{q}_m(t)$ |
| Velocity | $\mathbf{v}(t)$ | $\mathbf{v}_m(t)$ |
| Action | $\mathbf{a}(t)$ | $\mathbf{a}_m(t)$ for each UAV |
| Service binary | implicit | $x_{m,k}(t) \in \{0,1\}$ — UAV $m$ serves device $k$ at slot $t$, with $\sum_m x_{m,k}(t) \leq 1$ |

### 1.2 Joint MOMDP

$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \mathbf{r}, \gamma)$ where:

- **State**:
  $\mathbf{s}(t) = \big[\{\mathbf{q}_m, \mathbf{v}_m\}_{m=1}^M, \{\Delta\mathbf{q}_{m,k}, A_k, g_{m,k}\}, \{\tau_k\}, t/T\big] \in \mathbb{R}^{(2M+3MK)+K+1}$.

- **Action**:
  $\mathbf{a}(t) = \big[\{\boldsymbol{\nu}_m, \mathbf{p}_m, \boldsymbol{\eta}_m, \mathbf{x}_m\}_{m=1}^M\big]$ with per-UAV acceleration, transmit power, compression ratios, and binary scheduling vector $\mathbf{x}_m \in \{0,1\}^K$.

- **Reward**: same four components but aggregated across UAVs:
  - $r_1$: weighted sum of $S_k$ over served device-UAV pairs $(m,k)$ with $x_{m,k}=1$
  - $r_2$: per-device AoSI evolves once if served by *any* UAV
  - $r_3$: $-\sum_m E_m(t)$ with each UAV's energy
  - $r_4$: Jain index over device-level fidelity

### 1.3 Coordination Mechanism

Two architectural options to consider:

#### Option A — Centralized Training, Decentralized Execution (CTDE)

- All $M$ critics share an encoder $f_\phi$ during training
- At deployment each UAV uses local actor with its local state subset
- Pros: simple, leverages existing OADM
- Cons: scalability with $M$ (centralized critic state grows)

#### Option B — Federated Latent

- Each UAV has its own actor-critic
- OADM encoders synchronize via parameter averaging every $K_{\text{sync}}$ steps
- Pros: scales linearly in $M$, communication-efficient
- Cons: synchronization heuristic, non-trivial convergence

**Recommended**: Start with Option A for initial experiments; if scalability bottleneck appears, switch to Option B for the final paper.

### 1.4 Coverage / Handover Constraints

- **No double-service**: $\sum_m x_{m,k}(t) \leq 1$ enforced via softmax over UAVs in actor head.
- **Collision avoidance**: $\|\mathbf{q}_m(t) - \mathbf{q}_{m'}(t)\| \geq d_{\min}$ for all $m \neq m'$, added as soft penalty in reward or hard barrier.
- **Inter-UAV interference**: optionally model interference at device $k$ from other UAVs serving nearby devices on the same sub-band.

### 1.5 Code Changes (Approximate)

1. `environments/uav_semcom_env.py`: refactor `step()` to accept list of $M$ actions; broadcast device states; aggregate rewards.
2. `agent.py`: add `MultiAgentSemMORL` class wrapping $M$ actors with shared OADM encoder.
3. `train.py`: outer loop unchanged; inner loop iterates over UAVs.
4. `plot_results_line.py`: extend per-UAV traces, aggregate per-method.

**Effort**: ~4 weeks engineering + 2–4 weeks running experiments.

---

## 2. Theorem 1 — COR Contraction Proof Sketch (Task C.3)

### 2.1 Setup

Consider linear function approximation:
$$Q_\psi(\mathbf{s},\mathbf{a},\bm{\lambda}) = \psi^\top \phi(\mathbf{s},\mathbf{a},\bm{\lambda})$$

where $\phi: \mathcal{S}\times\mathcal{A}\times\Delta^3 \to \mathbb{R}^d$ is a fixed feature map with $\|\phi\| \leq B$.

The standard Bellman backup is
$$(\mathcal{T}Q)(\mathbf{s},\mathbf{a},\bm{\lambda}) = \mathbb{E}_{\mathbf{s}'}\!\big[\bm{\lambda}^\top \mathbf{r} + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}',\mathbf{a}',\bm{\lambda})\big].$$

The COR-augmented backup adds a smoothing term across two preferences $\bm{\lambda}_1, \bm{\lambda}_2$:
$$(\mathcal{T}_{\text{COR}}Q)(\cdot,\bm{\lambda}_1) = (\mathcal{T}Q)(\cdot,\bm{\lambda}_1) - \alpha\big[\bar{\rho} - \rho\big]_+ \big(Q(\cdot,\bm{\lambda}_1) - Q(\cdot,\bm{\lambda}_2)\big).$$

### 2.2 Theorem 1 (Statement)

> **Theorem 1.** Let $\beta := \alpha[\bar{\rho} - \rho]_+ \in [0, \alpha\bar{\rho}]$. If $\gamma + \beta < 1$, then $\mathcal{T}_{\text{COR}}$ is a contraction on the weighted $\ell_2$ space with modulus $\gamma + \beta$:
> $$\|\mathcal{T}_{\text{COR}}Q - \mathcal{T}_{\text{COR}}Q'\|_2 \leq (\gamma + \beta)\|Q - Q'\|_2.$$

### 2.3 Proof Sketch

For any two value functions $Q, Q'$:

1. **Standard Bellman part**:
   $\|\mathcal{T}Q - \mathcal{T}Q'\|_2 \leq \gamma \|Q - Q'\|_2$ (standard).

2. **COR penalty part**:
   $\|\beta(Q(\cdot,\bm{\lambda}_1) - Q(\cdot,\bm{\lambda}_2)) - \beta(Q'(\cdot,\bm{\lambda}_1) - Q'(\cdot,\bm{\lambda}_2))\|_2$
   $\leq \beta(\|Q(\cdot,\bm{\lambda}_1) - Q'(\cdot,\bm{\lambda}_1)\|_2 + \|Q(\cdot,\bm{\lambda}_2) - Q'(\cdot,\bm{\lambda}_2)\|_2)$
   $\leq 2\beta \|Q - Q'\|_2$.

3. **Triangle inequality**:
   $\|\mathcal{T}_{\text{COR}}Q - \mathcal{T}_{\text{COR}}Q'\|_2 \leq \gamma\|Q - Q'\|_2 + 2\beta\|Q-Q'\|_2$.

Take supremum over $\bm{\lambda}_1, \bm{\lambda}_2$ to obtain modulus $\gamma + 2\beta$. (If symmetrized over $\bm{\lambda}_1, \bm{\lambda}_2$ pairs, modulus tightens to $\gamma + \beta$.)

**Sufficient condition for contraction**: $\alpha \bar{\rho} < (1-\gamma)/2$. With our hyperparameters $\gamma = 0.995, \alpha = 0.5, \bar{\rho} = 0.25$, $\alpha\bar{\rho} = 0.125 < (1-\gamma)/2 = 0.0025$ — **wait this fails**.

> **Note**: with $\gamma = 0.995$ very close to 1, contraction modulus $\gamma + 2\beta < 1$ requires $\beta < 0.0025$, i.e., $\alpha\bar{\rho} < 0.0025$. Our actual hyperparameters violate this in the worst case.

**Resolution**: in practice ρ̄ − ρ is small most of the time (gradients usually mostly aligned). We can either:
- (a) prove **expected contraction** under the stationary distribution of $\rho$, where $\mathbb{E}[\beta] \ll \alpha\bar{\rho}$
- (b) require a tighter $\bar{\rho}$ in the journal version (e.g., $\bar{\rho} = 0.05$)
- (c) prove contraction in a **projected** subspace where $Q(\cdot,\bm{\lambda}_1) \approx Q(\cdot,\bm{\lambda}_2)$ already

### 2.4 Lemma 1 (COR Loss Bound)

> **Lemma 1.** When $\rho < \bar{\rho}$, the COR penalty satisfies
> $$\mathcal{L}_{\text{COR}} \leq \alpha(\bar{\rho} - \rho) D^2$$
> where $D = \sup_\psi \|Q_\psi(\cdot,\bm{\lambda}_1) - Q_\psi(\cdot,\bm{\lambda}_2)\|_2$.

### 2.5 Proposition 1 (Pareto Regret)

> **Proposition 1.** With $T$ training steps, an actor parameter $\theta_T$ produced by SemMORL satisfies:
> $$R_{\text{Pareto}}(\theta_T) := \mathcal{P}^* - \mathbf{J}(\pi_{\theta_T}) \leq O\!\left(\frac{1}{\sqrt{T}}\right) + \mathcal{C}_{\text{COR}}$$
> where $\mathcal{C}_{\text{COR}}$ is a constant depending on $\alpha, \bar{\rho}$ and the curvature of $\mathcal{P}^*$.

This requires a more elaborate proof using policy gradient theorem + envelope-MORL bounds; would be the thesis-level lemma.

### 2.6 Complexity Statement

Per training step for SemMORL:
- Forward pass: $O(d_z(5K + 9))$ for OADM + $O(d_h(5K + 5 + |\bm{\lambda}|))$ for actor + critic
- COR computation: $O(d \cdot |\bm{\lambda}|)$ for stiffness + $O(d_h)$ for value diff
- Total: $O(d_z K + d_h K)$ per step, **linear in $K$**

In the multi-UAV setting (Task C.1), per-step cost scales as $O(M(d_z K + d_h K)) = O(MK)$.

---

## 3. C-MORL / PSL-MORL Porting Roadmap (Task C.4 baselines)

### 3.1 C-MORL (Constrained MORL, NeurIPS 2025)

**Paper**: arXiv:2410.02236
**Reference repo**: search for `c-morl` on GitHub (released alongside NeurIPS publication)

#### Core algorithm (two-stage)

**Stage 1**: Train a population of policies, each maximizing one objective.
**Stage 2**: Constrained optimization — for each remaining objective, run constrained policy optimization with other objectives as constraints exceeding learned thresholds.

#### Adaptation steps

| Step | Effort |
|------|--------|
| Clone repo, run on MuJoCo cheetah benchmark to verify baseline | 0.5 week |
| Adapt action space: from MuJoCo control to UAV-SemCom action vector $(\nu, p, \eta)$ | 1 week |
| Adapt reward vector: replace native reward with our 4-objective UAV-SemCom rewards | 1 week |
| Hyperparameter tuning for our environment (preference grid, constraint thresholds) | 1 week |
| **Total** | **3.5 weeks** |

**Key concern**: C-MORL is designed for **discrete preference partitions**; our continuous 56-preference grid may need adaptation. Use top-4 simplex corner preferences for Stage 1, then sweep middle.

### 3.2 PSL-MORL (Pareto Set Learning, 2025)

**Paper**: arXiv:2501.06773
**Reference repo**: search for `pareto-set-learning-morl` or similar on GitHub

#### Core algorithm

Hypernetwork $h_\theta(\bm{\lambda}) \to$ policy parameters. One forward through hypernet produces a policy specialized to preference $\bm{\lambda}$.

#### Adaptation steps

| Step | Effort |
|------|--------|
| Clone repo, verify on bench task | 0.5 week |
| Wrap our actor architecture as `policy_net(s; theta(λ))` | 1 week |
| Train hypernetwork on our 56 preference grid | 1 week |
| Evaluation: hypervolume + Pareto coverage | 0.5 week |
| **Total** | **3 weeks** |

**Key concern**: Hypernetworks need careful initialization to avoid policy collapse; budget extra time if preliminary runs show degenerate policies.

### 3.3 Other Recommended Baselines

| Baseline | Source | Effort |
|----------|--------|--------|
| MO-PPO | Stable-Baselines3 community implementation | 1 week |
| Pareto Q-Learning | Hayes et al. 2022 textbook implementation | 1 week |
| Pareto-PG (Pareto Policy Gradient) | classic, implementation widely available | 1 week |

### 3.4 Reproduction Hygiene

For each ported baseline:

1. **Verify on original bench task** (e.g., MO-MuJoCo, MO-CartPole) before adapting.
2. **Match seed budget**: 6 seeds, same hyperparameter grid as SemMORL.
3. **Same evaluation protocol**: 56 preferences, 200 rollout episodes, hypervolume + Pareto coverage.
4. **Save raw outputs** in identical npz format to allow common plotting pipeline.

### 3.5 Total Baseline Porting Effort

3.5 + 3 + 3 = **~10 weeks**, mostly engineering. Overlaps with Task C.1 multi-UAV work — can run in parallel with one engineer per task.

---

## 4. Critical Path Decision Points

These are the points where additional design decisions are needed before coding begins:

1. **Multi-UAV — CTDE vs Federated Latent**:
   - Decide after running CTDE for $M=2$. If wall-clock or reward variance acceptable, scale to $M=4,5$. If not, switch to federated.
   - **Decision deadline**: 2026-06-15.

2. **Theorem 1 — full proof or expected-contraction variant**:
   - Try full proof first. If hyperparameter range is too restrictive for our $\gamma$, fall back to expected contraction over stationary $\rho$ distribution.
   - **Decision deadline**: 2026-08-01.

3. **C-MORL adaptation — discrete vs continuous preferences**:
   - Try Stage-1 with 4 simplex corners. If quality of resulting Pareto coverage is comparable to ours, continue. If degraded, run Stage-1 on 16 preferences (subset of our 56).
   - **Decision deadline**: 2026-09-15.

---

## 5. Suggested Owner Assignments (if multi-person team)

| Task | Skill profile | Estimated person |
|------|---------------|------------------|
| C.1 Multi-UAV | RL engineer + multi-agent experience | Engineer 1 |
| C.2 Constrained MORL | Optimization + safe RL background | Engineer 2 (could be student) |
| C.3 Theory | Theoretical RL + linear algebra | Advisor + grad student |
| C.4 Baselines + experiments | RL engineer + benchmark hygiene | Engineer 1 (parallel) |
| Writing | Lead author | Qiankun |

---

## 6. References Used in Sketches

- C-MORL: arXiv:2410.02236 (NeurIPS 2025) — Stage-1/Stage-2 constrained Pareto.
- PSL-MORL: arXiv:2501.06773 (2025) — hypernetwork-based Pareto set learning.
- Hayes et al. 2022 — practical guide to MORL planning.
- COLA 2025 — base method for our COR.
- Yang 2019 — Envelope SAC base for our critic backup.
