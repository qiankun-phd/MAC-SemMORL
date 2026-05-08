# Design Document: Baseline Ports + Unified Evaluation Protocol

**Status**: Draft — pending reviewer approval
**Date**: 2026-05-07
**Author**: @claude (drafted from PLAN.md §C.4, SKETCHES.md §3, ROADMAP.md P2.3-P2.7, Issue #8)
**Decision deadlines**: 2026-09-15 (C-MORL discrete-pref adaptation, see SKETCHES §3.1), 2026-09-01 (PSL-MORL re-implement vs wait, see SKETCHES §3.2)
**Roadmap items**: P2.3 (C-MORL), P2.4 (PSL-MORL), P2.5 (other classics), P2.6 (real channel), P2.7 (mobility)

---

## 0. Purpose and Scope

This document locks the engineering interface, evaluation protocol, and PR-phasing for the baseline-port effort (Task C.4) before any baseline code lands. It is the authoritative reference for:

- The five baselines being ported (which, why, in what order).
- The unified evaluation protocol every baseline must satisfy (so cross-method comparisons are honest).
- The result-storage format on disk so `experiments/analysis/` can compare them.
- The PR phasing for Issue #8 — six PRs over five to seven weeks.
- A pre-mortem on the most likely porting failures.

Companion documents: `docs/PLAN.md` §C.4, `docs/SKETCHES.md` §3 (per-baseline porting roadmap with repo URLs and pinned commits), `docs/DESIGN-multi-uav.md` (the env that all baselines must consume), `docs/DESIGN-constrained.md` (the constraints that one of the baselines, C-MORL, has native handling for).

---

## 1. Baselines: What and Why

Five external baselines plus our own (six methods total in the journal experiments table).

| # | Method | Year/venue | Why we need it | Per-method effort |
|---|--------|-----------|---------------|-------------------|
| 1 | **C-MORL** | NeurIPS 2025 | Constrained-MORL SOTA; same constraint formulation as our Issue #6. Without C-MORL we have no positioning anchor for the constrained novelty story. | 3.5 weeks |
| 2 | **PSL-MORL** | AAAI 2025 | Hypernetwork-based Pareto set learning; *current* SOTA on the unconstrained continuous-preference axis. Without PSL-MORL the unconstrained Pareto-front comparison is incomplete. | 3.0–3.5 weeks |
| 3 | **MO-PPO** | community SB3 | Standard MORL workhorse; mandatory for any TWC reviewer who wants a "vanilla" comparison point. Cheap to port. | 1.0 week |
| 4 | **Pareto Q-Learning** | Hayes et al. 2022 | Tabular/discrete-action MORL from the same survey we cite (`hayes2022survey`). Establishes the lower-bound "no-deep-RL" data point. | 1.0 week |
| 5 | **Pareto-PG** | classic | Vanilla Pareto policy gradient; the simplest baseline reviewers expect. | 0.5 week |

**Total**: 9.0–9.5 weeks if sequential, ≤ 7 weeks with two engineers in parallel.

### 1.1 Order of work

1. **MO-PPO first** (cheapest, validates the eval pipeline) → 1 week.
2. **Pareto-PG + Pareto Q-Learning together** (small, classic) → 1.5 weeks.
3. **C-MORL** (the highest-value port; biggest risk) → 3.5 weeks. Starts after the eval pipeline is proven.
4. **PSL-MORL last** (no public repo as of survey date — re-implement from paper) → 3.0–3.5 weeks. Decision deadline 2026-09-01 on whether to wait for an official release or re-implement.

This ordering means the journal "vanilla MORL row" ships in week 1, the "tabular/classical row" ships in week 2.5, and the two SOTA rows (C-MORL, PSL-MORL) finish by week 7 — the exact deadline in `docs/ROADMAP.md` Phase 2.

### 1.2 Methods explicitly *not* included as baselines

| Considered | Status | Reason |
|------------|--------|--------|
| Original COLA (conference paper) | Excluded as a baseline; included as **our prior work** in the comparison table | The conference paper is what we're extending; it's our M=1 reference, not a baseline. |
| GMAC (LLM-assisted multi-agent comm) | Deferred | Different problem formulation (cooperation-via-LLM), would need a custom adaptation that's bigger than the rest of the baselines combined. |
| MO-Q-DQN, MO-Actor-Critic | Excluded | Subsumed by Pareto Q-Learning (#4) and MO-PPO (#3). |
| Random / round-robin / greedy heuristics | Already in `baselines.py` (lines 97–158) | Treated as classical heuristics, not RL baselines. Reported in a separate paper subsection. |
| Federated MARL baselines | Deferred to a future paper | Multi-UAV federated MARL is itself one of our novelties (ADR-0005); comparing against another federated-MARL method is too narrow for the TWC scope. |

---

## 2. Unified Evaluation Protocol

Every ported baseline must conform to this protocol. The protocol is **stricter than the conference paper** (more seeds, more preferences, longer eval) so the journal numbers are more defensible against reviewer pushback.

### 2.1 Training budget

| Setting | Value | Rationale |
|---------|-------|-----------|
| Total env steps | $1{,}000{,}000$ | Empirically HV plateaus by 140K (per `t1_estimator.log`); 1M leaves headroom and matches the M=2 fix-energy pilot. |
| Seeds per (method, config) | 6 | Conference paper used 4; +2 for tighter confidence intervals. |
| Eval cadence | every 20K env steps | Matches our existing pilot logs. |
| Final eval | at step 1M + 200 | Matches existing checkpoint convention. |

Budget per method per config: $6 \text{ seeds} \times \approx 5\text{h wallclock} = 30\text{ GPU-hours}$ at M=2. Multiply by configs (M, K, mobility, robustness) for the full sweep.

### 2.2 Preference grid

| Setting | Value |
|---------|-------|
| Number of preferences for HV computation | $56$ (matches conference paper) |
| Generation method | `generate_w_batch_test(reward_num=4, step_size=0.2)` already in `agent.py:262` |
| Per-preference rollout length | $200$ steps (one episode) |
| Per-preference rollouts (train-time eval) | $1$ |
| Per-preference rollouts (final eval) | $5$, averaged |

C-MORL natively expects discrete preference partitions; we sample 4 simplex-corner preferences for its Stage-1 population, then sweep the 56-grid for Stage-2 / final HV (decision per SKETCHES §3.1).

### 2.3 Metrics reported

For each (method, config) the following must end up in the result file:

- **Hypervolume (HV)** at the final step, computed by `evluate_Hv_UT_and_spa(N=4, …)` against the all-zeros reference point.
- **HV trajectory** at every 20K-step eval (for learning-curve plots).
- **Sparsity** of the final EP (Pareto front spread).
- **Utility (UT)** = mean over the 56 preferences of $\max_\pi \mathbf{w}^\top\mathbf{r}$.
- **Per-objective episode means** (Fid, AoSI, Energy, Jain) at the final step.
- **Constraint violation rates** (only meaningful for constrained methods — C-MORL and our SemMORL with `--use_lagrangian`):
  - $\hat{c}_1$: $\Pr_t[\max_k A_k > A_{\max}]$
  - $\hat{c}_2$: $\sum_t E_{\text{fleet}}(t) > E_{\text{total}}$ (binary per episode)
  - $\hat{c}_3$: $\Pr_t[\min_k \rho_{\text{svc},k} < \rho_{\min}]$

### 2.4 Configuration matrix

For the journal experiments section, the full sweep is:

| Axis | Values | # configs |
|------|--------|-----------|
| Number of UAVs $M$ | 1, 2, 4, 5 | 4 |
| Number of devices $K$ | 5, 10, 20, 50 | 4 |
| Mobility model | none, RWP, Levy, Group | 4 |
| Channel | analytical, DeepMIMO replay | 2 |
| Robustness perturbation | none, ±2 dB, ±5 dB, pref-skew, traffic-shift | 5 |
| Constraint handler (SemMORL only) | none, lagrangian, barrier, projection | 4 |

Full Cartesian product is infeasible. Per `docs/PLAN.md` §C.4, the **anchor configuration** is $M=2, K=5$, mobility=none, analytical channel, no perturbation — every method runs this. Each axis is then varied **one at a time** from the anchor (line, not grid). Total: $5 + 3 + 3 + 3 + 1 + 4 + 3 = 22$ configs per method × 6 methods = **132 runs**, well within the GPU budget reserved in ADR-0004.

---

## 3. On-Disk Result Format

All baselines write a single npz file per (method, config, seed) into `results/`. The schema is:

```
results/{method}_{tag}_{seed}.npz
```

Where `{tag}` encodes the configuration as `M{m}_K{k}_mob{mob}_ch{ch}_pert{pert}_constr{handler}`. Example: `results/C-MORL_M2_K5_mobnone_chanalytical_pertnone_constrlagrangian_seed1.npz`.

Each npz file contains:

| Key | Shape | Dtype | Meaning |
|-----|-------|-------|---------|
| `eval_steps` | $(E,)$ | int64 | Eval step indices, ascending |
| `hv_trajectory` | $(E,)$ | float64 | HV at each eval |
| `ut_trajectory` | $(E,)$ | float64 | UT at each eval |
| `sparsity_trajectory` | $(E,)$ | float64 | Sparsity at each eval |
| `final_ep_objs` | $(P, 4)$ | float64 | Final EP point set, $P$ Pareto-optimal preferences |
| `final_ep_prefs` | $(P, 4)$ | float64 | Preference vectors for those points |
| `final_obj_means` | $(4,)$ | float64 | Per-objective episode means at step 1M |
| `final_obj_stds` | $(4,)$ | float64 | Per-objective episode stds |
| `c_violation_rates` | $(3,)$ | float64 | $[\hat c_1, \hat c_2, \hat c_3]$ violation rates; NaN if method is unconstrained |
| `wallclock_seconds` | scalar | float64 | Total training wallclock |
| `git_sha` | scalar | str | Git SHA of the code that ran |
| `config_dict` | scalar | str (JSON) | Full CLI args + env kwargs as JSON for reproducibility |

`experiments/analysis/analyze_results.py` will be extended (PR-G of this issue) to load this format and produce the journal comparison tables and Pareto-front plots.

---

## 4. The `Baseline` Interface

To minimise per-baseline porting effort and guarantee evaluation parity, every baseline ships as a class implementing the following interface in `baselines/`:

```python
class Baseline:
    name: str  # e.g., "C-MORL", "MO-PPO"

    def __init__(self, env, log_dir: str, seed: int, **method_kwargs): ...

    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        """Train for `num_steps` env steps. Should call `self._eval_and_save`
        every `eval_interval` steps so the result npz is written progressively."""

    def policy_fn(self, obs: np.ndarray, preference: np.ndarray) -> np.ndarray:
        """Deterministic policy lookup at uniform-preference eval. Used by
        the final-eval pass and by `scripts/diagnose_pilot.py`."""

    def save_results(self, output_path: str) -> None:
        """Write the npz file in the schema of §3. Subclass-provided
        defaults are merged with `c_violation_rates = nan` for unconstrained
        methods."""
```

A common runner script `scripts/run_baseline.py` consumes a `--baseline {name}` CLI flag and a config dict, instantiates the right subclass, and calls `train` + `save_results`. This is the same launching pattern as `main_uav.py` so server scheduling / wandb / nohup conventions stay identical.

The existing `baselines.py` (heuristic baselines: random, round-robin, greedy) becomes `baselines/heuristics.py` and is wrapped to satisfy the same interface so the heuristic comparison rows are a free byproduct.

---

## 5. Phased PR Plan

Six PRs, mergeable mostly in isolation. Letters (A–G) line up with the existing pattern (A = design, B = first impl, …).

| PR | Branch | Scope | Time | Gates |
|----|--------|-------|------|-------|
| **A (this PR)** | `claude/issue-8-design-baselines` | This design doc | day 1 | reviewer approval |
| **B** | `claude/issue-8-baseline-skeleton` | `baselines/` package layout, `Baseline` ABC, `scripts/run_baseline.py`, npz schema writer, smoke test of the skeleton with a no-op baseline | week 1 | smoke test passes; npz schema validated against §3 |
| **C** | `claude/issue-8-mo-ppo` | MO-PPO port from SB3 community | week 2 | reproduces SB3 reported HV on Hopper-2d within 5%; one full anchor-config run on UAV-SemCom completes |
| **D** | `claude/issue-8-classical` | Pareto-PG + Pareto Q-Learning | week 3-4 | one full anchor-config run each; HV non-degenerate (i.e., better than random heuristic) |
| **E** | `claude/issue-8-cmorl` | C-MORL port at pinned commit `67473b5` (SKETCHES §3.1) | week 4-7 | reproduces Building-3d Stage-1 pop on the upstream benchmark; one full anchor-config run on UAV-SemCom; constraint violation rates ≤ ε at final eval |
| **F** | `claude/issue-8-psl-morl` | PSL-MORL re-implementation from paper algorithm box | week 4-7 | re-implementation matches the paper's Mo-Hopper-2d numbers within 10%; one full anchor-config run on UAV-SemCom |
| **G** | `claude/issue-8-comparison` | Sweep all 22-config × 6-method anchor matrix; extend `analyze_results.py` to ingest the npz schema; produce the journal comparison table + Pareto-front figure | week 7 | reviewer can view final HV table and Pareto-front plot |

PR-E and PR-F may run in parallel after PR-B/C/D land, since they have independent skeleton consumers.

---

## 6. Pre-Mortem

Three failure modes most likely to derail the porting effort, ranked by probability.

### 6.1 C-MORL discrete-preference adaptation fails

**Symptom**: C-MORL Stage-2 step diverges or produces a degenerate Pareto front (single point) when fed our continuous 56-preference grid.

**Cause**: C-MORL's Stage-2 explicit constraint thresholds are computed assuming the Stage-1 population covers the simplex corners exactly. Our continuous grid does not naturally produce corner-aligned policies.

**Mitigation**:
- Decision deadline 2026-09-15 (already in SKETCHES §3.1 and ROADMAP D3): if Stage-2 still degenerate after 1.5 weeks of porting, fall back to "C-MORL-corner": Stage-1 is forced to the 4 simplex corners only, Stage-2 sweeps the middle 52 preferences. Document the deviation in the paper experiments section.
- Last-resort fallback: drop C-MORL's Stage-2 entirely and report Stage-1-only numbers as "C-MORL S1". The constrained-Pareto comparison then comes only from our SemMORL with `--use_lagrangian`. This weakens the positioning but doesn't kill the paper.

### 6.2 PSL-MORL has no public release before our deadline

**Symptom**: 2026-09-01 arrives, the paper authors still have not released code.

**Cause**: AAAI 2025 author release schedules are loose; 6+ months post-publication without code is not unusual.

**Mitigation**:
- Begin re-implementation from the paper algorithm box on 2026-08-15 in parallel with watching the authors' repo.
- Re-implementation lives in `baselines/psl_morl/` with citation to the paper version. If the official repo appears mid-implementation, add a one-liner note in the file header and continue with our version (citing both); switching mid-implementation is more disruptive than finishing.
- Sanity check the re-implementation against the paper's Mo-Hopper-2d numbers within 10% before claiming it's PSL-MORL.

### 6.3 Eval-protocol divergence between baselines

**Symptom**: "Method X reports HV = Y" but on closer inspection X used 4 seeds where the protocol calls for 6, or X used a different reference point for HV.

**Cause**: Each upstream baseline ships with its own eval defaults; those defaults will leak into our port unless the runner explicitly overrides them.

**Mitigation**:
- The `scripts/run_baseline.py` runner is the single source of truth for the eval cadence + seed list. Per-baseline classes only see `train(num_steps, eval_interval)` and cannot override the post-training final-eval pass.
- Final eval is a *separate* pass, run by `scripts/diagnose_pilot.py`-style code on the saved checkpoint, not by the baseline's internal eval. Same code for all 6 methods.
- PR-B includes a unit test that runs the no-op baseline through `run_baseline.py` and asserts the npz schema (§3) is correct. PR-G runs the same assertion on every actual result file before producing the comparison table.

---

## 7. Open Questions for Reviewer

Resolve before merging this PR.

1. **Pareto Q-Learning scope**: full continuous-state version (needs function approximation) or discrete-state via state aggregation? The Hayes et al. 2022 paper presents a tabular formulation; UAV-SemCom is continuous. Tabular-with-aggregation is faster (1 week) but a weaker baseline; FA version is 2.5 weeks. Default in this doc: tabular-with-aggregation, document the limitation.

2. **MO-PPO scalarisation**: weighted-sum, Tchebycheff, or both? Conference paper used weighted-sum; the SB3 community port supports both. Default: weighted-sum only (faster, matches conference). Tchebycheff as a deferred ablation only if a reviewer asks.

3. **Result format on disk**: do we want one npz per (method, config, seed), or one big npz per (method, config) with a seed dimension? Default in this doc is one-per-seed (simpler resumes, easier to lose one without losing all). Counter-argument: one-per-config is what `analyze_results.py` ultimately wants.

4. **"Method-fairness" budget**: is 1M env steps fair to PSL-MORL, which the authors trained for 5M? Default: yes for the anchor comparison (matches our SemMORL budget); add a "fairness-budget" robustness column at 5M for PSL-MORL only, run on the anchor config, in PR-G. Documenting this transparently is more defensible than silently letting one method get 5× more budget.

---

## 8. Acceptance for This Design PR

This document is approved for implementation when:

- [ ] Reviewer confirms the five baselines are the right set (§1.2).
- [ ] Reviewer agrees the eval protocol (§2) is strict enough vs the conference baseline.
- [ ] Reviewer approves the npz schema (§3) — extension after PR-B is harder than getting it right now.
- [ ] Reviewer agrees the six-PR phasing (§5) and the per-PR gates.
- [ ] Reviewer answers the four open questions (§7) or proposes alternates.

After approval, PR-B (skeleton) opens against `main`.
