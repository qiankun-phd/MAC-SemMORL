"""PSL-MORL baseline (Issue #8 PR-F, **stub**).

Pareto Set Learning for MORL via hypernetworks. Same stub strategy as
``c_morl.py`` (PR-E): registration + porting checklist + raise NotImplementedError
on ``train()``. Full re-implementation lands in follow-up commits.

Why a stub now: PSL-MORL has **no public repo** as of 2026-05-02 (SKETCHES §3.2),
so the port is a re-implementation from the paper algorithm box. Decision
deadline 2026-09-01 — until then we keep watching the authors' channels.

Upstream:
    paper:        Lin et al. 2025 — "PSL-MORL: Pareto Set Learning for MORL"
                  arXiv:2501.06773, AAAI 2025.
    repo:         not found as of 2026-05-02 — must re-implement.
    license:      N/A.

Algorithm summary (paper §3):
    Hypernetwork  h_θ : Δ^{N-1} → R^{|θ_actor|}
        — MLP that maps a preference vector to a policy-parameter vector.
    Parameter fusion  θ = (1 − α) θ_1 + α θ_2  with θ_2 = h_θ(λ)
        — base actor parameters θ_1 mixed with hyper-net output. α ∈ [0.01, 0.05]
        per env, grid-searched.
    Outer loop: sample λ from the simplex, run TD3 update with scalarised
        reward w^T r, backprop through the hypernetwork.

Re-implementation work (DESIGN-baselines.md §5 PR-F + SKETCHES §3.2.1, ~3.5 weeks):
    1. ``baselines/psl_morl/hypernetwork.py`` — MLP mapping
       λ ∈ R^N → flat θ_actor with linear layers + ReLU. (~0.5 week)
    2. ``baselines/psl_morl/actor.py`` — parameter-fusion actor, load
       θ_2 = h_θ(λ), fuse with base θ_1. (~1 week)
    3. ``baselines/psl_morl/train.py`` — outer loop: sample λ, TD3 update
       on scalarised reward, hypernetwork update. (~1 week)
    4. Hyperparameter tuning on our environment (α grid, hypernet width,
       hypernet depth). (~0.5 week)
    5. Evaluation harness — already provided by the runner + npz schema. (~0.5 week)

Risk per pre-mortem (DESIGN §6.2):
    Hypernetworks need careful init to avoid policy collapse — budget extra
    time if preliminary runs show degenerate policies. If the authors release
    the repo mid-implementation, switch costs more than continuing; ship our
    re-implementation with a citation note instead of switching.
    Sanity check: re-implementation must match the paper's MO-Hopper-2d
    numbers within 10% before it is allowed to claim the name PSL-MORL.

Files this stub touches:
    baselines/psl_morl.py        — this file (registration + plan)
    baselines/__init__.py        — adds the lazy import
    scripts/smoketest_psl_morl_stub.py — confirms registration works
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import Baseline
from .registry import register_baseline


@register_baseline("psl-morl")
class PSLMORLBaseline(Baseline):
    """PSL-MORL baseline — registration + plan stub.

    Constructable. ``train()`` raises until the re-implementation lands.
    ``policy_fn`` returns a midpoint action so the runner's final-eval pass
    cannot crash when the stub is exercised.
    """

    name = "psl-morl"

    def __init__(
        self,
        env,
        log_dir: str,
        seed: int,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(env, log_dir, seed, method_kwargs)
        # PSL-MORL is unconstrained — c_violation_rates stays NaN (default).

    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        raise NotImplementedError(
            "PSL-MORL training loop is not yet implemented. The paper has no "
            "public repo as of 2026-05-02 (SKETCHES §3.2); re-implementation "
            "is scheduled for weeks 4-7 of Issue #8 (DESIGN-baselines.md §5 "
            "PR-F). See baselines/psl_morl.py module docstring for the plan."
        )

    def policy_fn(self, obs: np.ndarray, preference: np.ndarray) -> np.ndarray:
        low = self.env.action_space.low
        high = self.env.action_space.high
        return ((low + high) / 2.0).astype(np.float32)
