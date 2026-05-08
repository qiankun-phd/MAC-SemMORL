"""C-MORL baseline (Issue #8 PR-E, **stub**).

This file establishes the registration entry, the configuration surface, the
upstream-repo provenance, and the porting checklist so the actual training
loop can land in a follow-up commit without re-deriving the integration
contract. ``train()`` raises NotImplementedError until the upstream code is
vendored and adapted (~3 weeks of work per DESIGN-baselines.md §5 PR-E gate).

The decision to ship the stub now (rather than a full port) is documented in
DESIGN-baselines.md §5 — PR-E is allocated week 4–7 and runs in parallel with
PR-F. Having the registration + config plumbing land early lets the runner +
analysis tooling start being exercised against the C-MORL filename / npz path
before the heavy port finishes.

Upstream:
    paper:        Liu et al. 2024 — "C-MORL: Constrained Multi-Objective RL"
                  arXiv:2410.02236, ICLR 2025.
    repo:         https://github.com/RuohLiuq/C-MORL
    pinned commit: 67473b5afbc1be55e2b8c6ae704afc927bf218ee (2025-08-27)
    license:      not declared at repo root as of 2026-05-02 — see SKETCHES §3.1
                  for the action item to confirm permitted use before vendoring.

Algorithm summary (Stage 1 + Stage 2):
    Stage 1 — train one policy per simplex-corner preference (4 corners for our
              N=4). Use vanilla PPO from the upstream repo.
    Stage 2 — for each *non-corner* preference in the eval grid, run a
              constrained PPO that maximises the targeted objective subject to
              the other three exceeding their Stage-1 levels minus a slack.
              Lagrangian dual handles the constraints.

Adaptation work (DESIGN-baselines.md §5 PR-E + SKETCHES §3.1):
    1. Vendor the C-MORL repo at the pinned commit under
       ``vendor/c_morl/`` with an attribution NOTICE and a deferred-license
       resolution path. (~0.5 week)
    2. Write a gym-compatible env adapter that exposes our
       ``UAVSemComEnv`` / ``MultiUAVSemComEnv`` through the
       ``gym.Env`` interface that C-MORL expects — its native interface is
       gym 0.21 four-tuple step(); ours uses the same convention so this is
       a thin shim. (~0.5 week)
    3. Adapt the action mapping from MuJoCo control to our
       (a_x, a_y, p_k, η_k) layout. (~1 week)
    4. Adapt the reward vector — replace native multi-objective reward with
       our 4-objective UAV-SemCom rewards. (~1 week)
    5. Hyperparameter tuning for our environment (preference grid, constraint
       thresholds, Stage-1 corner selection). (~0.5 week)
    Decision deadline 2026-09-15: if Stage-2 still degenerate after 1.5 weeks,
    fall back to "C-MORL-corner" or "C-MORL S1" per pre-mortem §6.1.

Files this stub touches:
    baselines/c_morl.py          — this file (registration + config + plan)
    baselines/__init__.py        — adds ``from . import c_morl`` import
    scripts/smoketest_cmorl_stub.py — confirms registration works without
                                      depending on the upstream repo
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import Baseline
from .registry import register_baseline


@register_baseline("c-morl")
class CMORLBaseline(Baseline):
    """C-MORL baseline — registration + plan stub.

    Constructable for runner-side smoke testing. ``train()`` raises until the
    full port lands. ``policy_fn`` returns a midpoint action so the runner's
    final-eval pass cannot crash when invoked.
    """

    name = "c-morl"

    def __init__(
        self,
        env,
        log_dir: str,
        seed: int,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(env, log_dir, seed, method_kwargs)
        self.is_constrained = True  # C-MORL enforces constraints natively
        # Once the port lands, c_violation_rates will be populated from the
        # final-eval pass on the saved checkpoint.

    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        raise NotImplementedError(
            "C-MORL training loop is not yet ported. See baselines/c_morl.py "
            "module docstring for the porting plan and DESIGN-baselines.md §5 "
            "PR-E for the schedule. Pinned upstream commit: 67473b5."
        )

    def policy_fn(self, obs: np.ndarray, preference: np.ndarray) -> np.ndarray:
        # Mid-range action — enough to make the runner's final-eval pass not
        # crash when the stub is exercised.
        low = self.env.action_space.low
        high = self.env.action_space.high
        return ((low + high) / 2.0).astype(np.float32)
