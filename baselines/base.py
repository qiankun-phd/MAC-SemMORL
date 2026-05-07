"""Baseline ABC for the journal-extension comparison study.

Every method ported in Issue #8 implements this interface so the unified
runner (``scripts/run_baseline.py``) can drive them through the protocol
locked in ``docs/DESIGN-baselines.md`` §2.

Why this exists: per-baseline upstream repos each ship with their own eval
defaults — different seed lists, different HV reference points, different
preference grids. If we let each port keep its native defaults, our
comparison numbers will not be honest. The ABC narrows what a baseline
implementation can vary: it controls the policy and the training loop;
it does *not* control the eval cadence, the seed list, or the metric
suite. Those are the runner's job.

Design notes:
    - ``train`` is a single blocking call. The runner sets ``num_steps``
      and ``eval_interval`` from the protocol; the baseline must call
      ``self._eval_and_log`` at least every ``eval_interval`` env steps so
      the HV trajectory is populated.
    - ``policy_fn`` is a deterministic policy lookup. Stochastic baselines
      should set their RNG to a fixed seed and return the mean action.
      The runner uses this for the *final* eval pass on saved
      checkpoints — the baseline's own internal eval is logged but not
      used for the journal numbers.
    - ``save_results`` writes the npz schema in
      ``docs/DESIGN-baselines.md`` §3. The default implementation in
      ``write_result_npz`` handles the standard fields; subclasses only
      need to populate ``c_violation_rates`` if they enforce constraints.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import numpy as np


class Baseline(ABC):
    """Abstract base class every Issue #8 baseline must satisfy."""

    name: str = "unset"  # subclasses must override

    def __init__(
        self,
        env,
        log_dir: str,
        seed: int,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.name == "unset":
            raise TypeError(
                f"{type(self).__name__} must set a class-level `name` attribute "
                f"matching its registry key."
            )
        self.env = env
        self.log_dir = log_dir
        self.seed = int(seed)
        self.method_kwargs = method_kwargs or {}

        # Trajectory buffers populated during training. Must satisfy
        # docs/DESIGN-baselines.md §3 schema.
        self.eval_steps: List[int] = []
        self.hv_trajectory: List[float] = []
        self.ut_trajectory: List[float] = []
        self.sparsity_trajectory: List[float] = []
        # Final-state buffers (populated at end of training).
        self.final_ep_objs: Optional[np.ndarray] = None
        self.final_ep_prefs: Optional[np.ndarray] = None
        self.final_obj_means: Optional[np.ndarray] = None
        self.final_obj_stds: Optional[np.ndarray] = None
        # NaN-by-default — only constrained baselines populate this.
        self.c_violation_rates: np.ndarray = np.full(3, np.nan)
        self.wallclock_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def train(
        self,
        num_steps: int,
        eval_interval: int,
        our_wandb=None,
    ) -> None:
        """Train for `num_steps` env steps. Subclass must call
        ``self._record_eval(step, hv, ut, sparsity)`` at least every
        ``eval_interval`` steps and set the four ``final_*`` buffers
        before returning.
        """

    @abstractmethod
    def policy_fn(
        self,
        obs: np.ndarray,
        preference: np.ndarray,
    ) -> np.ndarray:
        """Deterministic policy lookup. Used by the runner's final-eval
        pass on the saved checkpoint."""

    # ------------------------------------------------------------------
    # Helpers used by subclasses
    # ------------------------------------------------------------------
    def _record_eval(
        self,
        step: int,
        hv: float,
        ut: float,
        sparsity: float,
    ) -> None:
        """Append a row to the trajectory buffers. Trains call this at
        the eval_interval cadence."""
        self.eval_steps.append(int(step))
        self.hv_trajectory.append(float(hv))
        self.ut_trajectory.append(float(ut))
        self.sparsity_trajectory.append(float(sparsity))
