"""No-op baseline used to smoke-test the framework (Issue #8 PR-B).

Returns random actions, populates the trajectory buffers with zeros so the
schema validator passes, and exits. Not a research baseline — exists so that
``scripts/run_baseline.py --baseline noop`` can validate the runner +
npz schema in CI without depending on any external repo.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import Baseline
from .registry import register_baseline


@register_baseline("noop")
class NoOpBaseline(Baseline):
    """Random-action baseline. Useful only for skeleton testing."""

    name = "noop"

    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        rng = np.random.RandomState(self.seed)
        steps = 0
        t0 = time.time()
        while steps < num_steps:
            self.env.seed(self.seed + steps)
            obs = self.env.reset()
            done = False
            while not done and steps < num_steps:
                action = rng.uniform(
                    self.env.action_space.low, self.env.action_space.high
                )
                _obs, _reward, done, _info = self.env.step(action)
                steps += 1
                if steps % eval_interval == 0:
                    # Cheap "eval": no Pareto front to compute, just zeros so
                    # the schema is populated.
                    self._record_eval(step=steps, hv=0.0, ut=0.0, sparsity=0.0)
        # Final-state buffers — minimum schema-valid content.
        self.final_ep_objs = np.zeros((1, self.env.reward_num), dtype=np.float64)
        self.final_ep_prefs = np.full(
            (1, self.env.reward_num), 1.0 / self.env.reward_num, dtype=np.float64
        )
        self.final_obj_means = np.zeros(self.env.reward_num, dtype=np.float64)
        self.final_obj_stds = np.zeros(self.env.reward_num, dtype=np.float64)
        # c_violation_rates stays NaN — noop is unconstrained.
        self.wallclock_seconds = time.time() - t0

    def policy_fn(self, obs: np.ndarray, preference: np.ndarray) -> np.ndarray:
        # Mean of a uniform distribution is the midpoint of the action space.
        low, high = self.env.action_space.low, self.env.action_space.high
        return ((low + high) / 2.0).astype(np.float32)
