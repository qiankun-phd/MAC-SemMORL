"""Pareto Q-Learning baseline (Issue #8 PR-D, tabular variant).

Per DESIGN-baselines.md §7 Q1, the default scope here is **tabular with state
aggregation** — explicitly weaker than the function-approximation version,
chosen for porting speed (1 week budget). The journal experiments will
document this limitation.

Algorithmic simplification (also documented in §7 Q1): we store *one*
vector Q(s_macro, a_macro) per macro-cell rather than the full set of
non-dominated vectors. Action selection scalarises with the current
preference. This is the MO-Q variant of Pareto Q-Learning and remains the
"no-deep-RL lower bound" data point we want.

State aggregation (UAVSemComEnv only — Multi-UAV K=5 single-UAV path):
    - UAV position: 5×5 grid → 25 cells.
    - Mean AoSI bucket: low (≤2), medium (2–5), high (>5) → 3 cells.
    - Total macro-states: 75.

Action aggregation:
    - 9 macro-actions: 8 compass directions × fixed move speed, plus "stop".
    - Power and compression are fixed per macro-action (mid-range values).
    - Maps onto the env's continuous (a_x, a_y, p_k, η_k) action layout.

Limitations (will be cited verbatim in the paper):
    - Single-UAV only (M=1). Multi-UAV joint state would explode the macro
      table; requires a different aggregation scheme out of scope for #8.
    - Single Q-vector per cell (no Pareto set). FA version with
      non-dominated sets is deferred.

This is enough machinery to put a useful "below the line" data point on
the journal Pareto-front plot.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Dict, Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ._morl_eval import evluate_Hv_UT_and_spa, generate_w_batch_test
from .base import Baseline
from .registry import register_baseline


# Compass macro-actions: (dx, dy) at unit speed plus a stop.
MACRO_DIRS = np.array([
    (1.0, 0.0),    # E
    (-1.0, 0.0),   # W
    (0.0, 1.0),    # N
    (0.0, -1.0),   # S
    (0.7071, 0.7071),    # NE
    (-0.7071, 0.7071),   # NW
    (0.7071, -0.7071),   # SE
    (-0.7071, -0.7071),  # SW
    (0.0, 0.0),    # stop
], dtype=np.float32)
N_MACRO_ACTIONS = len(MACRO_DIRS)


def _state_to_cell(env, obs: np.ndarray) -> int:
    """Aggregate continuous state to a macro-cell id in [0, 75)."""
    # The first two dims of obs are normalised UAV pos in [0, 1].
    # AoSI starts at offset depending on K; pull from env directly.
    pos_x, pos_y = float(obs[0]), float(obs[1])
    # 5x5 grid
    bx = int(np.clip(pos_x * 5, 0, 4))
    by = int(np.clip(pos_y * 5, 0, 4))
    aosi_bucket = 0
    aosi = getattr(env, "aosi", None)
    if aosi is not None:
        m = float(np.mean(aosi))
        if m > 5.0:
            aosi_bucket = 2
        elif m > 2.0:
            aosi_bucket = 1
    return bx * 15 + by * 3 + aosi_bucket


def _macro_to_env_action(env, macro_idx: int) -> np.ndarray:
    """Map macro-action id to the continuous action vector the env expects.

    Layout (single-UAV): [a_x, a_y, p_1..K, η_1..K]. Macro-action only sets
    direction; power and compression are fixed at mid-range.
    """
    K = env.num_devices
    action = np.zeros(2 + 2 * K, dtype=np.float32)
    action[0:2] = MACRO_DIRS[macro_idx]
    # Fixed mid-range power (0.5 of [-1,1] mapping to half max_action).
    action[2:2 + K] = 0.5
    action[2 + K:2 + 2 * K] = 0.0  # compression at mid (sigmoid maps 0 → 0.5)
    return action


@register_baseline("pareto-q")
class ParetoQBaseline(Baseline):
    """Tabular MO-Q with single Q-vector per (macro-state, macro-action)."""

    name = "pareto-q"

    def __init__(self, env, log_dir: str, seed: int,
                 method_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(env, log_dir, seed, method_kwargs)
        if getattr(env, "num_uavs", 1) != 1:
            raise NotImplementedError(
                "Pareto-Q baseline supports M=1 only; multi-UAV aggregation "
                "is out of scope (DESIGN-baselines.md §7 Q1)."
            )
        mk = self.method_kwargs

        self.gamma = float(mk.get("gamma", 0.99))
        self.alpha = float(mk.get("alpha", 0.1))  # learning rate
        self.eps_start = float(mk.get("eps_start", 1.0))
        self.eps_end = float(mk.get("eps_end", 0.05))
        self.eps_decay_steps = int(mk.get("eps_decay_steps", 200_000))
        self.N = int(getattr(env, "reward_num", 4))

        self.n_states = 75  # 5x5 pos × 3 aosi buckets
        self.n_actions = N_MACRO_ACTIONS
        self.Q = np.zeros((self.n_states, self.n_actions, self.N), dtype=np.float64)

        self.rng = np.random.RandomState(self.seed)
        step_map = {3: 0.05, 4: 0.2, 5: 0.25}
        self.eval_prefs = generate_w_batch_test(
            self.N, step_size=step_map.get(self.N, 0.2)
        )

    # ------------------------------------------------------------------
    def _epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(self.eps_decay_steps, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _greedy_action(self, cell: int, pref: np.ndarray) -> int:
        scalarised = (self.Q[cell] * pref).sum(-1)  # (n_actions,)
        return int(np.argmax(scalarised))

    def _sample_pref(self) -> np.ndarray:
        p = self.rng.rand(self.N).astype(np.float32)
        p /= p.sum()
        return p

    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        env = self.env
        env.seed(self.seed)
        steps = 0
        t0 = time.time()
        next_eval = eval_interval
        cur_pref = self._sample_pref()
        state = env.reset()
        cell = _state_to_cell(env, state)

        while steps < num_steps:
            if self.rng.rand() < self._epsilon(steps):
                a_macro = int(self.rng.randint(self.n_actions))
            else:
                a_macro = self._greedy_action(cell, cur_pref)
            env_action = _macro_to_env_action(env, a_macro)
            next_state, reward, done, _ = env.step(env_action)
            next_cell = _state_to_cell(env, next_state)

            # Vector Bellman backup with scalarised greedy next action.
            r = np.asarray(reward, dtype=np.float64)
            if done:
                target = r
            else:
                a_next = self._greedy_action(next_cell, cur_pref)
                target = r + self.gamma * self.Q[next_cell, a_next]
            self.Q[cell, a_macro] += self.alpha * (target - self.Q[cell, a_macro])

            state = next_state
            cell = next_cell
            steps += 1
            if done:
                state = env.reset()
                cell = _state_to_cell(env, state)
                cur_pref = self._sample_pref()
            if steps >= next_eval:
                self._evaluate(steps)
                next_eval += eval_interval

        if not self.eval_steps or self.eval_steps[-1] < num_steps:
            self._evaluate(num_steps)
        self._populate_final_buffers()
        self.wallclock_seconds = time.time() - t0

    # ------------------------------------------------------------------
    def _evaluate(self, step: int) -> None:
        objs = []
        for pref in self.eval_prefs:
            objs.append(self._eval_episode(pref))
        objs = np.asarray(objs, dtype=np.float64)
        hv, sparsity, ut = evluate_Hv_UT_and_spa(self.N, objs, self.eval_prefs)
        self._record_eval(step=step, hv=hv, ut=ut, sparsity=sparsity)
        self._last_eval_objs = objs
        self._last_eval_prefs = np.asarray(self.eval_prefs, dtype=np.float64)

    def _eval_episode(self, preference: np.ndarray) -> np.ndarray:
        ep_obj = np.zeros(self.N, dtype=np.float64)
        self.env.seed(self.seed + 1_000_000 + int(preference[0] * 1e6))
        state = self.env.reset()
        cell = _state_to_cell(self.env, state)
        done = False
        while not done:
            a_macro = self._greedy_action(cell, preference)
            env_action = _macro_to_env_action(self.env, a_macro)
            state, reward, done, _ = self.env.step(env_action)
            cell = _state_to_cell(self.env, state)
            ep_obj += np.asarray(reward, dtype=np.float64)
        return ep_obj

    def _populate_final_buffers(self) -> None:
        if not hasattr(self, "_last_eval_objs"):
            self.final_ep_objs = np.zeros((1, self.N), dtype=np.float64)
            self.final_ep_prefs = np.full((1, self.N), 1.0 / self.N, dtype=np.float64)
            self.final_obj_means = np.zeros(self.N, dtype=np.float64)
            self.final_obj_stds = np.zeros(self.N, dtype=np.float64)
            return
        self.final_ep_objs = self._last_eval_objs
        self.final_ep_prefs = self._last_eval_prefs
        self.final_obj_means = self._last_eval_objs.mean(axis=0)
        self.final_obj_stds = self._last_eval_objs.std(axis=0)

    def policy_fn(self, obs: np.ndarray, preference: np.ndarray) -> np.ndarray:
        cell = _state_to_cell(self.env, obs)
        a_macro = self._greedy_action(cell, preference)
        return _macro_to_env_action(self.env, a_macro)
