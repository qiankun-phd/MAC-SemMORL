"""Pareto Policy Gradient baseline (Issue #8 PR-D).

Vanilla REINFORCE-style policy gradient with preference-conditioned policy
and a vector value baseline. Differs from MO-PPO by *not* clipping the
surrogate objective and *not* doing GAE — the simplest scalarised PG that
still trains.

This is the "no fancy machinery" lower bound that reviewers expect: if a
proposed method only beats Pareto-PG by a small margin, the proposed
machinery probably isn't doing much. If it dominates Pareto-PG by a large
margin, the machinery matters.

Algorithm per episode rollout:
    1. Sample preference w from the simplex.
    2. Run an episode, collecting (s_t, a_t, r_t_vec) tuples.
    3. Compute per-objective discounted returns G_t_vec.
    4. Vector-baseline subtraction: A_t_vec = G_t_vec - V(s_t, w)_vec.
    5. Scalarise: A_t = w · A_t_vec.
    6. Policy gradient: ∇ log π(a|s, w) · A_t.
    7. Critic regression on G_t_vec (per objective).

Default scalarisation: weighted-sum (DESIGN-baselines.md §7 Q2 default).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ._morl_eval import evluate_Hv_UT_and_spa, generate_w_batch_test
from .base import Baseline
from .registry import register_baseline
from .mo_ppo import PPOActor, PPOVectorCritic, _sample_pref


@register_baseline("pareto-pg")
class ParetoPGBaseline(Baseline):
    """REINFORCE-style scalarised PG with preference-conditioned policy."""

    name = "pareto-pg"

    def __init__(self, env, log_dir: str, seed: int,
                 method_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(env, log_dir, seed, method_kwargs)
        mk = self.method_kwargs

        self.gamma = float(mk.get("gamma", 0.99))
        self.lr_actor = float(mk.get("lr_actor", 3e-4))
        self.lr_critic = float(mk.get("lr_critic", 1e-3))
        self.entropy_coef = float(mk.get("entropy_coef", 0.0))
        self.episodes_per_update = int(mk.get("episodes_per_update", 4))
        self.hidden = tuple(mk.get("hidden", (64, 64)))
        self.device = torch.device(mk.get("device", "cpu"))

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.N = int(getattr(env, "reward_num", 4))
        self.max_action = env.action_space.high

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

        self.actor = PPOActor(self.state_dim, self.N, self.action_dim,
                              hidden=self.hidden).to(self.device)
        self.critic = PPOVectorCritic(self.state_dim, self.N, self.N,
                                      hidden=self.hidden).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        step_map = {3: 0.05, 4: 0.2, 5: 0.25}
        self.eval_prefs = generate_w_batch_test(
            self.N, step_size=step_map.get(self.N, 0.2)
        )

    # ------------------------------------------------------------------
    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        env = self.env
        env.seed(self.seed)
        steps = 0
        t0 = time.time()
        next_eval = eval_interval

        while steps < num_steps:
            # Collect a batch of episodes_per_update episodes at a sampled
            # preference each, then a single PG update.
            batch_states, batch_prefs, batch_raw_acts = [], [], []
            batch_returns_vec = []
            for _ in range(self.episodes_per_update):
                pref = _sample_pref(self.rng, self.N)
                ep_states, ep_raw_acts, ep_rewards = [], [], []
                state = env.reset()
                done = False
                while not done:
                    s_t = torch.as_tensor(state, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                    p_t = torch.as_tensor(pref, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        squashed, _, raw = self.actor.act(s_t, p_t)
                    a_squashed = squashed.cpu().numpy().reshape(-1)
                    a_raw = raw.cpu().numpy().reshape(-1)
                    next_state, reward, done, _ = env.step(a_squashed * self.max_action)
                    ep_states.append(state)
                    ep_raw_acts.append(a_raw)
                    ep_rewards.append(np.asarray(reward, dtype=np.float32))
                    state = next_state
                    steps += 1
                    if steps >= num_steps:
                        done = True
                    if steps >= next_eval:
                        self._evaluate(steps)
                        next_eval += eval_interval
                # Per-objective discounted returns.
                T = len(ep_rewards)
                G = np.zeros((T, self.N), dtype=np.float32)
                running = np.zeros(self.N, dtype=np.float32)
                for t in reversed(range(T)):
                    running = ep_rewards[t] + self.gamma * running
                    G[t] = running
                batch_states.append(np.asarray(ep_states, dtype=np.float32))
                batch_prefs.append(np.tile(pref, (T, 1)))
                batch_raw_acts.append(np.asarray(ep_raw_acts, dtype=np.float32))
                batch_returns_vec.append(G)
                if steps >= num_steps:
                    break
            self._update(batch_states, batch_prefs, batch_raw_acts, batch_returns_vec)

        if not self.eval_steps or self.eval_steps[-1] < num_steps:
            self._evaluate(num_steps)
        self._populate_final_buffers()
        self.wallclock_seconds = time.time() - t0

    def _update(self, batch_states, batch_prefs, batch_raw_acts, batch_returns_vec) -> None:
        states = torch.as_tensor(np.concatenate(batch_states), device=self.device)
        prefs = torch.as_tensor(np.concatenate(batch_prefs), device=self.device)
        raw_acts = torch.as_tensor(np.concatenate(batch_raw_acts), device=self.device)
        returns_vec = torch.as_tensor(np.concatenate(batch_returns_vec), device=self.device)

        # Critic update on per-objective returns.
        values = self.critic(states, prefs)
        critic_loss = F.mse_loss(values, returns_vec)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # Actor update with vector baseline.
        with torch.no_grad():
            baselines = self.critic(states, prefs)
        adv_vec = returns_vec - baselines
        adv_scalar = (adv_vec * prefs).sum(-1)
        adv_scalar = (adv_scalar - adv_scalar.mean()) / (adv_scalar.std() + 1e-8)
        logp, entropy = self.actor.evaluate(states, prefs, raw_acts)
        actor_loss = -(logp * adv_scalar).mean() - self.entropy_coef * entropy.mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

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
        done = False
        with torch.no_grad():
            while not done:
                s_t = torch.as_tensor(state, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
                p_t = torch.as_tensor(preference, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
                squashed, _, _ = self.actor.act(s_t, p_t, deterministic=True)
                action_env = squashed.cpu().numpy().reshape(-1) * self.max_action
                state, reward, done, _ = self.env.step(action_env)
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
        s_t = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        p_t = torch.as_tensor(preference, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            squashed, _, _ = self.actor.act(s_t, p_t, deterministic=True)
        return squashed.cpu().numpy().reshape(-1) * self.max_action
