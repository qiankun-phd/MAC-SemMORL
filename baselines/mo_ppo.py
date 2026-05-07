"""MO-PPO baseline (Issue #8 PR-C).

Vanilla weighted-sum scalarized PPO with a preference-conditioned policy and a
vector critic (one head per objective). This is the "vanilla MORL" comparison
point every TWC reviewer expects: standard PPO machinery, no COR, no OADM, no
constraint handler — only the preference vector and a multi-headed value
function distinguish it from single-objective PPO.

Default scalarisation: weighted-sum (DESIGN-baselines.md §7 Q2 default). The
Tchebycheff variant is left as a deferred ablation.

Architectural choices (all standard PPO knobs unless noted):

    Actor:  MLP(state ⊕ preference) → mean, log_std (state-independent log_std,
            shared parameter, initialised to log(0.5)). Tanh squashing on
            sampled action.
    Critic: MLP(state ⊕ preference) → N scalar values. The scalarised
            advantage that drives the policy update is `w^T A_vec`.
    Buffer: on-policy rollout buffer, GAE(λ=0.95) on each per-objective
            advantage, clipped surrogate objective.
    Update: minibatch SGD over the rollout for K epochs.

Eval:
    Every `eval_interval` env steps, evaluate the deterministic mean policy at
    each of the 56 fixed preferences (the existing
    `generate_w_batch_test(N=4, step=0.2)` grid). Collect the 4-D episode return
    per preference, compute hypervolume / utility / sparsity via the existing
    `evluate_Hv_UT_and_spa` helper from `agent.py`. This matches the
    SemMORL eval protocol exactly.
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
from torch.distributions import Normal

# Re-implementations of the SemMORL eval helpers; dependency-light so we
# don't pull tensorboard/visdom/rltorch through the heavy agent.py.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from ._morl_eval import evluate_Hv_UT_and_spa, generate_w_batch_test

from .base import Baseline
from .registry import register_baseline


def mlp(sizes, act=nn.Tanh, output_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif output_act is not None:
            layers.append(output_act())
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """Preference-conditioned Gaussian policy with state-independent log_std."""

    def __init__(self, state_dim: int, pref_dim: int, action_dim: int,
                 hidden=(64, 64)):
        super().__init__()
        self.body = mlp([state_dim + pref_dim, *hidden, action_dim])
        # Shared log_std parameter — standard PPO convention.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(0.5))))

    def forward(self, state: torch.Tensor, pref: torch.Tensor):
        mean = self.body(torch.cat([state, pref], dim=-1))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, state: torch.Tensor, pref: torch.Tensor, deterministic: bool = False):
        dist = self.forward(state, pref)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        # Tanh squash to [-1, 1] then env's max_action handled outside.
        squashed = torch.tanh(action)
        # log_prob with tanh correction: log p(u) - sum log(1 - tanh(u)^2 + 1e-6)
        logp = dist.log_prob(action).sum(-1) - torch.log(
            1 - squashed.pow(2) + 1e-6
        ).sum(-1)
        return squashed, logp, action  # raw action retained for re-evaluation

    def evaluate(self, state, pref, raw_action):
        dist = self.forward(state, pref)
        squashed = torch.tanh(raw_action)
        logp = dist.log_prob(raw_action).sum(-1) - torch.log(
            1 - squashed.pow(2) + 1e-6
        ).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy


class PPOVectorCritic(nn.Module):
    """One scalar value head per objective. Output shape (B, N)."""

    def __init__(self, state_dim: int, pref_dim: int, N: int, hidden=(64, 64)):
        super().__init__()
        self.body = mlp([state_dim + pref_dim, *hidden, N])

    def forward(self, state, pref):
        return self.body(torch.cat([state, pref], dim=-1))


class RolloutBuffer:
    """Fixed-size on-policy buffer. Per-objective advantages computed via GAE."""

    def __init__(self, T: int, state_dim: int, action_dim: int,
                 pref_dim: int, N: int, gamma: float, gae_lambda: float):
        self.T = T
        self.N = N
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = np.zeros((T, state_dim), dtype=np.float32)
        self.prefs = np.zeros((T, pref_dim), dtype=np.float32)
        self.raw_actions = np.zeros((T, action_dim), dtype=np.float32)
        self.squashed_actions = np.zeros((T, action_dim), dtype=np.float32)
        self.logp = np.zeros(T, dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)
        self.ptr = 0

    def add(self, state, pref, raw_action, squashed_action, logp, reward,
            value, done):
        i = self.ptr
        self.states[i] = state
        self.prefs[i] = pref
        self.raw_actions[i] = raw_action
        self.squashed_actions[i] = squashed_action
        self.logp[i] = logp
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def finish(self, last_value: np.ndarray):
        """GAE on each objective independently. last_value shape (N,)."""
        T, N = self.ptr, self.N
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            next_value = (
                last_value if t == T - 1 else self.values[t + 1]
            )
            next_nonterminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t] + self.gamma * next_value * next_nonterminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        self.advantages[:T] = adv
        self.returns[:T] = adv + self.values[:T]

    def reset(self):
        self.ptr = 0


def _sample_pref(rng: np.random.RandomState, N: int) -> np.ndarray:
    p = rng.rand(N).astype(np.float32)
    p /= p.sum()
    return p


@register_baseline("mo-ppo")
class MoPPOBaseline(Baseline):
    """Weighted-sum scalarized PPO with preference-conditioned policy."""

    name = "mo-ppo"

    def __init__(self, env, log_dir: str, seed: int,
                 method_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(env, log_dir, seed, method_kwargs)
        mk = self.method_kwargs

        self.gamma = float(mk.get("gamma", 0.99))
        self.gae_lambda = float(mk.get("gae_lambda", 0.95))
        self.lr = float(mk.get("lr", 3e-4))
        self.clip_eps = float(mk.get("clip_eps", 0.2))
        self.entropy_coef = float(mk.get("entropy_coef", 0.0))
        self.value_coef = float(mk.get("value_coef", 0.5))
        self.rollout_T = int(mk.get("rollout_T", 2048))
        self.minibatch_size = int(mk.get("minibatch_size", 64))
        self.update_epochs = int(mk.get("update_epochs", 10))
        self.hidden = tuple(mk.get("hidden", (64, 64)))
        self.eval_episodes_per_pref = int(mk.get("eval_episodes_per_pref", 1))
        self.device = torch.device(mk.get("device", "cpu"))

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.pref_dim = int(getattr(env, "reward_num", 4))
        self.N = self.pref_dim
        self.max_action = env.action_space.high

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.actor = PPOActor(
            self.state_dim, self.pref_dim, self.action_dim, hidden=self.hidden
        ).to(self.device)
        self.critic = PPOVectorCritic(
            self.state_dim, self.pref_dim, self.N, hidden=self.hidden
        ).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
        )
        self.rng = np.random.RandomState(self.seed)
        # Pre-generate the preference grid used at eval time so HV is comparable
        # to the SemMORL eval protocol.
        step_map = {3: 0.05, 4: 0.2, 5: 0.25}
        step = step_map.get(self.N, 0.2)
        self.eval_prefs = generate_w_batch_test(self.N, step_size=step)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, num_steps: int, eval_interval: int, our_wandb=None) -> None:
        T = self.rollout_T
        buf = RolloutBuffer(
            T=T, state_dim=self.state_dim, action_dim=self.action_dim,
            pref_dim=self.pref_dim, N=self.N,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        env = self.env
        env.seed(self.seed)
        state = env.reset()
        cur_pref = _sample_pref(self.rng, self.N)
        steps = 0
        t0 = time.time()
        next_eval = eval_interval

        while steps < num_steps:
            buf.reset()
            for _ in range(T):
                s_t = torch.as_tensor(state, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
                p_t = torch.as_tensor(cur_pref, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
                with torch.no_grad():
                    squashed, logp, raw = self.actor.act(s_t, p_t)
                    value = self.critic(s_t, p_t)
                a_squashed = squashed.cpu().numpy().reshape(-1)
                a_raw = raw.cpu().numpy().reshape(-1)
                action_env = a_squashed * self.max_action
                next_state, reward, done, _ = env.step(action_env)
                buf.add(
                    state=state, pref=cur_pref,
                    raw_action=a_raw, squashed_action=a_squashed,
                    logp=float(logp.cpu().item()),
                    reward=np.asarray(reward, dtype=np.float32),
                    value=value.cpu().numpy().reshape(-1),
                    done=float(done),
                )
                state = next_state
                steps += 1
                if done:
                    state = env.reset()
                    cur_pref = _sample_pref(self.rng, self.N)
                if steps >= num_steps:
                    break
                if steps >= next_eval:
                    self._evaluate(steps)
                    next_eval += eval_interval

            # Bootstrap last value for GAE
            with torch.no_grad():
                last_v = self.critic(
                    torch.as_tensor(state, dtype=torch.float32,
                                    device=self.device).unsqueeze(0),
                    torch.as_tensor(cur_pref, dtype=torch.float32,
                                    device=self.device).unsqueeze(0),
                ).cpu().numpy().reshape(-1)
            buf.finish(last_v)
            self._update(buf)

        # Final eval if we haven't already covered the last interval.
        if not self.eval_steps or self.eval_steps[-1] < num_steps:
            self._evaluate(num_steps)

        # Populate final-state buffers from the most recent eval episodes.
        self._populate_final_buffers()
        self.wallclock_seconds = time.time() - t0

    def _update(self, buf: RolloutBuffer) -> None:
        T = buf.ptr
        idxs = np.arange(T)
        states = torch.as_tensor(buf.states[:T], device=self.device)
        prefs = torch.as_tensor(buf.prefs[:T], device=self.device)
        raw_act = torch.as_tensor(buf.raw_actions[:T], device=self.device)
        old_logp = torch.as_tensor(buf.logp[:T], device=self.device)
        adv_vec = torch.as_tensor(buf.advantages[:T], device=self.device)
        ret_vec = torch.as_tensor(buf.returns[:T], device=self.device)
        # Scalarised advantage drives the policy update.
        adv_scalar = (adv_vec * prefs).sum(-1)
        # Per-rollout normalisation (standard PPO trick).
        adv_scalar = (adv_scalar - adv_scalar.mean()) / (adv_scalar.std() + 1e-8)

        for _ in range(self.update_epochs):
            self.rng.shuffle(idxs)
            for start in range(0, T, self.minibatch_size):
                mb = idxs[start:start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                logp, entropy = self.actor.evaluate(
                    states[mb_t], prefs[mb_t], raw_act[mb_t]
                )
                ratio = (logp - old_logp[mb_t]).exp()
                surr1 = ratio * adv_scalar[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) \
                        * adv_scalar[mb_t]
                actor_loss = -torch.min(surr1, surr2).mean()
                values = self.critic(states[mb_t], prefs[mb_t])
                critic_loss = F.mse_loss(values, ret_vec[mb_t])
                loss = (actor_loss + self.value_coef * critic_loss
                        - self.entropy_coef * entropy.mean())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    def _evaluate(self, step: int) -> None:
        objs = []
        for pref in self.eval_prefs:
            ep_obj = self._eval_episode(pref)
            objs.append(ep_obj)
        objs = np.asarray(objs, dtype=np.float64)
        hv, sparsity, ut = evluate_Hv_UT_and_spa(self.N, objs, self.eval_prefs)
        self._record_eval(step=step, hv=hv, ut=ut, sparsity=sparsity)
        self._last_eval_objs = objs
        self._last_eval_prefs = np.asarray(self.eval_prefs, dtype=np.float64)

    def _eval_episode(self, preference: np.ndarray) -> np.ndarray:
        """Run one deterministic episode at the given preference; return the
        4-D summed reward."""
        ep_obj = np.zeros(self.N, dtype=np.float64)
        # Use a separate seed offset to avoid eval-train state collision.
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
        """Set final_ep_objs / final_ep_prefs / final_obj_means / final_obj_stds
        from the most recent eval. These are the npz schema fields the runner
        will dump (DESIGN-baselines.md §3)."""
        if not hasattr(self, "_last_eval_objs"):
            self.final_ep_objs = np.zeros((1, self.N), dtype=np.float64)
            self.final_ep_prefs = np.full(
                (1, self.N), 1.0 / self.N, dtype=np.float64
            )
            self.final_obj_means = np.zeros(self.N, dtype=np.float64)
            self.final_obj_stds = np.zeros(self.N, dtype=np.float64)
            return
        self.final_ep_objs = self._last_eval_objs
        self.final_ep_prefs = self._last_eval_prefs
        self.final_obj_means = self._last_eval_objs.mean(axis=0)
        self.final_obj_stds = self._last_eval_objs.std(axis=0)

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------
    def policy_fn(self, obs: np.ndarray, preference: np.ndarray) -> np.ndarray:
        s_t = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        p_t = torch.as_tensor(preference, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            squashed, _, _ = self.actor.act(s_t, p_t, deterministic=True)
        return squashed.cpu().numpy().reshape(-1) * self.max_action
