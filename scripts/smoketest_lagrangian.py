"""Smoke test for the Lagrangian constrained-MORL path (Issue #6 PR-B).

Runs ~3K env steps with `use_lagrangian=True`, verifies:
    - Agent constructs without errors when constrained kwargs are passed.
    - `info["max_aosi"]` is exposed by both single- and multi-UAV envs.
    - `_compute_step_costs` produces a 3-vector with the documented sign convention.
    - `_shape_reward_lagrangian` returns a 4-D float32 array (Q-net shape unchanged).
    - The dual update fires after `start_steps + dual_update_every` and the
      `lambdas` array changes (or stays at zero if no violations — both are valid).
    - No NaN appears in the EMA cost or lambdas vector.

Usage:
    PYTHONPATH=. python scripts/smoketest_lagrangian.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch
import gym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import environments  # noqa: F401
from agent import SacAgent


def assert_finite(name: str, arr) -> None:
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{name} contains non-finite values: {a}")


def smoke_single_uav() -> None:
    print("=== single-UAV (UAV-SemCom-v0) smoke ===")
    env = gym.make("UAV-SemCom-v0", num_devices=5)
    env.seed(0)
    obs = env.reset()
    a = env.action_space.sample()
    _, _, _, info = env.step(a)
    assert "max_aosi" in info, "single-UAV env must expose max_aosi"
    print(f"  info.max_aosi = {info['max_aosi']:.3f}  (epsilon source for c_1)")
    print(f"  info.energy   = {info['energy']:.2f}    (c_2 input)")
    print(f"  info.service_rate = {info['service_rate']:.3f}  (c_3 input)")


def smoke_multi_uav() -> None:
    print("=== multi-UAV (UAV-SemCom-Multi-v0) smoke ===")
    env = gym.make("UAV-SemCom-Multi-v0", num_uavs=2, num_devices=5)
    env.seed(0)
    env.reset()
    a = env.action_space.sample()
    _, _, _, info = env.step(a)
    assert "max_aosi" in info, "multi-UAV env must expose max_aosi"
    print(f"  info.max_aosi = {info['max_aosi']:.3f}")


def smoke_agent_lagrangian(num_steps: int = 3000) -> None:
    print(f"=== Lagrangian agent smoke ({num_steps} env steps) ===")
    env = gym.make("UAV-SemCom-v0", num_devices=5)
    env.seed(0)

    # Tight constraints to force non-trivial duals (defaults are too loose to
    # show movement in 3K steps).
    cfg = dict(
        env_id="UAV-SemCom-v0",
        env=env,
        log_dir="logs/smoketest_lagrangian",
        num_steps=num_steps + 200,
        batch_size=64,
        memory_size=10000,
        start_steps=500,
        eval_interval=10000,
        cuda=False,
        seed=0,
        Use_Policy_Preference=True,
        Use_Critic_Preference=True,
        train_with_fixed_preference=False,
        latent_dim=50,
        Policy_use_latent=True,
        Policy_use_s=True,
        Policy_use_w=True,
        Critic_use_s=True,
        Critic_use_a=True,
        Critic_use_both=True,
        use_avg=True,
        regular_alpha=0.5,
        regular_bar=0.25,
        warm_steps=10**9,
        # Constrained MORL knobs:
        use_lagrangian=True,
        constraint_handler="lagrangian",
        constraint_thresholds=dict(
            A_max=1.5,        # tighter than default 3.0 → expect c_1 to fire
            epsilon_aosi=0.05,
            E_total_kJ=15.0,  # tighter than default 30 → expect c_2 to fire
            rho_min=0.95,     # tighter than default 0.7 → expect c_3 to fire
            service_window=20,
        ),
        lambda_lr=1e-2,        # 10x default to see movement in 3K steps
        lambda_max=10.0,
        lambda_init=[0.0, 0.0, 0.0],
        dual_update_every=200,
        ema_decay=0.9,
    )
    agent = SacAgent(**cfg)

    # Patch wandb hook to a no-op so logging doesn't require a real run.
    agent.our_wandb = None

    # Sanity check 1: cost helper returns 3-vector with right signs.
    fake_info = {"max_aosi": 5.0, "energy": agent.E_total_per_step * 2,
                 "service_rate": 0.0}
    c_step = agent._compute_step_costs(fake_info)
    assert c_step.shape == (3,), f"c_step shape mismatch: {c_step.shape}"
    assert c_step[0] > 0, f"c_1 should fire when max_aosi=5 > A_max=1.5: got {c_step[0]}"
    assert c_step[1] > 0, f"c_2 should fire when energy is 2x budget: got {c_step[1]}"
    assert c_step[2] > 0, f"c_3 should fire when service_rate=0: got {c_step[2]}"
    print(f"  sanity c_step (max_aosi=5, e=2*budget, svc=0) = {c_step.round(4).tolist()}  (all > 0)")

    # Sanity check 2: shape function returns 4-D float32.
    fake_reward = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    agent.lambdas = np.array([1.0, 1.0, 1.0])
    shaped = agent._shape_reward_lagrangian(fake_reward, c_step)
    assert shaped.shape == (4,), f"shaped shape: {shaped.shape}"
    assert shaped.dtype == np.float32, f"shaped dtype: {shaped.dtype}"
    expected = 1.0 - float(np.dot(np.array([1.0, 1.0, 1.0]), c_step)) / 4.0
    assert np.allclose(shaped, expected, atol=1e-4), f"shaped {shaped} != expected {expected}"
    print(f"  sanity shaped reward (uniform attribution) ≈ {expected:.4f}  ({shaped.tolist()})")

    # Reset state for the actual rollout.
    agent.lambdas = np.array(cfg["lambda_init"], dtype=np.float64)
    agent.ema_costs = np.zeros(3)
    agent.svc_window.clear()
    agent._last_dual_update_step = 0
    agent.steps = 0

    # Run a short rollout via the standard evluate path. We don't need the
    # training updates to fire — start_steps=500 means random actions for the
    # first 500 steps then policy actions, both go through evluate.
    print(f"  rolling out (start_steps={cfg['start_steps']}, dual_update_every={cfg['dual_update_every']})...")
    pref = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    eps_done = 0
    while agent.steps < num_steps:
        agent.evluate(pref, agent.policy, RL_agent=False)
        eps_done += 1

    # Final assertions.
    assert_finite("agent.lambdas", agent.lambdas)
    assert_finite("agent.ema_costs", agent.ema_costs)
    assert agent._last_dual_update_step >= cfg["start_steps"], (
        "dual update should have fired at least once after start_steps"
    )
    print(f"  episodes run: {eps_done}, total steps: {agent.steps}")
    print(f"  final lambdas       = {agent.lambdas.round(4).tolist()}")
    print(f"  final ema_costs     = {agent.ema_costs.round(4).tolist()}")
    print(f"  last dual update at step {agent._last_dual_update_step}")
    print(f"  svc_window len      = {len(agent.svc_window)}/{cfg['constraint_thresholds']['service_window']}")
    # With tightened thresholds and lr=1e-2, we expect at least one lambda > 0
    # after 3K steps. Soft check: warn but don't fail (env stochasticity).
    if not np.any(agent.lambdas > 1e-3):
        print("  WARN: all lambdas still ~0 — constraints may have stayed feasible "
              "by chance for this seed. Re-run with longer horizon to confirm.")
    else:
        print(f"  OK: at least one lambda has moved off zero — dual loop alive")


def smoke_default_off_unchanged() -> None:
    print("=== default-off (use_lagrangian=False) — baseline must be untouched ===")
    env = gym.make("UAV-SemCom-v0", num_devices=5)
    env.seed(0)
    cfg = dict(
        env_id="UAV-SemCom-v0", env=env, log_dir="logs/smoketest_off",
        num_steps=200, batch_size=64, memory_size=1000,
        start_steps=10, eval_interval=10000, cuda=False, seed=0,
        Use_Policy_Preference=True, Use_Critic_Preference=True,
        train_with_fixed_preference=False, latent_dim=50,
        Policy_use_latent=True, Policy_use_s=True, Policy_use_w=True,
        Critic_use_s=True, Critic_use_a=True, Critic_use_both=True,
        use_avg=True, regular_alpha=0.5, regular_bar=0.25,
        warm_steps=10**9,
        # Lagrangian explicitly off (the default, but pass through for the test):
        use_lagrangian=False,
    )
    agent = SacAgent(**cfg)
    agent.our_wandb = None
    pref = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    agent.evluate(pref, agent.policy, RL_agent=False)
    assert agent._last_dual_update_step == 0, (
        "dual update must NOT fire when use_lagrangian=False"
    )
    print(f"  OK: dual update count = 0, lambdas untouched ({agent.lambdas.tolist()})")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    smoke_single_uav()
    smoke_multi_uav()
    smoke_default_off_unchanged()
    smoke_agent_lagrangian(num_steps=3000)
    print("\nAll Lagrangian smoke checks passed.")
