"""Smoke test for the C-MORL stub (Issue #8 PR-E).

Confirms the stub registers cleanly, is constructable through the runner,
and that ``train()`` raises NotImplementedError with a useful message
pointing the reader at the porting plan. Intentionally narrow — the full
training loop is not exercised because it doesn't exist yet.

Usage:
    PYTHONPATH=. python scripts/smoketest_cmorl_stub.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    print("=== step 1: registration ===")
    from baselines import get_baseline, list_baselines
    names = list_baselines()
    assert "c-morl" in names, f"c-morl missing from registry: {names}"
    cls = get_baseline("c-morl")
    print(f"  registry returned: {cls.__name__}")

    print("\n=== step 2: construction ===")
    import gym
    import environments  # noqa: F401
    env = gym.make("UAV-SemCom-v0", num_devices=5, max_episode_steps=200)
    env.seed(0)
    bl = cls(env=env, log_dir=tempfile.mkdtemp(), seed=0, method_kwargs={})
    print(f"  constructed: {bl.name}")
    assert bl.is_constrained, "C-MORL should advertise itself as constrained"
    print(f"  is_constrained: {bl.is_constrained}")

    print("\n=== step 3: train() raises NotImplementedError with the right pointer ===")
    try:
        bl.train(num_steps=10, eval_interval=5)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as e:
        msg = str(e)
        assert "porting plan" in msg or "DESIGN-baselines.md" in msg or "67473b5" in msg, (
            f"NotImplementedError message should reference porting plan: {msg!r}"
        )
        print(f"  OK: train() raised with pointer to porting plan")
        print(f"  message: {msg}")

    print("\n=== step 4: policy_fn returns valid action shape ===")
    obs = env.reset()
    pref = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    action = bl.policy_fn(obs, pref)
    assert action.shape == env.action_space.shape, (
        f"policy_fn shape: {action.shape} vs {env.action_space.shape}"
    )
    assert np.all(action >= env.action_space.low - 1e-6), "action below low bound"
    assert np.all(action <= env.action_space.high + 1e-6), "action above high bound"
    print(f"  OK: policy_fn returns shape={action.shape}, in bounds")

    print("\nAll C-MORL stub smoke checks passed.")


if __name__ == "__main__":
    main()
