"""Smoke test for the PSL-MORL stub (Issue #8 PR-F).

Same shape as the C-MORL stub smoke test. Confirms registration, construction,
that ``train()`` raises with a useful pointer, and that ``policy_fn`` returns
a valid action.

Usage:
    PYTHONPATH=. python scripts/smoketest_psl_morl_stub.py
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
    assert "psl-morl" in names, f"psl-morl missing from registry: {names}"
    print(f"  registry: {names}")
    cls = get_baseline("psl-morl")
    assert cls.__name__ == "PSLMORLBaseline"
    print(f"  OK: psl-morl -> {cls.__name__}")

    print("\n=== step 2: construction ===")
    import gym
    import environments  # noqa: F401
    env = gym.make("UAV-SemCom-v0", num_devices=5, max_episode_steps=200)
    env.seed(0)
    bl = cls(env=env, log_dir=tempfile.mkdtemp(), seed=0, method_kwargs={})
    print(f"  OK: constructed {bl.name}")
    assert np.all(np.isnan(bl.c_violation_rates)), \
        "PSL-MORL is unconstrained — c_violation_rates should be NaN"

    print("\n=== step 3: train() raises with plan pointer ===")
    try:
        bl.train(num_steps=10, eval_interval=5)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as e:
        msg = str(e)
        assert ("re-implementation" in msg or "no public repo" in msg
                or "DESIGN-baselines.md" in msg), (
            f"NotImplementedError should reference porting plan: {msg!r}"
        )
        print(f"  OK: train() raised with plan pointer")
        print(f"  message: {msg}")

    print("\n=== step 4: policy_fn returns valid action ===")
    obs = env.reset()
    pref = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    action = bl.policy_fn(obs, pref)
    assert action.shape == env.action_space.shape, (
        f"shape: {action.shape} vs {env.action_space.shape}"
    )
    assert np.all(action >= env.action_space.low - 1e-6)
    assert np.all(action <= env.action_space.high + 1e-6)
    print(f"  OK: shape={action.shape}, in bounds")

    print("\nAll PSL-MORL stub smoke checks passed.")


if __name__ == "__main__":
    main()
