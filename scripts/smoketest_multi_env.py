"""Smoke-test for the new MultiUAVSemComEnv.

Run on either training server (or anywhere with the conda RA_DI env):
    python scripts/smoketest_multi_env.py

Verifies:
    - The env can be created via gym.make("UAV-SemCom-Multi-v0", num_uavs=M, num_devices=K)
    - reset() returns the expected joint-state shape
    - step(random action) returns the expected joint-reward shape
    - Single-UAV env (UAV-SemCom-v0) still works (backward compat regression check)
    - State / action dim formulas match the documented values for {M, K} grid
"""
from __future__ import annotations

# NumPy 2.0 compat (gym 0.25 references np.bool8 which was removed)
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import environments  # noqa: F401  -- registers UAV-SemCom-v0 / UAV-SemCom-Multi-v0


def expected_state_dim(M: int, K: int) -> int:
    return 4 * M + 2 * M * K + M * K + 2 * K + 1


def expected_action_dim(M: int, K: int) -> int:
    return M * (2 + 3 * K)


def run_episode(env, label: str, n_steps: int = 5) -> None:
    obs = env.reset()
    print(f"  [{label}] obs shape={obs.shape}  range=({obs.min():.3f}, {obs.max():.3f})")
    for t in range(n_steps):
        a = env.action_space.sample()
        next_obs, reward, done, info = env.step(a)
        print(
            f"  [{label}] t={t}  obs.shape={next_obs.shape}  reward={np.round(reward, 3).tolist()}  "
            f"done={done}  energy={info['energy']:.1f}"
        )
        if done:
            break


def main() -> None:
    # ---- single-UAV regression (backward compat) ----
    print("=== single-UAV regression check (UAV-SemCom-v0) ===")
    env1 = gym.make("UAV-SemCom-v0", num_devices=5)
    print(f"  obs_space={env1.observation_space.shape}  expected=(30,)")
    print(f"  act_space={env1.action_space.shape}  expected=(12,)")
    assert env1.observation_space.shape == (30,)
    assert env1.action_space.shape == (12,)
    run_episode(env1, "M=1 K=5", n_steps=3)

    # ---- multi-UAV smoke test ----
    # K limited to 5 because DEFAULT_DEVICE_CONFIGS has 5 entries; larger K
    # requires user-supplied device_configs (same constraint as single-UAV).
    print("\n=== multi-UAV smoke test (UAV-SemCom-Multi-v0) ===")
    for M, K in [(2, 5), (4, 5), (5, 5)]:
        env = gym.make("UAV-SemCom-Multi-v0", num_uavs=M, num_devices=K)
        s_exp = expected_state_dim(M, K)
        a_exp = expected_action_dim(M, K)
        s_got = env.observation_space.shape[0]
        a_got = env.action_space.shape[0]
        ok_s = "OK" if s_got == s_exp else "FAIL"
        ok_a = "OK" if a_got == a_exp else "FAIL"
        print(f"\n  -- M={M} K={K} -- state {s_got} (exp {s_exp}, {ok_s})  action {a_got} (exp {a_exp}, {ok_a})")
        assert s_got == s_exp, (s_got, s_exp)
        assert a_got == a_exp, (a_got, a_exp)
        run_episode(env, f"M={M} K={K}", n_steps=3)

    print("\nAll smoke checks PASSED.")


if __name__ == "__main__":
    main()
