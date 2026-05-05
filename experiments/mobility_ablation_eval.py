#!/usr/bin/env python3
"""Quick comparison: fixed IoT vs slow mobility on the same policies (no RL training)."""
import os
import sys
import numpy as np
import gym

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environments import *  # noqa: F401 — registers UAV-SemCom-v0
from baselines import greedy_aosi, random_policy, evaluate_policy

N_EP = 40
SEED = 0
DEVICE_SPEED = 0.5


def make_env(mobility: str):
    kw = dict(num_devices=5, max_episode_steps=200)
    if mobility == "none":
        return gym.make("UAV-SemCom-v0", **kw)
    return gym.make(
        "UAV-SemCom-v0",
        **kw,
        device_mobility=mobility,
        device_speed=DEVICE_SPEED,
    )


def summarize(name, R):
    """R: (n_ep, 4)"""
    s = R.sum(axis=1)
    return {
        "policy": name,
        "sum_mean": float(s.mean()),
        "sum_std": float(s.std()),
        "per_obj_mean": R.mean(axis=0).tolist(),
    }


def main():
    rows = []
    for mobility in ("none", "line", "drift"):
        env = make_env(mobility)
        Rg = evaluate_policy(env, greedy_aosi, n_episodes=N_EP, seed=SEED)
        Rr = evaluate_policy(env, random_policy, n_episodes=N_EP, seed=SEED)
        rows.append({"mobility": mobility, **summarize("Greedy AoSI", Rg)})
        rows.append({"mobility": mobility, **summarize("Random", Rr)})

    print(f"Episodes per cell: {N_EP}, seed base {SEED}, device_speed={DEVICE_SPEED} m/s (when mobility on)\n")
    print(f"{'mobility':<8} {'policy':<12} {'sum_reward_mean':>16} {'sum_reward_std':>16}")
    for r in rows:
        print(
            f"{r['mobility']:<8} {r['policy']:<12} {r['sum_mean']:>16.2f} {r['sum_std']:>16.2f}"
        )

    # Relative change vs none for Greedy AoSI
    g_none = rows[0]["sum_mean"]
    g_line = rows[2]["sum_mean"]
    g_drift = rows[4]["sum_mean"]
    print("\nGreedy AoSI total return vs fixed (none):")
    print(f"  line:  {(g_line - g_none) / abs(g_none) * 100:+.2f} %")
    print(f"  drift: {(g_drift - g_none) / abs(g_none) * 100:+.2f} %")


if __name__ == "__main__":
    main()
