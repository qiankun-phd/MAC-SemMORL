"""Diagnose per-step reward distributions across different policies."""
import numpy as np
import gym
import sys

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from environments import *
from baselines import fixed_trajectory_greedy_power, greedy_aosi, random_policy

env = gym.make("UAV-SemCom-v0", num_devices=5, max_episode_steps=200)

# Also compare with MuJoCo reference
print("=" * 72)
print("MuJoCo Ant reference: per-step reward ≈ [-2, 5] per objective")
print("Target: our per-step reward should be in similar range [0.5, 4.5]")
print("=" * 72)

policies = {
    "Random":     random_policy,
    "FT-GP":      fixed_trajectory_greedy_power,
    "Greedy AoSI": greedy_aosi,
}

for pol_name, pol_fn in policies.items():
    all_steps = []
    for ep in range(20):
        env.seed(42 + ep)
        obs = env.reset()
        done = False
        while not done:
            action = pol_fn(obs, env)
            obs, reward, done, info = env.step(action)
            all_steps.append(reward)

    steps = np.array(all_steps)
    print(f"\n{'='*40}")
    print(f"Policy: {pol_name}  ({len(steps)} steps)")
    print(f"{'='*40}")
    print(f"         {'Fidelity':>12} {'Freshness':>12} {'Energy':>12}")
    print(f"  mean:  {steps[:,0].mean():>12.3f} {steps[:,1].mean():>12.3f} {steps[:,2].mean():>12.3f}")
    print(f"  std:   {steps[:,0].std():>12.3f} {steps[:,1].std():>12.3f} {steps[:,2].std():>12.3f}")
    print(f"  min:   {steps[:,0].min():>12.3f} {steps[:,1].min():>12.3f} {steps[:,2].min():>12.3f}")
    print(f"  25%:   {np.percentile(steps[:,0],25):>12.3f} {np.percentile(steps[:,1],25):>12.3f} {np.percentile(steps[:,2],25):>12.3f}")
    print(f"  50%:   {np.percentile(steps[:,0],50):>12.3f} {np.percentile(steps[:,1],50):>12.3f} {np.percentile(steps[:,2],50):>12.3f}")
    print(f"  75%:   {np.percentile(steps[:,0],75):>12.3f} {np.percentile(steps[:,1],75):>12.3f} {np.percentile(steps[:,2],75):>12.3f}")
    print(f"  max:   {steps[:,0].max():>12.3f} {steps[:,1].max():>12.3f} {steps[:,2].max():>12.3f}")

    # Q-value estimate (gamma=0.99, ~200 steps horizon)
    ep_sum = steps.reshape(20, 200, 3).sum(axis=1)
    q_est = ep_sum.mean(axis=0)
    print(f"\n  Episode cumulative (≈ undiscounted Q):")
    print(f"         {q_est[0]:>12.1f} {q_est[1]:>12.1f} {q_est[2]:>12.1f}")
    print(f"  Ratio to max obj: {q_est[0]/q_est.max():>8.2f}   {q_est[1]/q_est.max():>8.2f}   {q_est[2]/q_est.max():>8.2f}")

print("\n" + "=" * 72)
print("DIAGNOSIS:")
print("- If any objective's per-step mean is >2x another's → imbalance risk")
print("- If ratio-to-max < 0.5 for any objective → Q-network gradient bias")
print("=" * 72)
