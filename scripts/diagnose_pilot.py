"""Diagnostic eval of a trained checkpoint — physical metrics from info dict.

Loads the saved policy + latent encoder for the given run and runs N
evaluation episodes at uniform preference. Reports per-episode total
energy (kJ), mean fidelity, mean AoSI, Jain index, service rate. The
goal is to compare against the conference-paper Table II numbers
(Fid 0.88, AoSI 1.06, Energy 21.96 kJ, Jain 0.98) to diagnose whether
the multi-UAV M=2 training is actually using both UAVs or whether the
energy reward normalisation is masking the cost.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# NumPy 2.0 compat for gym 0.25
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch
import gym

# Ensure local repo root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import environments  # noqa: F401  -- registers UAV-SemCom-v0 / UAV-SemCom-Multi-v0
from model import GaussianPolicy, Latent_Encoder


def build_networks(env, latent_dim: int = 50, use_avg: bool = True):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_num = env.reward_num

    encoder = Latent_Encoder(use_avg, obs_dim, action_dim, reward_num, latent_dim)
    policy_input = latent_dim + obs_dim + reward_num
    policy = GaussianPolicy(
        policy_input, action_dim, hidden_units=[128, 128],
        Use_Policy_Preference=True,
    )
    return encoder, policy


def diagnose(checkpoint_dir: str, num_uavs: int, num_devices: int,
             num_episodes: int, seed: int) -> None:
    if num_uavs == 1:
        env = gym.make("UAV-SemCom-v0", num_devices=num_devices)
    else:
        env = gym.make(
            "UAV-SemCom-Multi-v0",
            num_uavs=num_uavs, num_devices=num_devices,
        )
    env.seed(seed)

    encoder, policy = build_networks(env)
    encoder.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "encoder_final.pkl"), map_location="cpu"
    ))
    policy.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "policy_final.pkl"), map_location="cpu"
    ))
    encoder.eval()
    policy.eval()

    preference = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    max_action = env.action_space.high

    print(f"=== M={num_uavs}  K={num_devices}  episodes={num_episodes}  seed={seed}  pref=uniform ===")
    print(f"obs_dim={env.observation_space.shape[0]}  action_dim={env.action_space.shape[0]}")

    physical = {
        "weighted_avg_fidelity": [],
        "mean_aosi": [],
        "energy_per_step": [],   # joules/slot
        "energy_per_episode_kJ": [],
        "jain_fairness": [],
        "service_rate": [],
    }
    episode_rewards = []

    with torch.no_grad():
        for ep in range(num_episodes):
            state = env.reset()
            ep_metrics = {k: [] for k in physical if k != "energy_per_episode_kJ"}
            ep_reward = np.zeros(4)
            ep_energy_total = 0.0
            done = False

            while not done:
                s_t = torch.FloatTensor(state).unsqueeze(0)
                p_t = torch.FloatTensor(preference).unsqueeze(0)
                z = encoder.get_latent_features(s_t)
                inp = torch.cat([z, s_t, p_t], -1)
                _, _, mean_action = policy.sample(inp)
                action = mean_action.cpu().numpy().reshape(-1)
                state, reward, done, info = env.step(action * max_action)

                ep_metrics["weighted_avg_fidelity"].append(info["weighted_avg_fidelity"])
                ep_metrics["mean_aosi"].append(info["mean_aosi"])
                ep_metrics["energy_per_step"].append(info["energy"])
                ep_metrics["jain_fairness"].append(info["jain_fairness"])
                ep_metrics["service_rate"].append(info["service_rate"])
                ep_energy_total += info["energy"]
                ep_reward += reward

            for k, lst in ep_metrics.items():
                physical[k].append(np.mean(lst))
            physical["energy_per_episode_kJ"].append(ep_energy_total / 1000.0)
            episode_rewards.append(ep_reward)

    rew = np.array(episode_rewards)
    print(f"\n  Episode 4-D reward sums (avg over 200 steps each, then over {num_episodes} eps):")
    print(f"    [Fid, AoSI, Ener, Jain] reward = "
          f"{rew.mean(0).round(2).tolist()}  ± {rew.std(0).round(2).tolist()}")

    print(f"\n  Physical metrics (mean ± std over {num_episodes} eps):")
    for k, vals in physical.items():
        arr = np.array(vals)
        print(f"    {k:<28s}: {arr.mean():.4f}  ± {arr.std():.4f}")

    print("\n  Conference paper reference (M=1, K=5, Table II):")
    print("    weighted_avg_fidelity        : 0.88")
    print("    mean_aosi                    : 1.06")
    print("    energy_per_episode_kJ        : 21.96")
    print("    jain_fairness                : 0.98")
    print("    service_rate                 : 0.99")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--num_uavs", type=int, required=True)
    p.add_argument("--num_devices", type=int, default=5)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    diagnose(args.checkpoint_dir, args.num_uavs, args.num_devices, args.episodes, args.seed)
