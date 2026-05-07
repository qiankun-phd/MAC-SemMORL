"""
Communication baselines for UAV-SemCom (no RL training needed).

1. Fixed Trajectory + Greedy Power (FT-GP):
   UAV flies a circular trajectory visiting each device cluster;
   transmit power proportional to path loss; compression ratio fixed.

2. Greedy AoSI:
   UAV moves toward the device with the highest AoSI;
   allocate all power to that device; adaptive compression.

3. Random policy:
   Uniform random actions (lower bound).

Run:
  python baselines.py
  python baselines.py --device_mobility line --device_speed 0.5
"""

import argparse
import os
import numpy as np
import gym
from environments import *


def make_uav_semcom_env(
    num_devices=5,
    max_episode_steps=200,
    device_mobility="none",
    device_speed=0.0,
):
    """Match ``main_uav.py`` / env defaults (speed≤0 with mobility on → env uses 0.5 m/s)."""
    kw = dict(num_devices=num_devices, max_episode_steps=max_episode_steps)
    if device_mobility != "none":
        kw["device_mobility"] = device_mobility
        kw["device_speed"] = float(device_speed)
    return gym.make("UAV-SemCom-v0", **kw)


def evaluate_policy(env, policy_fn, n_episodes=50, seed=42):
    """Evaluate a deterministic policy over multiple episodes."""
    all_rewards = []
    for ep in range(n_episodes):
        env.seed(seed + ep)
        obs = env.reset()
        ep_reward = np.zeros(env.reward_num)
        done = False
        while not done:
            action = policy_fn(obs, env)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        all_rewards.append(ep_reward)
    return np.array(all_rewards)


def evaluate_comm_metrics(env, policy_fn, n_episodes=50, seed=42):
    """
    Episode-averaged raw communication scalars matching keys in RL ``comm_metrics`` npz.

    Aggregation mirrors ``Agent.get_objs`` per episode: time-average fidelity, AoSI, Jain,
    service rate; **sum** of per-slot energy over the episode, then mean across episodes.
    Saved arrays are length-1 so ``plot_results.load_comm_avg`` can ingest them like RL logs.
    """
    ep_mean_fids, ep_mean_aosis = [], []
    ep_total_energies, ep_mean_jains, ep_mean_srs = [], [], []
    for ep in range(n_episodes):
        env.seed(seed + ep)
        obs = env.reset()
        ep_fid = ep_aosi = ep_e = ep_j = ep_s = 0.0
        steps = 0
        done = False
        while not done:
            action = policy_fn(obs, env)
            obs, _reward, done, info = env.step(action)
            ep_fid += float(info.get("weighted_avg_fidelity", 0.0))
            ep_aosi += float(info.get("mean_aosi", 0.0))
            ep_e += float(info.get("energy", 0.0))
            ep_j += float(info.get("jain_fairness", 0.0))
            ep_s += float(info.get("service_rate", 0.0))
            steps += 1
        den = max(steps, 1)
        ep_mean_fids.append(ep_fid / den)
        ep_mean_aosis.append(ep_aosi / den)
        ep_total_energies.append(ep_e)
        ep_mean_jains.append(ep_j / den)
        ep_mean_srs.append(ep_s / den)
    return {
        "weighted_avg_fidelity": np.array([float(np.mean(ep_mean_fids))]),
        "mean_aosi": np.array([float(np.mean(ep_mean_aosis))]),
        "energy": np.array([float(np.mean(ep_total_energies))]),
        "jain_fairness": np.array([float(np.mean(ep_mean_jains))]),
        "service_rate": np.array([float(np.mean(ep_mean_srs))]),
    }


def fixed_trajectory_greedy_power(obs, env):
    """
    UAV flies a circle covering the area; power ∝ inverse distance;
    compression ratio set to 0.7 (moderate).
    """
    K = env.num_devices
    t = env.steps / env.max_episode_steps
    angle = 2 * np.pi * t
    radius = env.area_size * 0.35
    center = np.array([env.area_size / 2, env.area_size / 2])
    target = center + radius * np.array([np.cos(angle), np.sin(angle)])
    direction = target - env.uav_pos
    dist = np.linalg.norm(direction)
    if dist > 1.0:
        accel = direction / dist
    else:
        accel = np.zeros(2)

    distances = np.linalg.norm(env.device_positions - env.uav_pos, axis=1)
    inv_dist = 1.0 / (distances + 10.0)
    power_ratios = inv_dist / inv_dist.sum()

    compression = np.full(K, 0.7)

    action = np.concatenate([accel, power_ratios * 2 - 1, compression * 2 - 1])
    return np.clip(action, -1.0, 1.0)


def greedy_aosi(obs, env):
    """
    UAV moves toward device with highest AoSI;
    concentrate power on that device.
    """
    K = env.num_devices
    target_dev = np.argmax(env.aosi)
    direction = env.device_positions[target_dev] - env.uav_pos
    dist = np.linalg.norm(direction)
    if dist > 1.0:
        accel = direction / dist
    else:
        accel = np.zeros(2)

    power = np.full(K, -0.8)
    power[target_dev] = 1.0

    compression = np.full(K, 0.4)
    compression[target_dev] = 0.9

    action = np.concatenate([accel, power, compression * 2 - 1])
    return np.clip(action, -1.0, 1.0)


def random_policy(obs, env):
    """Uniform random actions."""
    return env.action_space.sample()


def nearest_device_round_robin(obs, env):
    """
    UAV moves to nearest un-served device in round-robin;
    balanced power allocation.
    """
    K = env.num_devices
    target_idx = env.steps % K
    direction = env.device_positions[target_idx] - env.uav_pos
    dist = np.linalg.norm(direction)
    if dist > 1.0:
        accel = direction / dist
    else:
        accel = np.zeros(2)

    power = np.zeros(K)
    power[target_idx] = 1.0
    others = [i for i in range(K) if i != target_idx]
    for i in others:
        power[i] = -0.5
    compression = np.full(K, 0.6) * 2 - 1

    action = np.concatenate([accel, power, compression])
    return np.clip(action, -1.0, 1.0)


if __name__ == "__main__":
    import numpy as _np
    if not hasattr(_np, "bool8"):
        _np.bool8 = _np.bool_

    parser = argparse.ArgumentParser(description="UAV-SemCom heuristic baselines (no RL).")
    parser.add_argument("--num_devices", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument(
        "--device_mobility",
        type=str,
        default="none",
        choices=["none", "line", "drift"],
    )
    parser.add_argument(
        "--device_speed",
        type=float,
        default=0.0,
        help="m/s; if ≤0 and mobility≠none, env uses 0.5 (same as main_uav)",
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--output_root",
        type=str,
        default="logs/uav",
        help="Under this, results go to mob_<mobility>/…",
    )
    args = parser.parse_args()

    env = make_uav_semcom_env(
        num_devices=args.num_devices,
        max_episode_steps=args.max_episode_steps,
        device_mobility=args.device_mobility,
        device_speed=args.device_speed,
    )

    mob_tag = args.device_mobility
    mob_dir = os.path.join(args.output_root, f"mob_{mob_tag}")
    os.makedirs(mob_dir, exist_ok=True)

    policies = {
        "FT-GP (Fixed Traj + Greedy Power)": fixed_trajectory_greedy_power,
        "Greedy AoSI":                       greedy_aosi,
        "Round Robin":                       nearest_device_round_robin,
        "Random":                            random_policy,
    }

    obj_names = ["SemFid", "Fresh", "Energy", "Fair"]
    n_obj = env.reward_num

    print("=" * 82)
    print(
        f"Baseline Evaluation (mobility={mob_tag}, {args.n_episodes} episodes, "
        f"{n_obj} objectives)"
    )
    print("=" * 82)
    header = f"{'Method':<40}" + "".join(f" {n:>8}" for n in obj_names[:n_obj]) + f" {'Sum':>8}"
    print(header)
    print("-" * 82)

    results = {}
    for name, fn in policies.items():
        rewards = evaluate_policy(env, fn, n_episodes=args.n_episodes)
        mean_r = rewards.mean(axis=0)
        std_r = rewards.std(axis=0)
        results[name] = (mean_r, std_r)
        vals = "".join(f" {mean_r[i]:>8.1f}" for i in range(n_obj))
        print(f"{name:<40}{vals} {mean_r.sum():>8.1f}")

    print("-" * 82)
    print("\nStandard deviations:")
    for name, (m, s) in results.items():
        stds = "".join(f"  ±{s[i]:>6.1f}" for i in range(n_obj))
        print(f"  {name:<38}{stds}")

    payload = {k.replace(" ", "_"): v[0] for k, v in results.items()}
    out_npz = os.path.join(mob_dir, "baseline_results_4obj.npz")
    np.savez(out_npz, **payload)
    print(f"\nResults saved to {out_npz}")
    if mob_tag == "none":
        legacy = os.path.join(args.output_root, "baseline_results_4obj.npz")
        np.savez(legacy, **payload)
        print(f"(mobility=none) Also wrote {legacy} for plot_results / final_results.")

    # Communication metrics for Fig.~(b) (same schema as RL summary/comm_metrics_*.npz)
    comm_dir_tags = {
        "FT-GP (Fixed Traj + Greedy Power)": "baseline_ft_gp",
        "Greedy AoSI": "baseline_greedy_aosi",
        "Round Robin": "baseline_round_robin",
        "Random": "baseline_random",
    }
    for name, fn in policies.items():
        tag = comm_dir_tags[name]
        summary_dir = os.path.join(mob_dir, tag, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        comm = evaluate_comm_metrics(env, fn, n_episodes=args.n_episodes)
        out_path = os.path.join(summary_dir, "comm_metrics_baseline.npz")
        np.savez(out_path, **comm)
        print(f"Communication metrics -> {out_path}")
        if mob_tag == "none":
            legacy_summary = os.path.join(args.output_root, tag, "summary")
            os.makedirs(legacy_summary, exist_ok=True)
            legacy_comm = os.path.join(legacy_summary, "comm_metrics_baseline.npz")
            np.savez(legacy_comm, **comm)
            print(f"  (none) legacy -> {legacy_comm}")
