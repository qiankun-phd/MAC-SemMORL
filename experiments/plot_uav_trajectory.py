#!/usr/bin/env python3
"""
Record UAV (and optional IoT) positions over one episode and save a 2D trajectory figure.

Modes:
  * Heuristic: policies from ``baselines.py`` (``--policy ...``).
  * COLA: load ``policy_*.pkl`` + ``encoder_*.pkl`` from a training run (``--cola_log_dir``).

Examples:
  python experiments/plot_uav_trajectory.py --policy greedy_aosi --seed 0 -o figures/uav_traj_greedy.png

  python experiments/plot_uav_trajectory.py --cola_log_dir logs/uav/cola_line_s1 \\
    --device_mobility line --preference 0.25 0.25 0.25 0.25 --cuda \\
    -o figures/uav_traj_cola_line.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from baselines import (  # noqa: E402
    fixed_trajectory_greedy_power,
    greedy_aosi,
    make_uav_semcom_env,
    nearest_device_round_robin,
    random_policy,
)
from model import GaussianPolicy, Latent_Encoder  # noqa: E402

POLICIES = {
    "greedy_aosi": greedy_aosi,
    "random": random_policy,
    "ft_gp": fixed_trajectory_greedy_power,
    "round_robin": nearest_device_round_robin,
}

_POLICY_USE_LATENT = True
_POLICY_USE_S = True
_POLICY_USE_W = True
_USE_POLICY_PREFERENCE = True
_LATENT_DIM_DEFAULT = 50
_HIDDEN = [128, 128]
_USE_AVG = True


def rollout_positions(env, policy_fn, seed: int):
    env.seed(seed)
    obs = env.reset()
    uav = [env.uav_pos.copy()]
    dev = [env.device_positions.copy()]
    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, _r, done, info = env.step(action)
        uav.append(np.asarray(info["uav_pos"], dtype=np.float64))
        dev.append(env.device_positions.copy())
    return np.stack(uav, axis=0), np.stack(dev, axis=0)


def _build_cola_networks(env, device: torch.device, latent_dim: int):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    rn = env.reward_num

    if _POLICY_USE_LATENT:
        in_dim = latent_dim
        if _POLICY_USE_S:
            in_dim += obs_dim
        if _POLICY_USE_W:
            in_dim += rn
        policy = GaussianPolicy(
            in_dim,
            act_dim,
            hidden_units=_HIDDEN,
            Use_Policy_Preference=_USE_POLICY_PREFERENCE,
        ).to(device)
    else:
        policy = GaussianPolicy(
            obs_dim + rn,
            act_dim,
            hidden_units=_HIDDEN,
            Use_Policy_Preference=_USE_POLICY_PREFERENCE,
        ).to(device)

    encoder = Latent_Encoder(
        _USE_AVG, obs_dim, act_dim, rn, latent_dim
    ).to(device)
    return policy, encoder


def _resolve_cola_ckpt_paths(log_dir: str, ckpt: str) -> tuple:
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if ckpt == "final":
        pol = os.path.join(ckpt_dir, "policy_final.pkl")
        enc = os.path.join(ckpt_dir, "encoder_final.pkl")
        if os.path.isfile(pol) and os.path.isfile(enc):
            return pol, enc
    pol = os.path.join(ckpt_dir, "policy_{}.pkl".format(ckpt))
    enc = os.path.join(ckpt_dir, "encoder_{}.pkl".format(ckpt))
    if os.path.isfile(pol) and os.path.isfile(enc):
        return pol, enc
    best = -1
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            m = re.match(r"policy_(\d+)\.pkl$", name)
            if m:
                best = max(best, int(m.group(1)))
    if best >= 0:
        return (
            os.path.join(ckpt_dir, "policy_{}.pkl".format(best)),
            os.path.join(ckpt_dir, "encoder_{}.pkl".format(best)),
        )
    raise FileNotFoundError(
        "No policy/encoder checkpoints under {} (tried final and numeric tags).".format(ckpt_dir)
    )


def make_cola_policy_fn(
    env,
    log_dir: str,
    preference: np.ndarray,
    *,
    latent_dim: int,
    cuda: bool,
    ckpt: str,
):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    policy, encoder = _build_cola_networks(env, device, latent_dim)
    pol_path, enc_path = _resolve_cola_ckpt_paths(log_dir, ckpt)
    policy.load_state_dict(torch.load(pol_path, map_location=device))
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    policy.eval()
    encoder.eval()

    pref = np.asarray(preference, dtype=np.float32)
    assert pref.shape == (env.reward_num,)
    pref_sum = float(pref.sum())
    if abs(pref_sum - 1.0) > 1e-3:
        pref = pref / pref_sum

    max_act = np.asarray(env.action_space.high, dtype=np.float32)

    def exploit_vec(state: np.ndarray) -> np.ndarray:
        st = torch.FloatTensor(state).unsqueeze(0).to(device)
        pr = torch.FloatTensor(pref).unsqueeze(0).to(device)
        if _POLICY_USE_LATENT:
            z = encoder.get_latent_features(st)
            parts = [z]
            if _POLICY_USE_S:
                parts.append(st)
            if _POLICY_USE_W:
                parts.append(pr)
            inp = torch.cat(parts, dim=-1)
        else:
            inp = torch.cat([st, pr], dim=-1)
        with torch.no_grad():
            _, _, action = policy.sample(inp)
        a = action.cpu().numpy().reshape(-1)
        return (a * max_act).astype(np.float64)

    def policy_fn(obs, env_ref):
        return exploit_vec(np.asarray(obs, dtype=np.float32))

    meta = {"pol_path": pol_path, "enc_path": enc_path, "device": str(device)}
    return policy_fn, meta


def main():
    p = argparse.ArgumentParser(
        description="Plot UAV trajectory (one episode): heuristic or COLA checkpoint."
    )
    p.add_argument(
        "--policy",
        type=str,
        default="greedy_aosi",
        choices=list(POLICIES.keys()),
        help="Heuristic policy (ignored if --cola_log_dir is set).",
    )
    p.add_argument(
        "--cola_log_dir",
        type=str,
        default=None,
        help="Training run dir containing checkpoints/ (e.g. logs/uav/cola_line_s1).",
    )
    p.add_argument(
        "--cola_ckpt",
        type=str,
        default="final",
        help="Checkpoint tag: final | 0 | 1 | ...",
    )
    p.add_argument(
        "--preference",
        type=float,
        nargs="+",
        default=None,
        help="Preference vector (length = reward_num). Default: uniform. COLA only.",
    )
    p.add_argument("--latent_dim", type=int, default=_LATENT_DIM_DEFAULT)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_devices", type=int, default=5)
    p.add_argument("--device_mobility", type=str, default="none", choices=["none", "line", "drift"])
    p.add_argument("--device_speed", type=float, default=0.0)
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join(ROOT, "figures", "uav_trajectory.png"),
    )
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    env = make_uav_semcom_env(
        num_devices=args.num_devices,
        max_episode_steps=200,
        device_mobility=args.device_mobility,
        device_speed=args.device_speed,
    )
    area = float(env.area_size)
    rn = env.reward_num

    if args.cola_log_dir:
        log_dir = args.cola_log_dir
        if not os.path.isdir(log_dir):
            log_dir = os.path.join(ROOT, args.cola_log_dir)
        if not os.path.isdir(log_dir):
            print("Missing log dir: {}".format(log_dir), file=sys.stderr)
            sys.exit(1)
        if args.preference is None:
            pref = np.ones(rn, dtype=np.float32) / rn
        else:
            pref = np.array(args.preference, dtype=np.float32)
            if pref.size != rn:
                p.error("--preference length {} != env.reward_num {}".format(pref.size, rn))
        policy_fn, meta = make_cola_policy_fn(
            env,
            log_dir,
            pref,
            latent_dim=args.latent_dim,
            cuda=args.cuda,
            ckpt=args.cola_ckpt,
        )
        title_tag = "COLA ({}, ckpt={})".format(
            os.path.basename(log_dir.rstrip("/")), args.cola_ckpt
        )
        print("Loaded {}\n       {}  device={}".format(
            meta["pol_path"], meta["enc_path"], meta["device"]
        ))
    else:
        policy_fn = POLICIES[args.policy]
        title_tag = args.policy
        if args.preference is not None:
            print("Note: --preference only used with --cola_log_dir; ignoring for heuristic.")

    uav_xy, dev_xy = rollout_positions(env, policy_fn, args.seed)
    env.close()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect("equal")
    ax.set_xlim(0, area)
    ax.set_ylim(0, area)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    mob = args.device_mobility
    ax.set_title(
        "UAV trajectory ({}, mobility={}, seed={})".format(title_tag, mob, args.seed),
        fontsize=10,
    )

    K = dev_xy.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 0.9, K))

    for k in range(K):
        ax.scatter(
            dev_xy[0, k, 0],
            dev_xy[0, k, 1],
            s=55,
            marker="o",
            facecolors="none",
            edgecolors=colors[k],
            linewidths=1.2,
            zorder=4,
        )
        if mob != "none" and dev_xy.shape[0] > 2:
            ax.plot(
                dev_xy[:, k, 0],
                dev_xy[:, k, 1],
                color=colors[k],
                alpha=0.35,
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )
        ax.scatter(
            dev_xy[-1, k, 0],
            dev_xy[-1, k, 1],
            s=60,
            c=[colors[k]],
            marker="s",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label="IoT {}".format(k) if k < 6 else None,
        )

    ax.plot(
        uav_xy[:, 0],
        uav_xy[:, 1],
        color="#1565C0",
        linewidth=1.6,
        zorder=3,
        label="UAV path",
    )
    ax.scatter(uav_xy[0, 0], uav_xy[0, 1], c="green", s=70, zorder=6, marker="^", label="UAV start")
    ax.scatter(uav_xy[-1, 0], uav_xy[-1, 1], c="red", s=70, zorder=6, marker="v", label="UAV end")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    out_abs = os.path.abspath(args.output)
    odir = os.path.dirname(out_abs)
    if odir:
        os.makedirs(odir, exist_ok=True)
    fig.savefig(out_abs, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {} ({} steps)".format(out_abs, uav_xy.shape[0]))


if __name__ == "__main__":
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
    main()
