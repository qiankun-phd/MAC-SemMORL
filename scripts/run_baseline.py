"""Unified runner for Issue #8 baselines (DESIGN-baselines.md §4).

Single source of truth for the eval cadence, seed list, and result-saving
protocol. Per-baseline classes only see ``train(num_steps, eval_interval)``
and cannot override the post-training final-eval pass.

Usage:
    PYTHONPATH=. python scripts/run_baseline.py \\
        --baseline noop --num_uavs 1 --num_devices 5 --seed 1 --num_steps 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import environments  # noqa: F401, E402  -- registers UAV-SemCom-v0 / -Multi-v0
from baselines import (  # noqa: E402
    NpzResult,
    get_baseline,
    list_baselines,
    write_result_npz,
)


def make_env(args):
    if args.num_uavs >= 2:
        return gym.make(
            "UAV-SemCom-Multi-v0",
            num_uavs=args.num_uavs,
            num_devices=args.num_devices,
            max_episode_steps=args.max_episode_steps,
            device_mobility=args.device_mobility,
            device_speed=args.device_speed,
        )
    return gym.make(
        "UAV-SemCom-v0",
        num_devices=args.num_devices,
        max_episode_steps=args.max_episode_steps,
        device_mobility=args.device_mobility,
        device_speed=args.device_speed,
    )


def build_npz(baseline) -> NpzResult:
    """Pull the schema-shaped buffers from a finished baseline. The runner
    keeps this conversion in one place so each Baseline subclass only has
    to populate its trajectory + final-state buffers."""
    return NpzResult(
        eval_steps=np.asarray(baseline.eval_steps, dtype=np.int64),
        hv_trajectory=np.asarray(baseline.hv_trajectory, dtype=np.float64),
        ut_trajectory=np.asarray(baseline.ut_trajectory, dtype=np.float64),
        sparsity_trajectory=np.asarray(
            baseline.sparsity_trajectory, dtype=np.float64
        ),
        final_ep_objs=baseline.final_ep_objs,
        final_ep_prefs=baseline.final_ep_prefs,
        final_obj_means=baseline.final_obj_means,
        final_obj_stds=baseline.final_obj_stds,
        c_violation_rates=baseline.c_violation_rates,
        wallclock_seconds=baseline.wallclock_seconds,
        config_dict={},  # filled by the runner
    )


def main():
    p = argparse.ArgumentParser(description="Issue #8 baseline runner")
    p.add_argument("--baseline", required=True, choices=list_baselines())
    p.add_argument("--num_uavs", type=int, default=1)
    p.add_argument("--num_devices", type=int, default=5)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--device_mobility", type=str, default="none",
                   choices=["none", "line", "drift"])
    p.add_argument("--device_speed", type=float, default=0.0)
    # Protocol knobs (DESIGN-baselines.md §2.1).
    p.add_argument("--num_steps", type=int, default=1_000_000)
    p.add_argument("--eval_interval", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=1)
    # Tagging knobs (DESIGN-baselines.md §3).
    p.add_argument("--mobility_tag", type=str, default=None,
                   help="Override the mobility tag in the output filename. "
                        "Default: --device_mobility value.")
    p.add_argument("--channel_tag", type=str, default="analytical",
                   choices=["analytical", "deepmimo"])
    p.add_argument("--perturbation_tag", type=str, default="none")
    p.add_argument("--constraint_handler", type=str, default=None,
                   help="Set when the baseline enforces constraints; affects "
                        "result filename + violation-rate field.")
    p.add_argument("--results_dir", type=str, default="results")
    # Method-specific kwargs as a JSON blob — keeps the runner agnostic.
    p.add_argument("--method_kwargs", type=str, default="{}",
                   help="JSON-encoded dict passed to the baseline's __init__")
    p.add_argument("--log_dir_root", type=str, default="logs/baselines")

    args = p.parse_args()

    method_kwargs: Dict[str, Any] = json.loads(args.method_kwargs)

    env = make_env(args)
    env.seed(args.seed)

    log_dir = os.path.join(
        args.log_dir_root, f"{args.baseline}_seed{args.seed}"
    )
    os.makedirs(log_dir, exist_ok=True)

    cls = get_baseline(args.baseline)
    baseline = cls(env=env, log_dir=log_dir, seed=args.seed,
                   method_kwargs=method_kwargs)
    print(f"[runner] training {args.baseline} for {args.num_steps} steps "
          f"(eval every {args.eval_interval}, seed={args.seed})", flush=True)
    baseline.train(num_steps=args.num_steps,
                   eval_interval=args.eval_interval)

    npz = build_npz(baseline)
    npz.config_dict = {
        "baseline": args.baseline,
        "num_uavs": args.num_uavs,
        "num_devices": args.num_devices,
        "num_steps": args.num_steps,
        "eval_interval": args.eval_interval,
        "seed": args.seed,
        "device_mobility": args.device_mobility,
        "device_speed": args.device_speed,
        "channel_tag": args.channel_tag,
        "perturbation_tag": args.perturbation_tag,
        "constraint_handler": args.constraint_handler,
        "method_kwargs": method_kwargs,
    }

    out_path = write_result_npz(
        result=npz,
        output_dir=args.results_dir,
        method=args.baseline,
        M=args.num_uavs,
        K=args.num_devices,
        seed=args.seed,
        mobility=args.mobility_tag or args.device_mobility,
        channel=args.channel_tag,
        perturbation=args.perturbation_tag,
        constraint_handler=args.constraint_handler,
    )
    print(f"[runner] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
