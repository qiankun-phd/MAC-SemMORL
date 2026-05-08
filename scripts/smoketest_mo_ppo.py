"""Smoke test for the MO-PPO baseline (Issue #8 PR-C).

Verifies the baseline trains end-to-end on UAV-SemCom-v0 for a few thousand
steps, populates trajectory + final-state buffers correctly, and writes a
schema-valid npz via the runner. Not a research-quality reproduction —
just enough to catch import-time / shape / NaN bugs before pushing.

Usage:
    PYTHONPATH=. python scripts/smoketest_mo_ppo.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    print("=== MO-PPO direct construction + tiny train ===")
    import gym
    import environments  # noqa: F401
    from baselines import get_baseline, validate_result_npz, write_result_npz, NpzResult

    env = gym.make("UAV-SemCom-v0", num_devices=5, max_episode_steps=200)
    env.seed(0)

    cls = get_baseline("mo-ppo")
    print(f"  registry returned: {cls.__name__}")

    method_kwargs = dict(
        rollout_T=200,         # 1 episode
        minibatch_size=32,
        update_epochs=2,
        eval_episodes_per_pref=1,
        device="cpu",
    )
    bl = cls(env=env, log_dir=tempfile.mkdtemp(), seed=0,
             method_kwargs=method_kwargs)

    bl.train(num_steps=600, eval_interval=200)

    # Sanity: trajectories populated
    assert len(bl.eval_steps) >= 2, f"got {len(bl.eval_steps)} eval points"
    assert len(bl.hv_trajectory) == len(bl.eval_steps)
    assert bl.final_ep_objs is not None and bl.final_ep_objs.shape[1] == 4
    assert bl.final_obj_means.shape == (4,)
    assert np.all(np.isnan(bl.c_violation_rates)), "MO-PPO unconstrained → NaN"
    print(f"  eval points: {bl.eval_steps}")
    print(f"  HV trajectory: {[round(h, 1) for h in bl.hv_trajectory]}")
    print(f"  final_obj_means: {bl.final_obj_means.round(2).tolist()}")
    print(f"  wallclock: {bl.wallclock_seconds:.2f}s")

    # NaN/Inf check on every buffer.
    for arr_name in ["hv_trajectory", "ut_trajectory", "sparsity_trajectory",
                     "final_obj_means", "final_obj_stds"]:
        arr = np.asarray(getattr(bl, arr_name))
        assert np.all(np.isfinite(arr)), f"{arr_name} has non-finite: {arr}"
    print("  OK: all numeric buffers finite")

    # End-to-end via runner
    print("\n=== runner end-to-end ===")
    tmpdir = tempfile.mkdtemp(prefix="smoketest_moppo_")
    try:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "run_baseline.py"),
            "--baseline", "mo-ppo",
            "--num_uavs", "1", "--num_devices", "5",
            "--num_steps", "600", "--eval_interval", "200",
            "--seed", "0",
            "--results_dir", tmpdir,
            "--method_kwargs", json.dumps({
                "rollout_T": 200, "minibatch_size": 32,
                "update_epochs": 2, "device": "cpu",
            }),
        ]
        env_var = os.environ.copy()
        env_var["PYTHONPATH"] = ROOT
        res = subprocess.run(cmd, env=env_var, capture_output=True, text=True)
        if res.returncode != 0:
            print("  STDOUT:", res.stdout)
            print("  STDERR:", res.stderr)
            raise AssertionError(f"runner exited {res.returncode}")
        files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
        assert len(files) == 1, f"expected 1 npz, got {files}"
        path = os.path.join(tmpdir, files[0])
        data = validate_result_npz(path)
        print(f"  saved: {files[0]}")
        print(f"  schema OK ({len(data)} keys present)")
        cfg = json.loads(str(data["config_dict"]))
        assert cfg["baseline"] == "mo-ppo"
        print(f"  config_dict round-trips: baseline={cfg['baseline']}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\nAll MO-PPO smoke checks passed.")


if __name__ == "__main__":
    main()
