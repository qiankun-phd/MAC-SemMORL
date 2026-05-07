"""Smoke test for the classical baselines (Issue #8 PR-D).

Two baselines registered: ``pareto-pg`` and ``pareto-q``. This script
constructs each, trains for a few hundred env steps, asserts the
trajectory + final-state buffers populate correctly, and runs them
through the runner so the produced npz validates against the schema.

Usage:
    PYTHONPATH=. python scripts/smoketest_classical.py
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


def _direct_smoke(name: str, kwargs: dict, num_steps: int, eval_interval: int):
    print(f"=== direct smoke: {name} ===")
    import gym
    import environments  # noqa: F401
    from baselines import get_baseline

    env = gym.make("UAV-SemCom-v0", num_devices=5, max_episode_steps=200)
    env.seed(0)
    cls = get_baseline(name)
    print(f"  registry returned: {cls.__name__}")
    bl = cls(env=env, log_dir=tempfile.mkdtemp(), seed=0,
             method_kwargs=kwargs)
    bl.train(num_steps=num_steps, eval_interval=eval_interval)
    assert len(bl.eval_steps) >= 1, f"no eval points for {name}"
    assert bl.final_ep_objs is not None and bl.final_ep_objs.shape[1] == 4
    assert np.all(np.isnan(bl.c_violation_rates)), f"{name} unconstrained → NaN"
    for arr_name in ["hv_trajectory", "ut_trajectory", "sparsity_trajectory",
                     "final_obj_means", "final_obj_stds"]:
        arr = np.asarray(getattr(bl, arr_name))
        assert np.all(np.isfinite(arr)), f"{name}.{arr_name} non-finite: {arr}"
    print(f"  eval points: {bl.eval_steps}")
    print(f"  HV trajectory: {[round(h, 1) for h in bl.hv_trajectory]}")
    print(f"  final_obj_means: {bl.final_obj_means.round(2).tolist()}")
    print(f"  wallclock: {bl.wallclock_seconds:.2f}s")
    print(f"  OK: all numeric buffers finite")


def _runner_smoke(name: str, kwargs: dict, num_steps: int, eval_interval: int):
    print(f"\n=== runner smoke: {name} ===")
    from baselines import validate_result_npz
    tmpdir = tempfile.mkdtemp(prefix=f"smoketest_{name.replace('-', '_')}_")
    try:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "run_baseline.py"),
            "--baseline", name,
            "--num_uavs", "1", "--num_devices", "5",
            "--num_steps", str(num_steps),
            "--eval_interval", str(eval_interval),
            "--seed", "0",
            "--results_dir", tmpdir,
            "--method_kwargs", json.dumps(kwargs),
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
        print(f"  schema OK ({len(data)} keys)")
        cfg = json.loads(str(data["config_dict"]))
        assert cfg["baseline"] == name
        print(f"  config_dict round-trips: baseline={cfg['baseline']}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    pg_kwargs = dict(
        episodes_per_update=2, hidden=(32, 32), device="cpu",
    )
    q_kwargs = dict(
        eps_decay_steps=500,
    )

    _direct_smoke("pareto-pg", pg_kwargs, num_steps=400, eval_interval=200)
    _direct_smoke("pareto-q", q_kwargs, num_steps=400, eval_interval=200)
    _runner_smoke("pareto-pg", pg_kwargs, num_steps=400, eval_interval=200)
    _runner_smoke("pareto-q", q_kwargs, num_steps=400, eval_interval=200)
    print("\nAll classical-baseline smoke checks passed.")


if __name__ == "__main__":
    main()
