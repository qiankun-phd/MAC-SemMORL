"""Smoke test for the baseline framework (Issue #8 PR-B).

Verifies the scaffolding runs end-to-end without depending on any external
baseline:
    1. The Baseline ABC + registry imports cleanly.
    2. The no-op baseline registers under name='noop' and is constructable.
    3. ``scripts/run_baseline.py --baseline noop`` for ~1000 env steps writes
       an npz file conforming to ``docs/DESIGN-baselines.md`` §3.
    4. ``validate_result_npz`` accepts the saved file with no missing keys.
    5. Backward-compat: ``from baselines import greedy_aosi`` still works
       so the four ``experiments/`` callers don't break.

Usage:
    PYTHONPATH=. python scripts/smoketest_baseline_skeleton.py
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


def step1_imports():
    print("=== step 1: imports ===")
    from baselines import (  # noqa: F401
        Baseline,
        get_baseline,
        list_baselines,
        register_baseline,
        write_result_npz,
        validate_result_npz,
        NpzResult,
        # backward-compat re-exports
        greedy_aosi,
        random_policy,
        nearest_device_round_robin,
        fixed_trajectory_greedy_power,
        evaluate_policy,
        evaluate_comm_metrics,
        make_uav_semcom_env,
    )
    print("  OK: ABC + registry + npz schema + 7 backward-compat symbols all import")
    return list_baselines


def step2_registration(list_baselines):
    print("=== step 2: registration ===")
    from baselines import get_baseline
    names = list_baselines()
    assert "noop" in names, f"noop missing from registry: {names}"
    cls = get_baseline("noop")
    assert cls.__name__ == "NoOpBaseline", f"unexpected class: {cls}"
    print(f"  OK: registry exposes {names}; noop -> {cls.__name__}")


def step3_runner(tmpdir):
    print("=== step 3: runner end-to-end ===")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "run_baseline.py"),
        "--baseline", "noop",
        "--num_uavs", "1",
        "--num_devices", "5",
        "--num_steps", "1000",
        "--eval_interval", "200",
        "--seed", "1",
        "--results_dir", tmpdir,
        "--method_kwargs", "{}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    print(f"  running: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("  STDOUT:", res.stdout)
        print("  STDERR:", res.stderr)
        raise AssertionError(f"runner exited {res.returncode}")
    print("  OK: runner exited 0")
    print(f"  stdout tail: {res.stdout.strip().splitlines()[-1] if res.stdout.strip() else '<empty>'}")


def step4_npz_schema(tmpdir):
    print("=== step 4: npz schema validation ===")
    from baselines import validate_result_npz, write_result_npz, NpzResult
    files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
    assert len(files) == 1, f"expected 1 npz file in {tmpdir}, got {files}"
    path = os.path.join(tmpdir, files[0])
    print(f"  found: {files[0]}")

    data = validate_result_npz(path)
    print(f"  OK: validator accepts file ({len(data)} keys present)")

    # Also check shapes match the noop baseline's expected output.
    assert data["eval_steps"].shape == data["hv_trajectory"].shape, (
        f"eval/hv shape mismatch: {data['eval_steps'].shape} vs {data['hv_trajectory'].shape}"
    )
    assert data["c_violation_rates"].shape == (3,), (
        f"c_violation_rates shape: {data['c_violation_rates'].shape}"
    )
    assert np.all(np.isnan(data["c_violation_rates"])), (
        "noop is unconstrained — c_violation_rates should be NaN"
    )
    assert data["final_obj_means"].shape == (4,), (
        f"final_obj_means shape: {data['final_obj_means'].shape}"
    )

    cfg = json.loads(str(data["config_dict"]))
    assert cfg["baseline"] == "noop"
    assert cfg["num_steps"] == 1000
    print(f"  OK: shapes + dtypes + config_dict round-trip correctly")
    print(f"  recorded git_sha: {data['git_sha']!s}")


def step5_corrupt_npz_detected():
    print("=== step 5: corrupt npz is detected ===")
    from baselines import validate_result_npz
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        # Write an npz missing several keys.
        np.savez(f.name, eval_steps=np.array([1, 2, 3]))
        path = f.name
    try:
        try:
            validate_result_npz(path)
            raise AssertionError("validator should have raised on missing keys")
        except AssertionError as e:
            msg = str(e)
            assert "missing required schema keys" in msg, msg
            print("  OK: validator raised on missing keys as expected")
    finally:
        os.unlink(path)


def step6_filename_tagging(tmpdir):
    print("=== step 6: filename tag generator ===")
    from baselines.npz_schema import make_config_tag
    tag = make_config_tag(
        method="C-MORL", M=2, K=5,
        mobility="none", channel="analytical",
        perturbation="none", constraint_handler="lagrangian",
    )
    expected = "C-MORL_M2_K5_mobnone_chanalytical_pertnone_constrlagrangian"
    assert tag == expected, f"tag mismatch:\n  got: {tag}\n  expected: {expected}"
    print(f"  OK: {tag}")


def main():
    list_baselines = step1_imports()
    step2_registration(list_baselines)
    tmpdir = tempfile.mkdtemp(prefix="smoketest_baseline_")
    try:
        step3_runner(tmpdir)
        step4_npz_schema(tmpdir)
        step5_corrupt_npz_detected()
        step6_filename_tagging(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nAll baseline-skeleton smoke checks passed.")


if __name__ == "__main__":
    main()
