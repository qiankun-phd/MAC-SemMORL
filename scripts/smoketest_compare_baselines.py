"""Smoke test for the comparison analyzer (Issue #8 PR-G).

Generates a handful of synthetic schema-valid npz files, runs the
analyzer on them, and verifies the outputs are produced + parse correctly.

Usage:
    PYTHONPATH=. python scripts/smoketest_compare_baselines.py
"""
from __future__ import annotations

import csv
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


def _fake_npz(method: str, M: int, K: int, seed: int, results_dir: str,
              constraint_handler: str = "none",
              c_violation_rates=None) -> str:
    """Drop a schema-valid synthetic npz at the canonical filename."""
    from baselines import write_result_npz, NpzResult

    rng = np.random.RandomState(seed * 10 + (M * K))
    eval_steps = np.arange(20_000, 1_020_000, 20_000, dtype=np.int64)
    hv = (400e9 + rng.randn(len(eval_steps)) * 20e9).astype(np.float64)
    ut = (800.0 + rng.randn(len(eval_steps)) * 5.0).astype(np.float64)
    sparsity = (1000.0 + rng.randn(len(eval_steps)) * 200.0).astype(np.float64)

    P = 56  # Pareto front size
    final_ep_objs = (
        np.array([700, 650, 750, 800], dtype=np.float64)
        + rng.randn(P, 4) * 30.0
    )
    final_ep_prefs = rng.dirichlet(np.ones(4), size=P)

    if c_violation_rates is None:
        c_violation_rates = np.full(3, np.nan)

    return write_result_npz(
        result=NpzResult(
            eval_steps=eval_steps,
            hv_trajectory=hv,
            ut_trajectory=ut,
            sparsity_trajectory=sparsity,
            final_ep_objs=final_ep_objs,
            final_ep_prefs=final_ep_prefs,
            final_obj_means=final_ep_objs.mean(axis=0),
            final_obj_stds=final_ep_objs.std(axis=0),
            c_violation_rates=np.asarray(c_violation_rates, dtype=np.float64),
            wallclock_seconds=12345.0,
            git_sha="deadbeef",
            config_dict={"baseline": method, "seed": seed},
        ),
        output_dir=results_dir,
        method=method, M=M, K=K, seed=seed,
        constraint_handler=constraint_handler if constraint_handler != "none" else None,
    )


def main():
    print("=== step 1: synthesise results dir ===")
    tmpdir = tempfile.mkdtemp(prefix="smoketest_compare_")
    try:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # 3 methods × 3 seeds at the M=2 K=5 anchor; one method (c-morl) is
        # constrained with populated violation rates.
        for seed in [1, 2, 3]:
            _fake_npz("mo-ppo", M=2, K=5, seed=seed, results_dir=results_dir)
            _fake_npz("pareto-pg", M=2, K=5, seed=seed, results_dir=results_dir)
            _fake_npz(
                "c-morl", M=2, K=5, seed=seed, results_dir=results_dir,
                constraint_handler="lagrangian",
                c_violation_rates=[0.04 + 0.01 * seed, 0.0, 0.05 - 0.01 * seed],
            )
        # One M=1 K=5 group with a single seed to verify multi-config grouping.
        _fake_npz("pareto-q", M=1, K=5, seed=1, results_dir=results_dir)

        npz_files = sorted(os.listdir(results_dir))
        assert len(npz_files) == 10, f"expected 10 files, got {len(npz_files)}"
        print(f"  created {len(npz_files)} npz files")

        print("\n=== step 2: load + group ===")
        from baselines import load_results_dir
        groups = load_results_dir(results_dir)
        assert len(groups) == 4, (
            f"expected 4 (method, config) groups, got {len(groups)}: "
            f"{[(g.method, g.M, g.K) for g in groups]}"
        )
        for g in groups:
            print(f"  {g.method:10s} M={g.M} K={g.K} seeds={g.n_seeds}")

        # Sanity check aggregate stats.
        m2_groups = [g for g in groups if g.M == 2]
        for g in m2_groups:
            assert g.n_seeds == 3, f"{g.method} expected 3 seeds, got {g.n_seeds}"
            mean, std = g.final_hv_mean_std()
            assert np.isfinite(mean) and std >= 0, f"bad stats: {mean} {std}"

        # Constrained group should have populated violation rates; unconstrained NaN.
        cmorl = [g for g in groups if g.method == "c-morl"][0]
        vr_mean, vr_std = cmorl.violation_rates_mean_std()
        assert not np.any(np.isnan(vr_mean)), "c-morl should have populated violation rates"
        print(f"  c-morl violation rates (mean): {vr_mean.round(3).tolist()}")

        moppo = [g for g in groups if g.method == "mo-ppo"][0]
        vr_mean, _ = moppo.violation_rates_mean_std()
        assert np.all(np.isnan(vr_mean)), "mo-ppo violation rates should be NaN"
        print(f"  mo-ppo violation rates: {vr_mean.tolist()} (all NaN, expected)")

        print("\n=== step 3: run the analyzer end-to-end ===")
        out_dir = os.path.join(tmpdir, "compare_out")
        cmd = [
            sys.executable,
            os.path.join(ROOT, "experiments", "analysis", "compare_baselines.py"),
            "--results_dir", results_dir,
            "--output_dir", out_dir,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print("  STDOUT:", res.stdout)
            print("  STDERR:", res.stderr)
            raise AssertionError(f"analyzer exited {res.returncode}")
        print("  OK: analyzer exited 0")
        print(f"  stdout: {res.stdout.strip()}")

        print("\n=== step 4: verify outputs ===")
        for fname in ["comparison_table.md", "comparison_table.csv",
                      "violation_table.md", "pareto_front.png"]:
            path = os.path.join(out_dir, fname)
            assert os.path.exists(path), f"missing output: {fname}"
            size = os.path.getsize(path)
            assert size > 0, f"empty output: {fname}"
            print(f"  OK: {fname} ({size} bytes)")

        # CSV row-count: 4 method-config groups → 4 rows + header.
        csv_path = os.path.join(out_dir, "comparison_table.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4, f"csv row count: {len(rows)}"
        print(f"  CSV has {len(rows)} rows: methods={sorted({r['method'] for r in rows})}")

        # Markdown table should mention every method.
        with open(os.path.join(out_dir, "comparison_table.md")) as f:
            md = f.read()
        for m in ("mo-ppo", "pareto-pg", "c-morl", "pareto-q"):
            assert m in md, f"method {m!r} missing from comparison_table.md"
        print(f"  markdown table mentions all 4 methods")

        # Violation table should mention only the constrained method.
        with open(os.path.join(out_dir, "violation_table.md")) as f:
            vio = f.read()
        assert "c-morl" in vio, "violation table should include c-morl"
        assert "mo-ppo" not in vio, "violation table should NOT include unconstrained mo-ppo"
        print(f"  violation table includes c-morl only (not mo-ppo)")

        print("\nAll comparison-analyzer smoke checks passed.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
