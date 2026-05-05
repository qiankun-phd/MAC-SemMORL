#!/usr/bin/env python3
"""
Compare COLA final ``objs`` (preference sweep) vs heuristic baseline mean objective vectors.

Uses the same HV / UT / sparsity as training: ``agent.evluate_Hv_UT_and_spa`` and
``generate_w_batch_test`` with step_size=0.2 for 4 objectives (matches ``SacAgent.run``).

Usage (repo root):
  python experiments/compare_mobility_cola_vs_baselines.py \\
    --mobility line \\
    --cola_summary_dir logs/uav/cola_mob_line_seed1/summary

  # Multiple seeds (reports HV mean ± std):
  python experiments/compare_mobility_cola_vs_baselines.py \\
    --mobility none \\
    --cola_summary_dir logs/uav/cola_mob_none_seed1/summary \\
    --cola_summary_dir logs/uav/cola_mob_none_seed2/summary

  python experiments/compare_mobility_cola_vs_baselines.py \\
    --mobility drift \\
    --baseline_npz logs/uav/mob_drift/baseline_results_4obj.npz \\
    --out_csv logs/uav/mob_drift/cola_vs_baselines.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Heavy import (torch, etc.) — matches training-side metrics exactly.
from agent import (  # noqa: E402
    check_dominated,
    evluate_Hv_UT_and_spa,
    generate_w_batch_test,
    get_ep_indices,
)


def latest_objs_path(summary_dir: str) -> str | None:
    pattern = os.path.join(summary_dir, "objs_*.npy")
    files = glob.glob(pattern)
    if not files:
        return None

    def step_key(path: str) -> int:
        base = os.path.basename(path)
        num = base.replace("objs_", "").replace(".npy", "")
        return int(num)

    return max(files, key=step_key)


def load_baseline_means(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    out = {}
    for k in data.files:
        arr = np.asarray(data[k], dtype=np.float64).ravel()
        out[k] = arr
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobility",
        type=str,
        default="none",
        help="Label for logging only (paths default under logs/uav/mob_<mobility>/).",
    )
    parser.add_argument(
        "--baseline_npz",
        type=str,
        default=None,
        help="Default: logs/uav/mob_<mobility>/baseline_results_4obj.npz",
    )
    parser.add_argument(
        "--cola_summary_dir",
        action="append",
        default=[],
        help="Repeat for each seed; uses latest objs_*.npy in each directory.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional CSV path for the summary table.",
    )
    args = parser.parse_args()

    mob = args.mobility
    baseline_npz = args.baseline_npz or os.path.join(
        ROOT, "logs", "uav", f"mob_{mob}", "baseline_results_4obj.npz"
    )
    if not os.path.isfile(baseline_npz):
        print(f"Missing baseline npz: {baseline_npz}", file=sys.stderr)
        print("Run: python baselines.py --device_mobility", mob, file=sys.stderr)
        sys.exit(1)

    if not args.cola_summary_dir:
        print("Provide at least one --cola_summary_dir", file=sys.stderr)
        sys.exit(1)

    baselines = load_baseline_means(baseline_npz)
    n_obj = next(iter(baselines.values())).shape[0]
    step_size = {2: 0.005, 3: 0.1, 4: 0.2, 5: 0.25, 6: 0.3}.get(n_obj)
    if step_size is None:
        raise SystemExit(f"Unsupported objective count {n_obj} for preference grid")
    pref_grid = generate_w_batch_test(n_obj, step_size)

    cola_runs = []
    for sdir in args.cola_summary_dir:
        path = latest_objs_path(sdir)
        if path is None:
            print(f"No objs_*.npy under {sdir}", file=sys.stderr)
            sys.exit(1)
        objs = np.load(path)
        if objs.ndim != 2 or objs.shape[1] != n_obj:
            raise SystemExit(f"Bad shape {objs.shape} in {path}, expected (N,{n_obj})")
        cola_runs.append({"dir": sdir, "path": path, "objs": objs})

    union_objs = (
        np.vstack([r["objs"] for r in cola_runs]) if cola_runs else np.zeros((0, n_obj))
    )

    rows = []
    hv_list = []
    for run in cola_runs:
        objs = run["objs"]
        hv, sp, ut = evluate_Hv_UT_and_spa(n_obj, objs, pref_grid)
        nd_idx = get_ep_indices(objs)
        nds = objs[nd_idx] if len(nd_idx) else objs[:0]
        hv_nd, _, _ = (
            evluate_Hv_UT_and_spa(n_obj, nds, pref_grid)
            if len(nds)
            else (0.0, 0.0, 0.0)
        )
        hv_list.append(hv)
        row = {
            "kind": "COLA",
            "name": os.path.basename(os.path.dirname(run["dir"])),
            "objs_path": run["path"],
            "HV": hv,
            "HV_nd": hv_nd,
            "UT": ut,
            "Sparsity": sp,
            "n_points": len(objs),
            "n_nd": len(nd_idx),
            "dominated_by_cola": "",
        }
        rows.append(row)

    # Baselines: single-point batches
    for bname, vec in sorted(baselines.items()):
        if vec.shape[0] != n_obj:
            continue
        batch = vec.reshape(1, -1)
        hv_b, sp_b, ut_b = evluate_Hv_UT_and_spa(n_obj, batch, pref_grid)
        dom = bool(check_dominated(union_objs, vec)) if len(union_objs) else False
        rows.append(
            {
                "kind": "baseline",
                "name": bname,
                "objs_path": baseline_npz,
                "HV": hv_b,
                "HV_nd": hv_b,
                "UT": ut_b,
                "Sparsity": sp_b,
                "n_points": 1,
                "n_nd": 1,
                "dominated_by_cola": dom,
            }
        )

    print(f"mobility={mob}  baseline_npz={baseline_npz}")
    print(f"COLA runs: {len(cola_runs)}  preference_grid: {len(pref_grid)} points")
    if len(hv_list) == 1:
        print(f"COLA HV (latest eval objs): {hv_list[0]:.6g}")
    else:
        m, s = float(np.mean(hv_list)), float(np.std(hv_list))
        print(f"COLA HV mean±std over seeds: {m:.6g} ± {s:.6g}")
    print("-" * 100)
    hdr = (
        f"{'kind':<10} {'name':<42} {'HV':>14} {'UT':>10} {'Sparsity':>10} "
        f"{'n_nd':>6} {'dominated_by_cola':>18}"
    )
    print(hdr)
    print("-" * 100)
    for r in rows:
        dom_s = ""
        if r["kind"] == "baseline":
            dom_s = str(r.get("dominated_by_cola", ""))
        print(
            f"{r['kind']:<10} {r['name']:<42} {r['HV']:>14.6g} {r['UT']:>10.3f} "
            f"{r['Sparsity']:>10.3f} {r['n_nd']:>6} {dom_s:>18}"
        )

    if args.out_csv:
        out_abs = os.path.abspath(args.out_csv)
        odir = os.path.dirname(out_abs)
        if odir:
            os.makedirs(odir, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(out_abs, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_abs}")


if __name__ == "__main__":
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
    main()
