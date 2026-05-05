"""Comprehensive final results for 4-objective UAV-SemCom GLOBECOM paper (line mobility)."""
import glob
import os
import re

import numpy as np

log_dir = "logs/uav"
OBJ_NAMES = ["SemFid", "Fresh", "Energy", "Fair"]
N_OBJ = 4

pattern = re.compile(r"episode:\s+(\d+)\s+.*?rl reward:\s*\[([^\]]+)\]")
step_pattern = re.compile(r"_(\d+)\.(?:npy|npz)$")


def file_step(path):
    match = step_pattern.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def latest_step_file(files):
    return max(files, key=file_step) if files else None


def parse_log(name, last_n=50):
    path = os.path.join(log_dir, f"{name}.log")
    episodes = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                vals = [float(x) for x in m.group(2).split()]
                if len(vals) >= N_OBJ:
                    episodes.append(vals[:N_OBJ])
    eps = np.array(episodes[-last_n:])
    return eps.mean(axis=0), eps.std(axis=0), len(episodes)


def load_objs(exp_name, step="final"):
    """Load Pareto front objs from the latest or specified step."""
    summary_dir = os.path.join(log_dir, exp_name, "summary")
    if not os.path.isdir(summary_dir):
        return None
    files = glob.glob(os.path.join(summary_dir, "objs_*.npy"))
    if not files:
        return None
    if step == "final":
        return np.load(latest_step_file(files))
    for f in files:
        if f"objs_{step}" in f:
            return np.load(f)
    return np.load(latest_step_file(files))


def load_comm_metrics(exp_name, step="final"):
    summary_dir = os.path.join(log_dir, exp_name, "summary")
    if not os.path.isdir(summary_dir):
        return None
    files = glob.glob(os.path.join(summary_dir, "comm_metrics_*.npz"))
    if not files:
        return None
    if step == "final":
        f = latest_step_file(files)
    else:
        f = next((p for p in files if f"comm_metrics_{step}" in p), None)
        if f is None:
            f = latest_step_file(files)
    return dict(np.load(f))


def compute_hv_ut_sp(objs, ref_point=None):
    """Compute hypervolume, utility, sparsity from Pareto front objectives."""
    if objs is None or len(objs) == 0:
        return 0, 0, 0
    n_obj = objs.shape[1]
    if ref_point is None:
        ref_point = np.zeros(n_obj)
    shifted = objs - ref_point
    shifted = shifted[np.all(shifted > 0, axis=1)]
    if len(shifted) == 0:
        return 0, 0, 0
    vol = float(np.prod(shifted, axis=1).sum())
    ut = float(np.min(shifted, axis=1).mean())
    if len(shifted) > 1:
        sorted_objs = shifted[shifted[:, 0].argsort()]
        diffs = np.diff(sorted_objs, axis=0)
        sp = float(np.sum(diffs ** 2))
    else:
        sp = 0
    return vol, ut, sp


DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6]


def existing_exps(prefix, seeds=DEFAULT_SEEDS):
    exps = []
    for s in seeds:
        name = f"{prefix}_line_s{s}"
        if os.path.exists(os.path.join(log_dir, f"{name}.log")):
            exps.append(name)
    return exps


SEEDS = {
    "COLA Full": existing_exps("cola"),
    "No COR": existing_exps("no_cor"),
    "No OADM": existing_exps("no_oadm"),
    "Envelope SAC": existing_exps("envelope"),
}
WS = {
    "WS-Fidelity": existing_exps("ws_fidelity"),
    "WS-Balanced": existing_exps("ws_balanced"),
    "WS-Energy": existing_exps("ws_energy"),
}
COMM_BASELINE_DIRS = {
    "FT-GP": "mob_line/baseline_ft_gp",
    "Greedy AoSI": "mob_line/baseline_greedy_aosi",
    "Round Robin": "mob_line/baseline_round_robin",
    "Random": "mob_line/baseline_random",
}

print("=" * 100)
print("FINAL RESULTS — 4-Objective UAV Semantic Communication LINE mobility (200K steps)")
print("=" * 100)

header = f"{'Method':<24}" + "".join(f" {n:>10}" for n in OBJ_NAMES) + f" {'Sum':>10}"
print(f"\n{header}")
print("-" * 100)

method_means = {}
for method, seeds in SEEDS.items():
    seed_means = []
    for s in seeds:
        m, _, _ = parse_log(s, last_n=50)
        seed_means.append(m)
    arr = np.array(seed_means)
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    method_means[method] = avg
    vals = "".join(f" {avg[i]:>5.1f}±{std[i]:<4.1f}" for i in range(N_OBJ))
    print(f"{method + ' (avg±std)':<24}{vals} {avg.sum():>5.1f}±{std.sum():.1f}")

for method, seeds in WS.items():
    seed_means = []
    for s in seeds:
        m, _, _ = parse_log(s, last_n=50)
        seed_means.append(m)
    if not seed_means:
        continue
    arr = np.array(seed_means)
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    method_means[method] = avg
    vals = "".join(f" {avg[i]:>5.1f}+-{std[i]:<4.1f}" for i in range(N_OBJ))
    print(f"{method + ' (avg+-std)':<24}{vals} {avg.sum():>5.1f}+-{std.sum():.1f}")

print("-" * 100)
print("Heuristic Baselines:")
print("-" * 100)

heur_file = os.path.join(log_dir, "mob_line", "baseline_results_4obj.npz")
if os.path.exists(heur_file):
    heur = np.load(heur_file)
    for key in heur.files:
        m = heur[key]
        name = key.replace("_", " ")[:24]
        vals = "".join(f" {m[i]:>10.1f}" for i in range(min(N_OBJ, len(m))))
        print(f"{name:<24}{vals} {m.sum():>10.1f}")

cola_avg = method_means["COLA Full"]

print(f"\n{'=' * 100}")
print("ABLATION ANALYSIS (vs COLA Full avg)")
print(f"{'=' * 100}")
for method in ["No COR", "No OADM", "Envelope SAC"]:
    m = method_means[method]
    diff = m - cola_avg
    pct = diff / cola_avg * 100
    print(f"\n  {method}:")
    for i in range(N_OBJ):
        print(f"    {OBJ_NAMES[i]:>8}: {diff[i]:+.1f} ({pct[i]:+.1f}%)")
    print(f"    {'Sum':>8}: {diff.sum():+.1f} ({diff.sum()/cola_avg.sum()*100:+.1f}%)")

print(f"\n{'=' * 100}")
print("HYPERVOLUME / UTILITY / SPARSITY (from Pareto front evaluation)")
print(f"{'=' * 100}")
print(f"{'Method':<24} {'HV':>14} {'UT':>10} {'Sparsity':>10}")
print("-" * 60)

for method, seeds in SEEDS.items():
    hvs, uts, sps = [], [], []
    for s in seeds:
        objs = load_objs(s)
        if objs is not None:
            hv, ut, sp = compute_hv_ut_sp(objs)
            hvs.append(hv)
            uts.append(ut)
            sps.append(sp)
    if hvs:
        print(f"{method:<24} {np.mean(hvs):>10.0f}±{np.std(hvs):<5.0f}"
              f"{np.mean(uts):>7.1f}±{np.std(uts):<4.1f}"
              f"{np.mean(sps):>7.1f}±{np.std(sps):<4.1f}")

for method, seeds in WS.items():
    hvs, uts, sps = [], [], []
    for s in seeds:
        objs = load_objs(s)
        if objs is not None:
            hv, ut, sp = compute_hv_ut_sp(objs)
            hvs.append(hv)
            uts.append(ut)
            sps.append(sp)
    if hvs:
        print(f"{method:<24} {np.mean(hvs):>10.0f}+-{np.std(hvs):<5.0f}"
              f"{np.mean(uts):>7.1f}+-{np.std(uts):<4.1f}"
              f"{np.mean(sps):>7.1f}+-{np.std(sps):<4.1f}")

print(f"\n{'=' * 100}")
print("RAW COMMUNICATION METRICS (avg over Pareto front preferences)")
print(f"{'=' * 100}")
print(f"{'Method':<24} {'AvgFidelity':>12} {'MeanAoSI':>10} {'Energy(J)':>10} {'JainIdx':>10} {'SvcRate':>10}")
print("-" * 80)

for method, seeds in SEEDS.items():
    all_cm = {k: [] for k in ["weighted_avg_fidelity", "mean_aosi", "energy", "jain_fairness", "service_rate"]}
    for s in seeds:
        cm = load_comm_metrics(s)
        if cm:
            for k in all_cm:
                if k in cm:
                    all_cm[k].append(np.mean(cm[k]))
    if all_cm["weighted_avg_fidelity"]:
        fid = np.mean(all_cm["weighted_avg_fidelity"])
        aosi = np.mean(all_cm["mean_aosi"])
        eng = np.mean(all_cm["energy"])
        jain = np.mean(all_cm["jain_fairness"])
        sr = np.mean(all_cm["service_rate"])
        print(f"{method:<24} {fid:>12.4f} {aosi:>10.2f} {eng:>10.1f} {jain:>10.4f} {sr:>10.4f}")

for method, seeds in WS.items():
    all_cm = {k: [] for k in ["weighted_avg_fidelity", "mean_aosi", "energy", "jain_fairness", "service_rate"]}
    for s in seeds:
        cm = load_comm_metrics(s)
        if cm:
            for k in all_cm:
                if k in cm:
                    all_cm[k].append(np.mean(cm[k]))
    if all_cm["weighted_avg_fidelity"]:
        fid = np.mean(all_cm["weighted_avg_fidelity"])
        aosi = np.mean(all_cm["mean_aosi"])
        eng = np.mean(all_cm["energy"])
        jain = np.mean(all_cm["jain_fairness"])
        sr = np.mean(all_cm["service_rate"])
        print(f"{method:<24} {fid:>12.4f} {aosi:>10.2f} {eng:>10.1f} {jain:>10.4f} {sr:>10.4f}")

for method, exp_name in COMM_BASELINE_DIRS.items():
    path = os.path.join(log_dir, exp_name, "summary", "comm_metrics_baseline.npz")
    if not os.path.exists(path):
        continue
    cm = dict(np.load(path))
    fid = np.mean(cm.get("weighted_avg_fidelity", [0]))
    aosi = np.mean(cm.get("mean_aosi", [0]))
    eng = np.mean(cm.get("energy", [0]))
    jain = np.mean(cm.get("jain_fairness", [0]))
    sr = np.mean(cm.get("service_rate", [0]))
    print(f"{method:<24} {fid:>12.4f} {aosi:>10.2f} {eng:>10.1f} {jain:>10.4f} {sr:>10.4f}")

print(f"\n{'=' * 100}")
print("REWARD BALANCE VERIFICATION")
print(f"{'=' * 100}")
per_step = cola_avg / 200.0
print(f"COLA per-step rewards: " + " ".join(f"{OBJ_NAMES[i]}={per_step[i]:.2f}" for i in range(N_OBJ)))
ratios = [cola_avg[i] / cola_avg[j] for i in range(N_OBJ) for j in range(N_OBJ) if i != j]
all_ok = all(0.4 < r < 2.5 for r in ratios)
print(f"Objective balance: {'PASSED' if all_ok else 'CHECK NEEDED'}")
print(f"Max ratio: {max(ratios):.2f}, Min ratio: {min(ratios):.2f}")
