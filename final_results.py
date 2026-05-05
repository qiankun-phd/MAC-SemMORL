"""Comprehensive final results for 4-objective UAV-SemCom GLOBECOM paper."""
import re, os, glob
import numpy as np

log_dir = "logs/uav"
OBJ_NAMES = ["SemFid", "Fresh", "Energy", "Fair"]
N_OBJ = 4

pattern = re.compile(r"episode:\s+(\d+)\s+.*?rl reward:\s*\[([^\]]+)\]")


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
    files = sorted(glob.glob(os.path.join(summary_dir, "objs_*.npy")))
    if not files:
        return None
    if step == "final":
        return np.load(files[-1])
    for f in files:
        if f"objs_{step}" in f:
            return np.load(f)
    return np.load(files[-1])


def load_comm_metrics(exp_name, step="final"):
    summary_dir = os.path.join(log_dir, exp_name, "summary")
    if not os.path.isdir(summary_dir):
        return None
    files = sorted(glob.glob(os.path.join(summary_dir, "comm_metrics_*.npz")))
    if not files:
        return None
    f = files[-1] if step == "final" else files[-1]
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


SEEDS = {
    "COLA Full": ["cola_full_s1", "cola_full_s2", "cola_full_s3"],
    "No COR": ["no_cor_s1", "no_cor_s2", "no_cor_s3"],
    "No OADM": ["no_oadm_s1", "no_oadm_s2", "no_oadm_s3"],
    "Envelope SAC": ["envelope_s1", "envelope_s2", "envelope_s3"],
}
WS = {
    "WS-Fidelity": "ws_fidelity_s1",
    "WS-Balanced": "ws_balanced_s1",
    "WS-Energy": "ws_energy_s1",
}

print("=" * 100)
print("FINAL RESULTS — 4-Objective UAV Semantic Communication (200K steps)")
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

for method, log_name in WS.items():
    m, s, _ = parse_log(log_name, last_n=50)
    method_means[method] = m
    vals = "".join(f" {m[i]:>5.1f}±{s[i]:<4.1f}" for i in range(N_OBJ))
    print(f"{method:<24}{vals} {m.sum():>10.1f}")

print("-" * 100)
print("Heuristic Baselines:")
print("-" * 100)

heur_file = os.path.join(log_dir, "baseline_results_4obj.npz")
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

for method, log_name in WS.items():
    objs = load_objs(log_name)
    if objs is not None:
        hv, ut, sp = compute_hv_ut_sp(objs)
        print(f"{method:<24} {hv:>14.0f} {ut:>10.1f} {sp:>10.1f}")

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

for method, log_name in WS.items():
    cm = load_comm_metrics(log_name)
    if cm:
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
