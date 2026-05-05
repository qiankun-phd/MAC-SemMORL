"""Generate IEEE-compliant publication figures for GLOBECOM paper (line mobility).

IEEE formatting rules applied:
  - 8 pt Times New Roman for all figure labels, ticks, legends
  - Full words for axis labels (no abbreviations)
  - Units in parentheses, e.g. "Training Steps (x1000)"
  - No ratio-style axis labels
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- IEEE style: 8pt Times New Roman (via STIX) ----------
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

LOG_DIR = "logs/uav"
FIG_DIR = os.path.join("figures", "line")
os.makedirs(FIG_DIR, exist_ok=True)

N_OBJ = 4

# Full descriptive labels (IEEE: use words, not abbreviations)
OBJ_LABEL = [
    "Semantic Fidelity",
    "Information Freshness",
    "Energy Efficiency",
    "Coverage Fairness",
]

# Shorter Y-axis labels for subplots (avoid truncation)
OBJ_YLABEL = [
    "Semantic\nFidelity",
    "Information\nFreshness",
    "Energy\nEfficiency",
    "Coverage\nFairness",
]

COLORS = {
    "COLA":     "#D32F2F",
    "No COR":   "#1976D2",
    "No OADM":  "#388E3C",
    "Envelope": "#F57C00",
    "WS-Fidelity":  "#7B1FA2",
    # Panel (b): avoid brown/teal pairs that look alike on screen/print
    "WS-Balanced":  "#2E7D32",
    "WS-Energy":    "#E65100",
    "FT-GP":        "#546E7A",
    "Greedy AoSI":  "#AD1457",
    "Round Robin":  "#4527A0",
    "Random":       "#78909C",
}
MARKERS = {
    "COLA": "o", "No COR": "s", "No OADM": "^", "Envelope": "D",
    "WS-Fidelity": "v", "WS-Balanced": "<", "WS-Energy": ">",
    "FT-GP": "P", "Greedy AoSI": "X", "Round Robin": "h", "Random": "d",
}

DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6]
STEP_PATTERN = re.compile(r"_(\d+)\.(?:npy|npz)$")


def file_step(path):
    match = STEP_PATTERN.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def latest_step_file(files):
    return max(files, key=file_step) if files else None


def existing_exps(prefix, seeds=DEFAULT_SEEDS):
    exps = []
    for s in seeds:
        name = f"{prefix}_line_s{s}"
        if os.path.exists(os.path.join(LOG_DIR, f"{name}.log")):
            exps.append(name)
    return exps


SEEDS = {
    "COLA":     existing_exps("cola"),
    "No COR":   existing_exps("no_cor"),
    "No OADM":  existing_exps("no_oadm"),
    "Envelope": existing_exps("envelope"),
}


_DISPLAY_LABEL = {
    "COLA":         "SemMORL",
    "No COR":       "SemMORL w/o COR",
    "No OADM":      "SemMORL w/o OADM",
    "Envelope":     "Envelope SAC",
    "WS-Fidelity":  "WS-Fid.",
    "WS-Balanced":  "WS-Bal.",
    "WS-Energy":    "WS-Eng.",
    "FT-GP":        "FT-GP",
    "Greedy AoSI":  "Greedy AoSI",
    "Round Robin":  "Round Robin",
    "Random":       "Random",
}


def legend_label(method_key):
    """Unified paper-facing display name for all figure legends."""
    return _DISPLAY_LABEL.get(method_key, method_key)


comm_legend_label = legend_label
WS_EXPS = {
    "WS-Fidelity": existing_exps("ws_fidelity"),
    "WS-Balanced": existing_exps("ws_balanced"),
    "WS-Energy":   existing_exps("ws_energy"),
}

# Heuristic comm_metrics written by ``python baselines.py`` (summary/comm_metrics_baseline.npz).
COMM_BASELINE_DIRS = {
    "FT-GP": "mob_line/baseline_ft_gp",
    "Greedy AoSI": "mob_line/baseline_greedy_aosi",
    "Round Robin": "mob_line/baseline_round_robin",
    "Random": "mob_line/baseline_random",
}

IEEE_COL_W = 3.5   # single-column width (inches) for IEEE 2-column
IEEE_DBL_W = 7.16  # double-column width (inches)


def load_training_curve(exp_name):
    pattern = re.compile(r"episode:\s+(\d+)\s+.*?rl reward:\s*\[([^\]]+)\]")
    path = os.path.join(LOG_DIR, f"{exp_name}.log")
    steps, rewards = [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep = int(m.group(1))
                vals = [float(x) for x in m.group(2).split()]
                if len(vals) >= N_OBJ:
                    steps.append(ep * 200)
                    rewards.append(vals[:N_OBJ])
    return np.array(steps), np.array(rewards)


def smooth(y, window=20):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


# =====================================================================
# Figure 1: Training Convergence Curves
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(IEEE_DBL_W, 4.5), sharex=True)

for method, seeds in SEEDS.items():
    all_curves = []
    min_len = int(1e9)
    for s in seeds:
        _, rw = load_training_curve(s)
        all_curves.append(rw)
        min_len = min(min_len, len(rw))
    arr = np.array([c[:min_len] for c in all_curves])
    steps_arr = np.arange(min_len) * 200

    for obj_i in range(N_OBJ):
        ax = axes[obj_i // 2][obj_i % 2]
        mean = smooth(arr[:, :, obj_i].mean(axis=0))
        std = smooth(arr[:, :, obj_i].std(axis=0))
        x = np.linspace(0, steps_arr[-1], len(mean))
        ax.plot(x / 1000, mean, color=COLORS[method], label=legend_label(method))
        ax.fill_between(x / 1000, mean - std, mean + std,
                         alpha=0.12, color=COLORS[method])

for method, exp_list in WS_EXPS.items():
    if not exp_list:
        continue
    all_curves = []
    min_len = int(1e9)
    for exp in exp_list:
        _, rw = load_training_curve(exp)
        all_curves.append(rw)
        min_len = min(min_len, len(rw))
    arr = np.array([c[:min_len] for c in all_curves])
    steps_arr = np.arange(min_len) * 200
    for obj_i in range(N_OBJ):
        ax = axes[obj_i // 2][obj_i % 2]
        y = smooth(arr[:, :, obj_i].mean(axis=0))
        x = np.linspace(0, steps_arr[-1], len(y))
        ax.plot(x / 1000, y, color=COLORS[method], label=legend_label(method),
                linewidth=0.7, linestyle="--")

fig1_ylabel = [
    "Semantic Fidelity",
    "Information Freshness",
    "Energy Efficiency",
    "Coverage Fairness",
]
for obj_i in range(N_OBJ):
    ax = axes[obj_i // 2][obj_i % 2]
    ax.set_ylabel(fig1_ylabel[obj_i])
    ax.grid(True, alpha=0.25)
    if obj_i >= 2:
        ax.set_xlabel("Training Steps (\u00d71000)")

handles, labels = axes[0][0].get_legend_handles_labels()
seen = {}
unique_h, unique_l = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = True
        unique_h.append(h)
        unique_l.append(l)

fig.legend(unique_h, unique_l, ncol=4, fontsize=7,
           framealpha=0.9, edgecolor="none",
           loc="upper center", bbox_to_anchor=(0.5, 0.04))
plt.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.13,
                     hspace=0.15, wspace=0.25)
plt.savefig(os.path.join(FIG_DIR, "fig1_training_curves.pdf"))
plt.savefig(os.path.join(FIG_DIR, "fig1_training_curves.png"))
plt.close()
print("[1/4] Training curves saved.")


# =====================================================================
# Figure 2: Pareto Front (Pairwise 2D Projections)
# =====================================================================
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

fig, axes = plt.subplots(2, 3, figsize=(IEEE_DBL_W, 4.2))
axes_flat = axes.flatten()

for idx, (a, b) in enumerate(pairs):
    ax = axes_flat[idx]
    for method, seeds in SEEDS.items():
        all_a, all_b = [], []
        for s in seeds:
            summary_dir = os.path.join(LOG_DIR, s, "summary")
            files = glob.glob(os.path.join(summary_dir, "objs_*.npy"))
            if files:
                objs = np.load(latest_step_file(files))
                all_a.extend(objs[:, a].tolist())
                all_b.extend(objs[:, b].tolist())
        if all_a:
            ax.scatter(all_a, all_b, c=COLORS[method], marker=MARKERS[method],
                       s=10, alpha=0.4, label=legend_label(method), edgecolors="none")
    ax.set_xlabel(OBJ_LABEL[a])
    ax.set_ylabel(OBJ_LABEL[b])
    ax.grid(True, alpha=0.2)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7,
           frameon=True, framealpha=0.9, edgecolor="none",
           bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(h_pad=0.6, w_pad=0.5)
plt.savefig(os.path.join(FIG_DIR, "fig2_pareto_front.pdf"))
plt.savefig(os.path.join(FIG_DIR, "fig2_pareto_front.png"))
plt.close()
print("[2/4] Pareto front (2D pairwise) saved.")


# =====================================================================
# Figure 3: Bar Chart — Overall Performance Comparison
# =====================================================================
methods_order = [
    "COLA", "No COR", "No OADM", "Envelope",
    "WS-Fidelity", "WS-Balanced", "WS-Energy",
    "FT-GP", "Greedy AoSI", "Round Robin", "Random",
]
method_display = [
    "SemMORL", "SemMORL\nw/o COR", "SemMORL\nw/o OADM", "Envelope SAC",
    "WS-Fid.", "WS-Bal.", "WS-Eng.",
    "FT-GP", "Greedy\nAoSI", "Round\nRobin", "Random",
]

pattern_log = re.compile(r"episode:\s+(\d+)\s+.*?rl reward:\s*\[([^\]]+)\]")

def get_mean_std(exp_list, last_n=50):
    seed_means = []
    for name in exp_list:
        path = os.path.join(LOG_DIR, f"{name}.log")
        if not os.path.exists(path):
            continue
        eps = []
        with open(path) as f:
            for line in f:
                m = pattern_log.search(line)
                if m:
                    v = [float(x) for x in m.group(2).split()]
                    if len(v) >= N_OBJ:
                        eps.append(v[:N_OBJ])
        if not eps:
            continue
        arr = np.array(eps[-last_n:])
        seed_means.append(arr.mean(axis=0))
    if not seed_means:
        return np.zeros(N_OBJ), np.zeros(N_OBJ)
    a = np.array(seed_means)
    return a.mean(axis=0), a.std(axis=0)


def load_comm_avg(exp_list):
    """Aggregate raw communication metrics from experiment summary npz (MORL runs).

    Returns one row per method (before dividing by COLA): fidelity [0,1], proxy
    freshness 1/mean_AoSI, *inverse* total episode energy 1/E_kJ (so larger =
    less energy), Jain index, service rate. The energy entry is **not** the RL
    reward channel ``Energy Efficiency'' in the env (that is a normalized score);
    it is derived from logged total kJ only, so Envelope SAC can look much worse
    here than on the RL energy reward while still being consistent with Table~III
    kJ columns.
    """
    vals = {k: [] for k in [
        "weighted_avg_fidelity", "mean_aosi", "energy",
        "jain_fairness", "service_rate"]}
    for s in exp_list:
        summary_dir = os.path.join(LOG_DIR, s, "summary")
        files = glob.glob(os.path.join(summary_dir, "comm_metrics_*.npz"))
        if files:
            cm = dict(np.load(latest_step_file(files)))
            for k in vals:
                if k in cm:
                    vals[k].append(np.mean(cm[k]))
    if not vals["weighted_avg_fidelity"]:
        return None
    fid = np.mean(vals["weighted_avg_fidelity"])
    freshness = 1.0 / max(np.mean(vals["mean_aosi"]), 0.01)
    e_mean = max(np.mean(vals["energy"]), 1.0)
    inv_energy = 1.0 / e_mean  # same as E_COLA/E_method after ratioing vs COLA
    jain = np.mean(vals["jain_fairness"])
    sr = np.mean(vals["service_rate"])
    return np.array([fid, freshness, inv_energy, jain, sr])


bar_data = {}
for method, seeds in SEEDS.items():
    bar_data[method] = get_mean_std(seeds)
for method, exp_list in WS_EXPS.items():
    bar_data[method] = get_mean_std(exp_list)

heur_file = os.path.join(LOG_DIR, "mob_line", "baseline_results_4obj.npz")
if os.path.exists(heur_file):
    heur = np.load(heur_file)
    heur_map = {
        "FT-GP": "FT-GP_(Fixed_Traj_+_Greedy_Power)",
        "Greedy AoSI": "Greedy_AoSI",
        "Round Robin": "Round_Robin",
        "Random": "Random",
    }
    for display, key in heur_map.items():
        if key in heur:
            bar_data[display] = (heur[key], np.zeros(N_OBJ))

radar_raw = {}
for method, seeds in SEEDS.items():
    v = load_comm_avg(seeds)
    if v is not None:
        radar_raw[method] = v
for method, exp_list in WS_EXPS.items():
    v = load_comm_avg(exp_list)
    if v is not None:
        radar_raw[method] = v
for method, sub in COMM_BASELINE_DIRS.items():
    v = load_comm_avg([sub])
    if v is not None:
        radar_raw[method] = v


def minmax_normalize_columns(M):
    """Each column of M (n_methods x n_metrics) scaled to [0, 1] across rows."""
    out = np.zeros_like(M, dtype=float)
    for k in range(M.shape[1]):
        c = M[:, k].astype(float)
        lo, hi = float(c.min()), float(c.max())
        if hi - lo < 1e-12:
            out[:, k] = 1.0
        else:
            out[:, k] = (c - lo) / (hi - lo)
    return out


bar_colors = ["#D32F2F", "#F57C00", "#388E3C", "#1976D2"]

comm_bar_labels = [
    "Semantic\nFidelity",
    "Information\nFreshness",
    "Episode\nEnergy",
    "Jain\nFairness",
    "Service\nRate",
]

# ----- Figures 3+4 (combined): reward bars + comm-metric bars (RL logs + baselines.py heuristics) -----
fig = plt.figure(figsize=(IEEE_DBL_W, 7.35))
gs = fig.add_gridspec(2, 1, height_ratios=[1.22, 1.12], hspace=0.40)
ax1 = fig.add_subplot(gs[0, 0])

x = np.arange(len(methods_order))
width = 0.19
for obj_i in range(N_OBJ):
    means = [bar_data.get(m, (np.zeros(N_OBJ), np.zeros(N_OBJ)))[0][obj_i]
             for m in methods_order]
    stds  = [bar_data.get(m, (np.zeros(N_OBJ), np.zeros(N_OBJ)))[1][obj_i]
             for m in methods_order]
    offset = (obj_i - 1.5) * width
    ax1.bar(x + offset, means, width, yerr=stds, label=OBJ_LABEL[obj_i],
            color=bar_colors[obj_i], capsize=1.5, alpha=0.85,
            error_kw={"linewidth": 0.5})

ax1.set_xticks(x)
ax1.set_xticklabels(method_display, rotation=25, ha="right")
ax1.set_ylabel("Cumulative Episode Reward")
ax1.grid(True, axis="y", alpha=0.25)
ax1.axvline(x=6.5, color="gray", linestyle=":", linewidth=0.6)
ymax = ax1.get_ylim()[1]
ax1.text(3.0, ymax * 1.02, "RL-based Methods",
         ha="center", fontsize=7, fontstyle="italic", color="gray")
ax1.text(8.5, ymax * 1.02, "Heuristic Baselines",
         ha="center", fontsize=7, fontstyle="italic", color="gray")
ax1.legend(ncol=4, fontsize=7, framealpha=0.9, edgecolor="none",
           loc="upper center", bbox_to_anchor=(0.5, -0.16))
ax1.text(-0.06, 1.02, "(a)", transform=ax1.transAxes, fontsize=9,
         fontweight="bold", va="bottom")

ax2 = fig.add_subplot(gs[1, 0])
# Match Fig.~(a) method order; heuristics require ``baselines.py`` comm_metrics outputs.
comm_bar_order = [
    "COLA", "No COR", "No OADM", "Envelope",
    "WS-Fidelity", "WS-Balanced", "WS-Energy",
    "FT-GP", "Greedy AoSI", "Round Robin", "Random",
]
present_comm = [m for m in comm_bar_order if m in radar_raw]
if present_comm:
    M_raw = np.stack([radar_raw[m] for m in present_comm], axis=0)
    M_nm = minmax_normalize_columns(M_raw)
    comm_bar_vals = {m: M_nm[i] for i, m in enumerate(present_comm)}
    n_k = 5
    x2 = np.arange(n_k, dtype=float)
    n_b = len(present_comm)
    bw = min(0.11, 0.92 / max(n_b, 1))
    for i, method in enumerate(present_comm):
        off = (i - (n_b - 1) / 2.0) * bw
        edg = "white" if method == "COLA" else "none"
        ax2.bar(
            x2 + off,
            comm_bar_vals[method],
            bw,
            label=comm_legend_label(method),
            color=COLORS[method],
            alpha=0.88 if method != "COLA" else 0.55,
            edgecolor=edg,
            linewidth=0.3,
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(comm_bar_labels, fontsize=7)
    ax2.set_ylabel("Normalized score")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(
        ncol=6,
        fontsize=5.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        framealpha=0.92,
        edgecolor="none",
    )
else:
    ax2.text(0.5, 0.5, "(No comm_metrics summaries in logs)", ha="center",
             va="center", transform=ax2.transAxes, fontsize=8, color="gray")

ax2.text(-0.06, 1.02, "(b)", transform=ax2.transAxes, fontsize=9,
         fontweight="bold", va="bottom")

fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.17)
plt.savefig(os.path.join(FIG_DIR, "fig3_4_combined.pdf"))
plt.savefig(os.path.join(FIG_DIR, "fig3_4_combined.png"))
plt.close()
print("[3/4] Combined performance + communication metrics saved.")

# Stand-alone Fig. 3 (same width as before) for slides or legacy docs
fig, ax = plt.subplots(figsize=(IEEE_DBL_W, 3.6))
for obj_i in range(N_OBJ):
    means = [bar_data.get(m, (np.zeros(N_OBJ), np.zeros(N_OBJ)))[0][obj_i]
             for m in methods_order]
    stds  = [bar_data.get(m, (np.zeros(N_OBJ), np.zeros(N_OBJ)))[1][obj_i]
             for m in methods_order]
    offset = (obj_i - 1.5) * width
    ax.bar(x + offset, means, width, yerr=stds, label=OBJ_LABEL[obj_i],
           color=bar_colors[obj_i], capsize=1.5, alpha=0.85,
           error_kw={"linewidth": 0.5})
ax.set_xticks(x)
ax.set_xticklabels(method_display, rotation=25, ha="right")
ax.set_ylabel("Cumulative Episode Reward")
ax.grid(True, axis="y", alpha=0.25)
ax.axvline(x=6.5, color="gray", linestyle=":", linewidth=0.6)
ymax = ax.get_ylim()[1]
ax.text(3.0, ymax * 1.02, "RL-based Methods",
        ha="center", fontsize=7, fontstyle="italic", color="gray")
ax.text(8.5, ymax * 1.02, "Heuristic Baselines",
        ha="center", fontsize=7, fontstyle="italic", color="gray")
ax.legend(ncol=4, fontsize=7, framealpha=0.9, edgecolor="none",
          loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.subplots_adjust(bottom=0.25)
plt.savefig(os.path.join(FIG_DIR, "fig3_bar_comparison.pdf"))
plt.savefig(os.path.join(FIG_DIR, "fig3_bar_comparison.png"))
plt.close()


# =====================================================================
# Figure 5: Hypervolume Convergence
# =====================================================================
fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))
eval_steps = list(range(20000, 200001, 20000))

for method, seeds in SEEDS.items():
    hv_per_step = {s: [] for s in eval_steps}
    for s in seeds:
        summary_dir = os.path.join(LOG_DIR, s, "summary")
        for step in eval_steps:
            files = glob.glob(os.path.join(summary_dir, f"objs_{step}.npy"))
            if files:
                objs = np.load(files[0])
                vol = float(np.prod(np.maximum(objs, 0), axis=1).sum())
                hv_per_step[step].append(vol)

    steps_plot, means_plot, stds_plot = [], [], []
    for step in eval_steps:
        if hv_per_step[step]:
            steps_plot.append(step / 1000)
            means_plot.append(np.mean(hv_per_step[step]))
            stds_plot.append(np.std(hv_per_step[step]))

    if steps_plot:
        m_arr = np.array(means_plot)
        s_arr = np.array(stds_plot)
        x = np.array(steps_plot)
        ax.plot(x, m_arr, color=COLORS[method], marker=MARKERS[method],
                markersize=3, label=legend_label(method))
        ax.fill_between(x, m_arr - s_arr, m_arr + s_arr,
                         alpha=0.12, color=COLORS[method])

ax.set_xlabel("Training Steps (\u00d71000)")
ax.set_ylabel("Hypervolume")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_hv_convergence.pdf"))
plt.savefig(os.path.join(FIG_DIR, "fig5_hv_convergence.png"))
plt.close()
print("[4/4] HV convergence saved.")

print(f"\nAll line figures saved to {FIG_DIR}/")
