"""Quick training progress check from log files."""
import re, sys, os
import numpy as np

log_dir = "logs/uav"
experiments = [
    "cola_full_s1", "cola_full_s2", "cola_full_s3",
    "no_cor_s1", "no_cor_s2", "no_cor_s3",
    "no_oadm_s1", "no_oadm_s2", "no_oadm_s3",
    "envelope_s1", "envelope_s2", "envelope_s3",
    "ws_fidelity_s1", "ws_balanced_s1", "ws_energy_s1",
]

pattern = re.compile(
    r"episode:\s+(\d+)\s+.*?rl reward:\s*\[([^\]]+)\]"
)

obj_names = ["SemFid", "Fresh", "Energy", "Fair"]

print(f"{'Experiment':<22} {'Episodes':>8} {'Steps':>8} | "
      + " ".join(f"{n:>8}" for n in obj_names) +
      f" | {'Sum':>8}  (last 20 ep avg)")
print("-" * 110)

for exp in experiments:
    path = os.path.join(log_dir, f"{exp}.log")
    if not os.path.exists(path):
        print(f"{exp:<22} (no log)")
        continue

    episodes = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep = int(m.group(1))
                vals = [float(x) for x in m.group(2).split()]
                if len(vals) >= 3:
                    episodes.append((ep, vals))

    if not episodes:
        print(f"{exp:<22} (no episodes)")
        continue

    last_ep = episodes[-1][0]
    steps_est = last_ep * 200
    n_obj = len(episodes[-1][1])

    last_n = episodes[-20:] if len(episodes) >= 20 else episodes
    rewards = np.array([e[1] for e in last_n])
    means = rewards.mean(axis=0)

    vals_str = " ".join(f"{means[i]:>8.1f}" if i < n_obj else f"{'N/A':>8}" for i in range(4))
    print(f"{exp:<22} {last_ep:>8} {steps_est:>8} | {vals_str} | {means.sum():>8.1f}")
