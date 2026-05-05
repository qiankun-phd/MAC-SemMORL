#!/usr/bin/env python3
import re, numpy as np, sys

logs = {
    'COLA_Full': '/home/qiankun/CommRL/COLA/logs/uav/cola_full_seed1.log',
    'No_COR':    '/home/qiankun/CommRL/COLA/logs/uav/no_cor_seed1.log',
    'No_OADM':   '/home/qiankun/CommRL/COLA/logs/uav/no_oadm_seed1.log',
}

out = []
out.append("=" * 72)
out.append("UAV-SemCom COLA Experiment Results (seed=1, 200K steps)")
out.append("=" * 72)

for name, path in logs.items():
    rewards = []
    with open(path) as f:
        for line in f:
            m = re.search(r'rl reward: \[([\d.\s\-e]+)\]', line)
            if m:
                vals = [float(x) for x in m.group(1).split()]
                if len(vals) == 3:
                    rewards.append(vals)
    
    arr = np.array(rewards)
    last50 = arr[-50:]
    
    out.append(f"\n--- {name} (total {len(rewards)} episodes) ---")
    out.append(f"  Objectives:       Sem.Fidelity   Info.Fresh.   Energy.Eff.")
    out.append(f"  Last 50 ep mean:  {last50[:,0].mean():>10.1f}   {last50[:,1].mean():>10.1f}   {last50[:,2].mean():>10.1f}")
    out.append(f"  Last 50 ep std:   {last50[:,0].std():>10.1f}   {last50[:,1].std():>10.1f}   {last50[:,2].std():>10.1f}")
    out.append(f"  Sum of means:     {last50.mean(axis=0).sum():.1f}")

out.append("\n" + "=" * 72)

result = "\n".join(out)
with open("/home/qiankun/CommRL/COLA/analysis_output.txt", "w") as f:
    f.write(result)
print("DONE", file=sys.stderr)
