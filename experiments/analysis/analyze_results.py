import re, numpy as np

logs = {
    'COLA Full': 'logs/uav/cola_full_seed1.log',
    'No COR':    'logs/uav/no_cor_seed1.log',
    'No OADM':   'logs/uav/no_oadm_seed1.log',
}

print("=" * 70)
print("UAV-SemCom Experiment Results (seed 1)")
print("=" * 70)

for name, path in logs.items():
    rewards = []
    with open(path) as f:
        for line in f:
            m = re.search(r'rl reward: \[([\d.\s\-e]+)\]', line)
            if m:
                vals = [float(x) for x in m.group(1).split()]
                rewards.append(vals)
    
    all_r = np.array(rewards)
    last200 = all_r[-200:] if len(all_r) >= 200 else all_r
    last50 = all_r[-50:] if len(all_r) >= 50 else all_r
    
    print(f"\n--- {name} ({len(rewards)} episodes, step≈{len(rewards)*200}) ---")
    print(f"  Obj names:        Sem.Fidelity   Info.Fresh.    Energy Eff.")
    print(f"  Last 200 avg:     {last200[:,0].mean():>10.1f}   {last200[:,1].mean():>10.1f}   {last200[:,2].mean():>10.1f}")
    print(f"  Last 200 std:     {last200[:,0].std():>10.1f}   {last200[:,1].std():>10.1f}   {last200[:,2].std():>10.1f}")
    print(f"  Last  50 avg:     {last50[:,0].mean():>10.1f}   {last50[:,1].mean():>10.1f}   {last50[:,2].mean():>10.1f}")
    print(f"  Last  50 std:     {last50[:,0].std():>10.1f}   {last50[:,1].std():>10.1f}   {last50[:,2].std():>10.1f}")
    print(f"  Sum(avg last200): {last200.mean(axis=0).sum():.1f}")

print("\n" + "=" * 70)
