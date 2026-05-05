"""Diagnose why fidelity is too easy - check actual SNR values."""
import numpy as np
import gym

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from environments import *

env = gym.make("UAV-SemCom-v0", num_devices=5, max_episode_steps=200)
env.seed(42)
env.reset()

print("=" * 60)
print("CHANNEL & SNR DIAGNOSIS")
print("=" * 60)

# Noise power
print(f"\nNoise PSD: {-174} dBm/Hz = {-174-30} dBW/Hz")
print(f"BW per device: {env.bw_per_device/1e6:.1f} MHz")
print(f"Noise power: {env.noise_power:.2e} W ({10*np.log10(env.noise_power):.1f} dBW)")
print(f"Max power/dev: {env.max_power} W ({10*np.log10(env.max_power):.1f} dBW)")

# Channel gains at typical distances
print(f"\nTypical distances and channel gains:")
for dist in [50, 100, 200, 300, 400]:
    d3d = np.sqrt(dist**2 + env.uav_height**2)
    theta = np.degrees(np.arctan2(env.uav_height, max(dist, 1.0)))
    p_los = 1.0 / (1.0 + env.a2g_a * np.exp(-env.a2g_b * (theta - env.a2g_a)))
    wl = env.c / env.carrier_freq
    fspl = 20 * np.log10(4 * np.pi * d3d / wl)
    pl = p_los * (fspl + env.eta_los_db) + (1 - p_los) * (fspl + env.eta_nlos_db)
    g_db = -pl
    g_lin = 10**(g_db/10)

    snr_max = env.max_power * g_lin / env.noise_power
    snr_db = 10 * np.log10(snr_max)
    snr_half = (env.max_power/2) * g_lin / env.noise_power
    snr_half_db = 10 * np.log10(snr_half)

    fids = []
    for k in range(env.num_devices):
        cfg = env.device_configs[k]
        x = cfg["sem_c"] * max(snr_db, 0) / 10.0 * (0.55 ** cfg["sem_b"])
        fids.append(float(np.clip(1.0 - np.exp(-x), 0.0, 1.0)))

    print(f"  d_h={dist:3d}m → PL={pl:.1f}dB  SNR(max_P)={snr_db:.1f}dB  "
          f"SNR(half_P)={snr_half_db:.1f}dB  fid@η=0.55: {[f'{f:.3f}' for f in fids]}")

# What SNR range gives meaningful fidelity variation?
print(f"\n{'='*60}")
print("FIDELITY vs SNR (device 0 = image, sem_c=0.6, sem_b=0.5, η=0.55)")
print("=" * 60)
cfg0 = env.device_configs[0]
for snr in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    x = cfg0["sem_c"] * max(snr, 0) / 10.0 * (0.55 ** cfg0["sem_b"])
    fid = 1.0 - np.exp(-x)
    bar = "█" * int(fid * 40)
    print(f"  SNR={snr:2d}dB → fid={fid:.3f} {bar}")

print(f"\n{'='*60}")
print("PROBLEM: Current SNR ≈ 45-50 dB → fidelity saturated near 1.0")
print("TARGET:  Need SNR ≈ 5-25 dB → fidelity in [0.2, 0.8] range")
print("FIX:     Increase effective noise density from -174 to ~-150 dBm/Hz")
print("         (accounts for receiver noise figure + urban interference)")
print("=" * 60)

# Simulate with fixed noise density
print(f"\n{'='*60}")
print("SIMULATION: Effect of noise density adjustment")
print("=" * 60)
for npd in [-174, -164, -154, -150, -144]:
    npd_dbw = npd - 30
    noise_w = 10 ** (npd_dbw / 10) * env.bw_per_device
    
    fid_at_200m = []
    dist = 200
    d3d = np.sqrt(dist**2 + env.uav_height**2)
    theta = np.degrees(np.arctan2(env.uav_height, max(dist, 1.0)))
    p_los = 1.0 / (1.0 + env.a2g_a * np.exp(-env.a2g_b * (theta - env.a2g_a)))
    wl = env.c / env.carrier_freq
    fspl = 20 * np.log10(4 * np.pi * d3d / wl)
    pl = p_los * (fspl + env.eta_los_db) + (1 - p_los) * (fspl + env.eta_nlos_db)
    g_lin = 10**((-pl)/10)
    
    for pf in [0.2, 0.5, 1.0]:
        pwr = env.max_power * pf
        snr_l = pwr * g_lin / noise_w
        snr_d = 10*np.log10(max(snr_l, 1e-10))
        
        fids_all = []
        for k in range(env.num_devices):
            cfg = env.device_configs[k]
            x = cfg["sem_c"] * max(snr_d, 0) / 10.0 * (0.55 ** cfg["sem_b"])
            fids_all.append(1.0 - np.exp(-x))
        
        avg_fid = np.average(fids_all, weights=[d["priority"] for d in env.device_configs[:5]])
        fid_at_200m.append((pf, snr_d, avg_fid))
    
    print(f"\n  NPD = {npd} dBm/Hz  (noise = {noise_w:.2e} W)")
    for pf, snr_d, af in fid_at_200m:
        above = "✓" if af >= 0.7 else "✗"
        print(f"    P={pf:.0%} → SNR={snr_d:5.1f}dB  avg_fid={af:.3f}  serve(≥0.7)? {above}")
