"""Find optimal noise PSD + serve threshold for balanced 3-objective rewards."""
import numpy as np
import gym
import sys
from itertools import product

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

sys.path.insert(0, "/home/qiankun/CommRL/COLA")
from environments.uav_semcom_env import UAVSemComEnv

from baselines import random_policy, fixed_trajectory_greedy_power, greedy_aosi

def evaluate_config(npd, threshold, aosi_inc):
    env = UAVSemComEnv(
        num_devices=5,
        max_episode_steps=200,
        noise_power_density_dbm_hz=npd,
        fidelity_serve_threshold=threshold,
    )
    # Patch AoSI increase rate for testing
    orig_step = env.step.__func__

    def patched_step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.steps += 1
        accel = action[:2] * self.max_accel
        power = (action[2:2+self.num_devices] + 1.0) / 2.0 * self.max_power
        eta = (action[2+self.num_devices:] + 1.0) / 2.0 * 0.9 + 0.1
        self.uav_vel += accel * self.slot_duration
        speed = np.linalg.norm(self.uav_vel)
        if speed > self.max_speed:
            self.uav_vel *= self.max_speed / speed
        self.uav_pos += self.uav_vel * self.slot_duration
        for dim in range(2):
            if self.uav_pos[dim] < 0:
                self.uav_pos[dim] = 0; self.uav_vel[dim] = abs(self.uav_vel[dim]) * 0.5
            elif self.uav_pos[dim] > self.area_size:
                self.uav_pos[dim] = self.area_size; self.uav_vel[dim] = -abs(self.uav_vel[dim]) * 0.5
        gains_db = self._channel_gains_db()
        fidelities = np.zeros(self.num_devices)
        for k in range(self.num_devices):
            g_linear = 10**(gains_db[k]/10)
            snr_linear = power[k] * g_linear / self.noise_power
            snr_db = 10*np.log10(max(snr_linear, 1e-10))
            fidelities[k] = self._semantic_fidelity(snr_db, eta[k], k)
            if fidelities[k] >= self.serve_threshold:
                self.aosi[k] = max(1.0, self.aosi[k] * 0.3)
            else:
                self.aosi[k] += aosi_inc
        spd = np.linalg.norm(self.uav_vel)
        e_prop = (self.hover_power + self.move_power_coeff * spd**2) * self.slot_duration
        e_comm = float(np.sum(power)) * self.slot_duration
        e_total = e_prop + e_comm
        r_fidelity = float(np.sum(self.priorities * fidelities) / np.sum(self.priorities)) * 4.0 + 0.5
        mean_aosi = float(np.mean(self.aosi))
        r_freshness = 4.0 * np.exp(-mean_aosi / 4.0) + 0.5
        r_energy = ((self.max_energy - e_total) / max(self.max_energy - self.min_energy, 1e-6) * 4.0 + 0.5)
        r_energy = max(0.5, r_energy)
        reward = np.array([r_fidelity, r_freshness, r_energy], dtype=np.float32)
        done = self.steps >= self.max_episode_steps
        return self._get_obs(), reward, done, {}

    import types
    env.step = types.MethodType(patched_step, env)

    results = {}
    for pol_name, pol_fn in [("Random", random_policy), ("Greedy_AoSI", greedy_aosi)]:
        all_steps = []
        for ep in range(10):
            env.seed(42 + ep)
            obs = env.reset()
            done = False
            while not done:
                action = pol_fn(obs, env)
                obs, reward, done, _ = env.step(action)
                all_steps.append(reward)
        steps = np.array(all_steps)
        results[pol_name] = {
            "mean": steps.mean(axis=0),
            "ep_sum": steps.reshape(10, 200, 3).sum(axis=1).mean(axis=0),
        }
    return results


print(f"{'NPD':>5} {'Thresh':>6} {'AoSI+':>5} | {'Rnd_Fid':>7} {'Rnd_Fre':>7} {'Rnd_Eng':>7} | {'Grd_Fid':>7} {'Grd_Fre':>7} {'Grd_Eng':>7} | {'Diff_F':>6} {'Diff_R':>6} {'Ratio':>6} {'SCORE':>6}")
print("-" * 120)

best_score = -1
best_cfg = None

for npd in [-164, -160, -158, -156, -154]:
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
        for aosi_inc in [1.0, 1.5, 2.0]:
            res = evaluate_config(npd, threshold, aosi_inc)
            rnd = res["Random"]["mean"]
            grd = res["Greedy_AoSI"]["mean"]

            diff_fid = grd[0] - rnd[0]
            diff_fre = grd[1] - rnd[1]
            diff_eng = grd[2] - rnd[2]

            ep_rnd = res["Random"]["ep_sum"]
            ep_grd = res["Greedy_AoSI"]["ep_sum"]

            max_ep = max(ep_grd.max(), ep_rnd.max())
            ratios = np.minimum(ep_rnd, ep_grd) / max(max_ep, 1)
            min_ratio = ratios.min()

            diffs = np.array([diff_fid, diff_fre, abs(diff_eng)])
            balanced = diffs.min() / max(diffs.max(), 1e-6)

            score = (min_ratio * 0.4 +                 # all objectives sizable
                     balanced * 0.3 +                    # differentiation balanced
                     min(diff_fid, 1.5)/1.5 * 0.15 +   # fidelity varies
                     min(diff_fre, 1.5)/1.5 * 0.15)    # freshness varies

            if score > best_score:
                best_score = score
                best_cfg = (npd, threshold, aosi_inc)

            print(f"{npd:>5} {threshold:>6.2f} {aosi_inc:>5.1f} | "
                  f"{rnd[0]:>7.3f} {rnd[1]:>7.3f} {rnd[2]:>7.3f} | "
                  f"{grd[0]:>7.3f} {grd[1]:>7.3f} {grd[2]:>7.3f} | "
                  f"{diff_fid:>6.3f} {diff_fre:>6.3f} {min_ratio:>6.3f} {score:>6.3f}")

print(f"\n{'='*60}")
print(f"BEST CONFIG: NPD={best_cfg[0]} dBm/Hz, threshold={best_cfg[1]}, aosi_inc={best_cfg[2]}")
print(f"SCORE: {best_score:.3f}")
print(f"{'='*60}")
