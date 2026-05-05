"""
Multi-UAV Semantic Communication Multi-Objective Environment.

Extension of UAVSemComEnv (single UAV) to M cooperatively-serving UAVs.
Implements the formulation specified in `docs/DESIGN-multi-uav.md` §1 and the
per-UAV federated CTDE path locked in ADR-0005 (`docs/DECISIONS.md`).

State (flat joint, dim = 4M + 2MK + MK + 2K + 1):
    M*4         per-UAV [pos_x, pos_y, vel_x, vel_y] (normalised)
    M*K*2       relative position UAV m -> device k (normalised)
    M*K         channel gain UAV m -> device k (normalised)
    K           per-device AoSI (normalised)
    K           per-device traffic-type id (normalised)
    1           remaining episode time

Action (flat joint, dim = M*(2 + 3K)):
    For each UAV m, action chunk of size (2 + 3K):
        2     acceleration (a_x, a_y)
        K     transmit power per device
        K     compression ratio per device
        K     scheduling logits — softmax across UAVs gives per-device assignment

Joint reward (4-dim, same components as single-UAV, aggregated):
    r_1 fidelity   weighted-priority sum over (m, k) pairs with x_{m,k} = 1
    r_2 freshness  inverse of mean AoSI; AoSI updates if ANY UAV serves
    r_3 energy     sum of per-UAV propulsion + comm energy
    r_4 fairness   Jain index over per-device fidelity

Constraints implemented:
    - No-double-service: per-device softmax across UAVs in the action decode.
    - Collision avoidance: soft penalty added to r_3 if ||q_m - q_{m'}|| < d_min.
    - Inter-UAV interference: optional flag, default off (matches DESIGN §3.3).
"""

from __future__ import annotations

import numpy as np
import gym
from gym import spaces

from environments.uav_semcom_env import DEFAULT_DEVICE_CONFIGS


class MultiUAVSemComEnv(gym.Env):
    """Multi-UAV variant of `UAVSemComEnv`.

    Backward-compat: `UAVSemComEnv` is unchanged. Use this class via gym id
    `UAV-SemCom-Multi-v0`. Requires `num_uavs >= 2`.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_uavs: int = 2,
        num_devices: int = 5,
        area_size: float = 500.0,
        uav_height: float = 100.0,
        max_speed: float = 20.0,
        max_accel: float = 5.0,
        max_power_per_device: float = 0.1,
        total_bandwidth: float = 5e6,
        carrier_freq: float = 2e9,
        noise_power_density_dbm_hz: float = -158,
        slot_duration: float = 1.0,
        max_episode_steps: int = 200,
        fidelity_serve_threshold: float = 0.45,
        device_configs=None,
        device_mobility: str = "none",
        device_speed: float = 0.0,
        d_min: float = 50.0,
        collision_penalty_weight: float = 1.0,
        use_interference: bool = False,
    ):
        super().__init__()

        if num_uavs < 2:
            raise ValueError("MultiUAVSemComEnv requires num_uavs >= 2; for single UAV use UAVSemComEnv.")

        self.num_uavs = num_uavs
        self.num_devices = num_devices
        self.area_size = area_size
        self.uav_height = uav_height
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_power = max_power_per_device
        self.total_bandwidth = total_bandwidth
        # bandwidth split equally across (UAV, device) sub-channels
        self.bw_per_subchannel = total_bandwidth / (num_uavs * num_devices)
        self.carrier_freq = carrier_freq
        self.slot_duration = slot_duration
        self.serve_threshold = fidelity_serve_threshold
        self.d_min = d_min
        self.w_c = collision_penalty_weight
        self.use_interference = use_interference

        if device_mobility not in ("none", "line", "drift"):
            raise ValueError("device_mobility must be 'none', 'line', or 'drift'")
        self.device_mobility = device_mobility
        if device_mobility != "none" and device_speed <= 0:
            device_speed = 0.5
        self.device_speed = float(device_speed)
        self._device_margin = self.area_size * 0.1
        self._devices_move_line = (device_mobility == "line" and device_speed > 0)
        self._devices_move_drift = (device_mobility == "drift" and device_speed > 0)

        npd_dbw = noise_power_density_dbm_hz - 30
        self.noise_power = 10 ** (npd_dbw / 10) * self.bw_per_subchannel

        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps

        # Multi-objective: 4 objectives same as single-UAV
        self.reward_num = 4
        self.obj_dim = 4

        # Propulsion (per UAV)
        self.hover_power = 100.0
        self.move_power_coeff = 3.0

        # A2G channel
        self.a2g_a = 9.61
        self.a2g_b = 0.16
        self.eta_los_db = 1.0
        self.eta_nlos_db = 20.0
        self.c = 3e8

        if device_configs is None:
            device_configs = DEFAULT_DEVICE_CONFIGS
        assert len(device_configs) >= num_devices
        self.device_configs = device_configs[:num_devices]
        self.priorities = np.array([d["priority"] for d in self.device_configs])

        # ---------- spaces ----------
        # State dim: 4M + 2MK + MK + 2K + 1
        self.state_dim = (
            4 * num_uavs                    # per-UAV pos (2) + vel (2)
            + 2 * num_uavs * num_devices    # rel pos UAV m -> device k
            + num_uavs * num_devices        # channel gain (m, k)
            + num_devices                   # AoSI per device
            + num_devices                   # device type id per device
            + 1                             # time remaining
        )
        # Per-UAV action: 2 (accel) + K (power) + K (eta) + K (schedule logits)
        self.per_uav_action_dim = 2 + 3 * num_devices
        self.action_dim = num_uavs * self.per_uav_action_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32,
        )

        # Reward normalisation helpers (per-UAV upper bound, summed across M)
        self.max_energy = (
            (self.hover_power + self.move_power_coeff * max_speed ** 2)
            + max_power_per_device * num_devices
        ) * slot_duration * num_uavs
        self.min_energy = self.hover_power * slot_duration * num_uavs

        # Runtime state
        self.steps = 0
        self.uav_pos = None        # shape (M, 2)
        self.uav_vel = None        # shape (M, 2)
        self.device_positions = None  # shape (K, 2)
        self.device_velocities = None
        self.aosi = None              # shape (K,)
        self.device_type_ids = None   # shape (K,)
        self.np_random = np.random.RandomState()

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        # Spread M UAVs along a line at the centre row, separated by d_min*1.2.
        spacing = max(self.d_min * 1.2, self.area_size / (self.num_uavs + 1))
        centre_y = self.area_size / 2
        start_x = (self.area_size - spacing * (self.num_uavs - 1)) / 2
        self.uav_pos = np.stack([
            np.array([start_x + spacing * m, centre_y], dtype=np.float64)
            for m in range(self.num_uavs)
        ])
        self.uav_vel = np.zeros((self.num_uavs, 2), dtype=np.float64)

        m_marg = self._device_margin
        self.device_positions = self.np_random.uniform(
            m_marg, self.area_size - m_marg, size=(self.num_devices, 2)
        )
        self.device_velocities = np.zeros((self.num_devices, 2), dtype=np.float64)
        if self._devices_move_line:
            ang = self.np_random.uniform(0, 2 * np.pi, size=self.num_devices)
            sp = self.device_speed
            self.device_velocities[:, 0] = np.cos(ang) * sp
            self.device_velocities[:, 1] = np.sin(ang) * sp

        self.aosi = np.ones(self.num_devices, dtype=np.float64) * 10.0
        self.device_type_ids = np.arange(self.num_devices, dtype=np.float32) / self.num_devices

        return self._get_obs()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.steps += 1

        # Decode joint action into per-UAV chunks, then per-component slices.
        # action shape: (M * (2 + 3K),)
        per = self.per_uav_action_dim
        K = self.num_devices
        accel = np.zeros((self.num_uavs, 2))
        power = np.zeros((self.num_uavs, K))
        eta = np.zeros((self.num_uavs, K))
        sched_logits = np.zeros((self.num_uavs, K))
        for m in range(self.num_uavs):
            chunk = action[m * per : (m + 1) * per]
            accel[m] = chunk[0:2] * self.max_accel
            power[m] = (chunk[2 : 2 + K] + 1.0) / 2.0 * self.max_power
            eta[m] = (chunk[2 + K : 2 + 2 * K] + 1.0) / 2.0 * 0.9 + 0.1
            sched_logits[m] = chunk[2 + 2 * K : 2 + 3 * K]

        # Per-device softmax across UAVs => no-double-service assignment.
        # x_assign[m, k] is the soft probability that UAV m serves device k.
        # During training the soft probability is used; for hard binary scheduling
        # take argmax along axis=0 at evaluation time.
        x_assign = self._softmax(sched_logits, axis=0)  # (M, K)

        # Per-UAV kinematics
        self.uav_vel += accel * self.slot_duration
        speeds = np.linalg.norm(self.uav_vel, axis=1)
        for m in range(self.num_uavs):
            if speeds[m] > self.max_speed:
                self.uav_vel[m] *= self.max_speed / speeds[m]
        self.uav_pos += self.uav_vel * self.slot_duration
        # Boundary reflection per UAV
        for m in range(self.num_uavs):
            for dim in range(2):
                if self.uav_pos[m, dim] < 0:
                    self.uav_pos[m, dim] = 0
                    self.uav_vel[m, dim] = abs(self.uav_vel[m, dim]) * 0.5
                elif self.uav_pos[m, dim] > self.area_size:
                    self.uav_pos[m, dim] = self.area_size
                    self.uav_vel[m, dim] = -abs(self.uav_vel[m, dim]) * 0.5

        # Optional device mobility (same dynamics as single-UAV)
        if self._devices_move_line:
            lo, hi = self._device_margin, self.area_size - self._device_margin
            self.device_positions += self.device_velocities * self.slot_duration
            for k in range(self.num_devices):
                for dim in range(2):
                    if self.device_positions[k, dim] < lo:
                        self.device_positions[k, dim] = lo
                        self.device_velocities[k, dim] *= -1.0
                    elif self.device_positions[k, dim] > hi:
                        self.device_positions[k, dim] = hi
                        self.device_velocities[k, dim] *= -1.0
        elif self._devices_move_drift:
            lo, hi = self._device_margin, self.area_size - self._device_margin
            jitter = (
                self.np_random.randn(self.num_devices, 2).astype(np.float64)
                * (0.5 * self.device_speed * self.slot_duration)
            )
            self.device_positions += jitter
            np.clip(self.device_positions, lo, hi, out=self.device_positions)

        # Per-(m, k) channel gains (M, K) in dB
        gains_db = self._channel_gains_db()

        # Per-(m, k) fidelity in [0, 1] using single-UAV semantic model.
        # SNR computed per (m, k) accounting for assignment weight.
        fidelity_mk = np.zeros((self.num_uavs, K))
        for m in range(self.num_uavs):
            for k in range(K):
                if self.use_interference:
                    interf = sum(
                        power[m_p, k] * 10 ** (gains_db[m_p, k] / 10)
                        for m_p in range(self.num_uavs)
                        if m_p != m
                    )
                else:
                    interf = 0.0
                g_lin = 10 ** (gains_db[m, k] / 10)
                snr_lin = power[m, k] * g_lin / (self.noise_power + interf)
                snr_db = 10 * np.log10(max(snr_lin, 1e-10))
                fidelity_mk[m, k] = self._semantic_fidelity(snr_db, eta[m, k], k)

        # Effective per-device fidelity = sum over UAVs weighted by assignment.
        # Hard semantics: device k gets serviced by UAV argmax_m x_assign[m, k].
        # Soft training: weight fidelity by x_assign so gradients flow.
        fidelities = (x_assign * fidelity_mk).sum(axis=0)  # (K,)

        # AoSI update: served if effective fidelity exceeds threshold.
        for k in range(K):
            if fidelities[k] >= self.serve_threshold:
                self.aosi[k] = max(1.0, self.aosi[k] * 0.3)
            else:
                self.aosi[k] += 1.0

        # Per-UAV propulsion energy
        spds = np.linalg.norm(self.uav_vel, axis=1)
        e_prop = (self.hover_power + self.move_power_coeff * spds ** 2) * self.slot_duration
        # Communication energy weighted by assignment
        e_comm = (x_assign * power).sum() * self.slot_duration
        e_total = float(e_prop.sum() + e_comm)

        # Collision penalty on r_3
        coll_pen = 0.0
        for m1 in range(self.num_uavs):
            for m2 in range(m1 + 1, self.num_uavs):
                d = np.linalg.norm(self.uav_pos[m1] - self.uav_pos[m2])
                if d < self.d_min:
                    coll_pen += (self.d_min - d) ** 2
        coll_pen *= self.w_c

        # Reward components (4) — same scaling as single-UAV (~[0.5, 4.5])
        weighted_avg_fid = float(np.sum(self.priorities * fidelities) / np.sum(self.priorities))
        mean_aosi = float(np.mean(self.aosi))
        fid_sum_sq = float(np.sum(fidelities ** 2))
        fid_sum = float(np.sum(fidelities))
        jain_idx = (fid_sum ** 2) / max(K * fid_sum_sq, 1e-10)
        service_rate = float(np.sum(fidelities >= self.serve_threshold)) / K

        r_fidelity = weighted_avg_fid * 4.0 + 0.5
        r_freshness = 4.0 * np.exp(-mean_aosi / 4.0) + 0.5
        r_energy = (
            (self.max_energy - e_total - coll_pen)
            / max(self.max_energy - self.min_energy, 1e-6) * 4.0 + 0.5
        )
        r_energy = max(0.5, r_energy)
        r_fairness = (jain_idx - 1.0 / K) / max(1.0 - 1.0 / K, 1e-10) * 4.0 + 0.5

        reward = np.array(
            [r_fidelity, r_freshness, r_energy, r_fairness], dtype=np.float32
        )
        done = self.steps >= self.max_episode_steps
        info = {
            "obj": reward,
            "fidelities": fidelities.copy(),
            "fidelity_mk": fidelity_mk.copy(),
            "x_assign": x_assign.copy(),
            "aosi": self.aosi.copy(),
            "energy": e_total,
            "collision_penalty": coll_pen,
            "uav_pos": self.uav_pos.copy(),
            "jain_fairness": jain_idx,
            "service_rate": service_rate,
            "weighted_avg_fidelity": weighted_avg_fid,
            "mean_aosi": mean_aosi,
        }
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self):
        # All blocks in normalised range, then concat to flat (state_dim,).
        pos_norm = (self.uav_pos / self.area_size).flatten()         # 2M
        vel_norm = (self.uav_vel / self.max_speed).flatten()         # 2M
        rel_pos = np.zeros((self.num_uavs, self.num_devices, 2))
        for m in range(self.num_uavs):
            rel_pos[m] = (self.device_positions - self.uav_pos[m]) / self.area_size
        rel_pos_flat = rel_pos.flatten()                              # 2MK
        gains_db = self._channel_gains_db()                           # (M, K)
        gains_norm = ((gains_db + 130.0) / 60.0).flatten()            # MK
        aosi_norm = np.clip(self.aosi / 50.0, 0, 1)                   # K
        type_ids = self.device_type_ids                               # K
        t_remain = np.array([1.0 - self.steps / self.max_episode_steps])  # 1

        obs = np.concatenate([
            pos_norm, vel_norm, rel_pos_flat, gains_norm,
            aosi_norm, type_ids, t_remain,
        ]).astype(np.float32)
        return obs

    def _channel_gains_db(self):
        """Per-(m, k) A2G channel gain in dB. Returns (M, K) array."""
        gains = np.zeros((self.num_uavs, self.num_devices))
        wavelength = self.c / self.carrier_freq
        for m in range(self.num_uavs):
            for k in range(self.num_devices):
                dx = self.uav_pos[m] - self.device_positions[k]
                d_h = np.linalg.norm(dx)
                d_3d = np.sqrt(d_h ** 2 + self.uav_height ** 2)
                theta_deg = np.degrees(np.arctan2(self.uav_height, max(d_h, 1.0)))
                p_los = 1.0 / (1.0 + self.a2g_a * np.exp(
                    -self.a2g_b * (theta_deg - self.a2g_a)
                ))
                fspl = 20.0 * np.log10(4.0 * np.pi * d_3d / wavelength)
                pl = p_los * (fspl + self.eta_los_db) + (1 - p_los) * (fspl + self.eta_nlos_db)
                fading = self.np_random.normal(0, 2.0)
                gains[m, k] = -pl + fading
        return gains

    def _semantic_fidelity(self, snr_db, eta_v, dev_idx):
        cfg = self.device_configs[dev_idx]
        x = cfg["sem_c"] * max(snr_db, 0) / 10.0 * (eta_v ** cfg["sem_b"])
        return float(np.clip(1.0 - np.exp(-x), 0.0, 1.0))

    @staticmethod
    def _softmax(x, axis):
        x_shift = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x_shift)
        return e / np.sum(e, axis=axis, keepdims=True)
