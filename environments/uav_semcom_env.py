"""
UAV Semantic Communication Multi-Objective Environment

System: Single UAV serving K heterogeneous IoT devices via semantic communication.
The UAV acts as an aerial semantic relay with edge encoding capability.

Objectives (4, all positive, higher is better):
  1. Semantic Fidelity   — weighted average semantic similarity across devices
  2. Information Freshness — inverse of average Age of Semantic Information (AoSI)
  3. Energy Efficiency    — normalized energy saving relative to worst-case
  4. Coverage Fairness    — Jain's fairness index on per-device fidelity

Channel: Air-to-Ground probabilistic LoS model (Al-Hourani et al.)
Semantic: SNR-and-compression dependent fidelity per data type
Energy:  Rotary-wing propulsion + communication power

IoT mobility (optional, default off): set ``device_mobility`` to ``"line"``
(constant slow speed, straight line, reflect at area margin) or ``"drift"``
(uncorrelated Gaussian step per slot, clipped to margin). Use ``device_speed``
in m/s (e.g. 0.3--1.0; UAV max speed is 20 m/s).
"""

import numpy as np
import gym
from gym import spaces


# ---------------------------------------------------------------------------
# Device type configurations
#   sem_c, sem_b : parameters in  S = 1 - exp(-c * (SNR_dB/10) * eta^b)
#   priority     : weight when computing fidelity reward
# ---------------------------------------------------------------------------
DEFAULT_DEVICE_CONFIGS = [
    {"name": "image",   "sem_c": 0.6, "sem_b": 0.5, "priority": 1.0},
    {"name": "text",    "sem_c": 1.0, "sem_b": 0.7, "priority": 0.8},
    {"name": "sensor",  "sem_c": 1.5, "sem_b": 0.9, "priority": 1.2},
    {"name": "video",   "sem_c": 0.4, "sem_b": 0.4, "priority": 0.9},
    {"name": "control", "sem_c": 2.0, "sem_b": 1.0, "priority": 1.5},
]


class UAVSemComEnv(gym.Env):
    """
    State  (30-dim):
        UAV pos (2) | UAV vel (2) | relative device pos (K*2)
        | device AoSI (K) | channel gains (K) | device types (K) | time (1)

    Action (12-dim):
        UAV acceleration (2) | transmit power per device (K) | compression ratio (K)
        All outputs in [-1, 1]; remapped inside step().
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_devices: int = 5,
        area_size: float = 500.0,
        uav_height: float = 100.0,
        max_speed: float = 20.0,
        max_accel: float = 5.0,
        max_power_per_device: float = 0.1,       # Watts
        total_bandwidth: float = 5e6,             # Hz
        carrier_freq: float = 2e9,                # Hz
        noise_power_density_dbm_hz: float = -158, # dBm/Hz (thermal -174 + 6dB NF + 10dB urban interference)
        slot_duration: float = 1.0,               # seconds
        max_episode_steps: int = 200,
        fidelity_serve_threshold: float = 0.45,
        device_configs=None,
        device_mobility: str = "none",
        device_speed: float = 0.0,
    ):
        super().__init__()

        self.num_devices = num_devices
        self.area_size = area_size
        self.uav_height = uav_height
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_power = max_power_per_device
        self.total_bandwidth = total_bandwidth
        self.bw_per_device = total_bandwidth / num_devices
        self.carrier_freq = carrier_freq
        self.slot_duration = slot_duration
        self.serve_threshold = fidelity_serve_threshold

        if device_mobility not in ("none", "line", "drift"):
            raise ValueError(
                "device_mobility must be 'none', 'line', or 'drift'"
            )
        self.device_mobility = device_mobility
        if device_mobility != "none" and device_speed <= 0:
            device_speed = 0.5
        self.device_speed = float(device_speed)
        self._device_margin = self.area_size * 0.1
        self._devices_move_line = (
            self.device_mobility == "line" and self.device_speed > 0
        )
        self._devices_move_drift = (
            self.device_mobility == "drift" and self.device_speed > 0
        )

        # Noise power per sub-band (Watts)
        npd_dbw = noise_power_density_dbm_hz - 30          # dBm→dBW /Hz
        self.noise_power = 10 ** (npd_dbw / 10) * self.bw_per_device

        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps

        # ---------- multi-objective ----------
        self.reward_num = 4
        self.obj_dim = 4

        # ---------- UAV propulsion (simplified rotary-wing) ----------
        self.hover_power = 100.0   # W
        self.move_power_coeff = 3.0  # W / (m/s)^2

        # ---------- A2G channel (urban) ----------
        self.a2g_a = 9.61
        self.a2g_b = 0.16
        self.eta_los_db = 1.0
        self.eta_nlos_db = 20.0
        self.c = 3e8

        # ---------- device configs ----------
        if device_configs is None:
            device_configs = DEFAULT_DEVICE_CONFIGS
        assert len(device_configs) >= num_devices
        self.device_configs = device_configs[:num_devices]
        self.priorities = np.array([d["priority"] for d in self.device_configs])

        # ---------- spaces ----------
        self.state_dim = (
            2                   # UAV pos
            + 2                 # UAV vel
            + num_devices * 2   # relative pos
            + num_devices       # AoSI
            + num_devices       # channel gain
            + num_devices       # device type
            + 1                 # time
        )
        self.action_dim = 2 + num_devices + num_devices

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32,
        )

        # ---------- reward normalisation helpers ----------
        self.max_energy = (
            self.hover_power
            + self.move_power_coeff * max_speed ** 2
            + max_power_per_device * num_devices
        ) * slot_duration
        self.min_energy = self.hover_power * slot_duration

        # ---------- runtime state ----------
        self.steps = 0
        self.uav_pos = None
        self.uav_vel = None
        self.device_positions = None
        self.device_velocities = None
        self.aosi = None
        self.device_type_ids = None
        self.np_random = np.random.RandomState()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        self.uav_pos = np.array(
            [self.area_size / 2, self.area_size / 2], dtype=np.float64
        )
        self.uav_vel = np.zeros(2, dtype=np.float64)

        m = self._device_margin
        self.device_positions = self.np_random.uniform(
            m, self.area_size - m, size=(self.num_devices, 2)
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
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.steps += 1

        # ---- decode actions ----
        accel = action[:2] * self.max_accel
        power = (action[2 : 2 + self.num_devices] + 1.0) / 2.0 * self.max_power
        eta   = (action[2 + self.num_devices :] + 1.0) / 2.0 * 0.9 + 0.1  # [0.1, 1.0]

        # ---- UAV kinematics ----
        self.uav_vel += accel * self.slot_duration
        speed = np.linalg.norm(self.uav_vel)
        if speed > self.max_speed:
            self.uav_vel *= self.max_speed / speed
        self.uav_pos += self.uav_vel * self.slot_duration
        # boundary reflection
        for dim in range(2):
            if self.uav_pos[dim] < 0:
                self.uav_pos[dim] = 0
                self.uav_vel[dim] = abs(self.uav_vel[dim]) * 0.5
            elif self.uav_pos[dim] > self.area_size:
                self.uav_pos[dim] = self.area_size
                self.uav_vel[dim] = -abs(self.uav_vel[dim]) * 0.5

        # ---- slow IoT mobility (optional; default off) ----
        if self._devices_move_line:
            lo, hi = self._device_margin, self.area_size - self._device_margin
            dt = self.slot_duration
            self.device_positions += self.device_velocities * dt
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
            step = (
                self.np_random.randn(self.num_devices, 2).astype(np.float64)
                * (0.5 * self.device_speed * self.slot_duration)
            )
            self.device_positions += step
            np.clip(self.device_positions, lo, hi, out=self.device_positions)

        # ---- channel & semantic ----
        gains_db = self._channel_gains_db()
        fidelities = np.zeros(self.num_devices)
        for k in range(self.num_devices):
            g_linear = 10 ** (gains_db[k] / 10)
            snr_linear = power[k] * g_linear / self.noise_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-10))
            fidelities[k] = self._semantic_fidelity(snr_db, eta[k], k)

            # AoSI update: served only when fidelity exceeds threshold
            if fidelities[k] >= self.serve_threshold:
                self.aosi[k] = max(1.0, self.aosi[k] * 0.3)
            else:
                self.aosi[k] += 1.0

        # ---- energy ----
        spd = np.linalg.norm(self.uav_vel)
        e_prop = (self.hover_power + self.move_power_coeff * spd ** 2) * self.slot_duration
        e_comm = float(np.sum(power)) * self.slot_duration
        e_total = e_prop + e_comm

        # ---- raw communication metrics ----
        K = self.num_devices
        weighted_avg_fid = float(
            np.sum(self.priorities * fidelities) / np.sum(self.priorities)
        )
        mean_aosi = float(np.mean(self.aosi))
        fid_sum_sq = float(np.sum(fidelities ** 2))
        fid_sum = float(np.sum(fidelities))
        jain_idx = (fid_sum ** 2) / max(K * fid_sum_sq, 1e-10)
        service_rate = float(np.sum(fidelities >= self.serve_threshold)) / K

        # ---- rewards (all in ~[0.5, 4.5], higher = better) ----
        r_fidelity = weighted_avg_fid * 4.0 + 0.5

        r_freshness = 4.0 * np.exp(-mean_aosi / 4.0) + 0.5

        r_energy = (
            (self.max_energy - e_total)
            / max(self.max_energy - self.min_energy, 1e-6)
            * 4.0
            + 0.5
        )
        r_energy = max(0.5, r_energy)

        # Jain fairness: mapped from [1/K, 1.0] to [0.5, 4.5]
        r_fairness = (jain_idx - 1.0 / K) / max(1.0 - 1.0 / K, 1e-10) * 4.0 + 0.5

        reward = np.array(
            [r_fidelity, r_freshness, r_energy, r_fairness], dtype=np.float32
        )
        done = self.steps >= self.max_episode_steps
        info = {
            "obj": reward,
            "fidelities": fidelities.copy(),
            "aosi": self.aosi.copy(),
            "energy": e_total,
            "uav_pos": self.uav_pos.copy(),
            "jain_fairness": jain_idx,
            "service_rate": service_rate,
            "weighted_avg_fidelity": weighted_avg_fid,
            "mean_aosi": mean_aosi,
            "max_aosi": float(np.max(self.aosi)),
        }
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self):
        pos_norm = self.uav_pos / self.area_size
        vel_norm = self.uav_vel / self.max_speed
        rel_pos  = ((self.device_positions - self.uav_pos) / self.area_size).flatten()
        aosi_norm = np.clip(self.aosi / 50.0, 0, 1)
        gains_norm = (self._channel_gains_db() + 130) / 60.0
        t_remain = np.array([1.0 - self.steps / self.max_episode_steps])

        obs = np.concatenate([
            pos_norm, vel_norm, rel_pos,
            aosi_norm, gains_norm,
            self.device_type_ids,
            t_remain,
        ]).astype(np.float32)
        return obs

    def _channel_gains_db(self):
        """A2G probabilistic LoS channel model."""
        gains = np.zeros(self.num_devices)
        wavelength = self.c / self.carrier_freq
        for k in range(self.num_devices):
            dx = self.uav_pos - self.device_positions[k]
            d_h = np.linalg.norm(dx)
            d_3d = np.sqrt(d_h ** 2 + self.uav_height ** 2)
            theta_deg = np.degrees(np.arctan2(self.uav_height, max(d_h, 1.0)))

            p_los = 1.0 / (1.0 + self.a2g_a * np.exp(
                -self.a2g_b * (theta_deg - self.a2g_a)
            ))
            fspl = 20.0 * np.log10(4.0 * np.pi * d_3d / wavelength)
            pl = p_los * (fspl + self.eta_los_db) + (1 - p_los) * (fspl + self.eta_nlos_db)

            # add Rician/Rayleigh small-scale fading (log-normal approx, σ=2 dB)
            fading = self.np_random.normal(0, 2.0)
            gains[k] = -pl + fading
        return gains

    def _semantic_fidelity(self, snr_db, eta, dev_idx):
        """SNR-and-compression dependent semantic fidelity."""
        cfg = self.device_configs[dev_idx]
        x = cfg["sem_c"] * max(snr_db, 0) / 10.0 * (eta ** cfg["sem_b"])
        return float(np.clip(1.0 - np.exp(-x), 0.0, 1.0))
