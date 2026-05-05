"""Multi-agent SemMORL wrapper for the journal extension Phase 1.

Implements the per-UAV federated CTDE path locked in ADR-0005 (`docs/DECISIONS.md`).

Phase 1.1 pilot (this file): joint-action CTDE with shared trunk —
SacAgent's policy outputs the full joint action `M*(2+3K)`; the env unflattens
it into per-UAV chunks. This is functionally equivalent to "M actors with
shared trunk" and reuses the existing SacAgent training loop, replay buffer,
COR loss, and OADM encoder unchanged. Per-UAV partial-observability slicing
helpers are provided here so Phase 1.2 (true per-UAV actor heads) can build
on top without re-deriving the indexing arithmetic.

Phase 1.2 hook (NOT YET IMPLEMENTED): split the policy network into M heads
that operate on per-UAV observation slices, enable optional FedAvg encoder
sync (`sync_encoders` placeholder below).
"""

from __future__ import annotations

import numpy as np

from agent import SacAgent


class MultiAgentSemMORL(SacAgent):
    """SacAgent variant with multi-UAV bookkeeping.

    Phase 1.1 behaviour: identical to SacAgent on the joint state/action
    presented by `MultiUAVSemComEnv`. Adds:

    - `num_uavs`, `num_devices` attributes for logging / ablation knobs.
    - `extract_local_obs(joint_obs, m)` — slice for partial-observability
      experiments (Phase 1.2).
    - `split_action(joint_action)` — list of M per-UAV action chunks.
    - `sync_encoders()` — FedAvg placeholder for federated-latent fallback
      (DESIGN §2.2). Currently a no-op since Phase 1.1 uses one shared
      encoder.
    """

    def __init__(self, env_id, env, log_dir, *, num_uavs: int, num_devices: int, **kwargs):
        if num_uavs < 2:
            raise ValueError("MultiAgentSemMORL requires num_uavs >= 2; for single UAV use SacAgent.")
        if env.action_space.shape[0] != num_uavs * (2 + 3 * num_devices):
            raise ValueError(
                f"env action_space mismatch: got {env.action_space.shape[0]}, "
                f"expected M*(2+3K) = {num_uavs}*(2+3*{num_devices}) "
                f"= {num_uavs * (2 + 3 * num_devices)}"
            )

        super().__init__(env_id=env_id, env=env, log_dir=log_dir, **kwargs)

        self.num_uavs = num_uavs
        self.num_devices = num_devices
        self.per_uav_action_dim = 2 + 3 * num_devices
        # Per-UAV obs dim under the partial-observability split:
        #   4 (own pos+vel) + 2K (rel_pos) + K (gain) + K (aosi) + K (type) + 1 (time)
        self.per_uav_obs_dim = 5 + 5 * num_devices

    # ------------------------------------------------------------------
    # Per-UAV slicing helpers (used in Phase 1.2; safe no-ops in Phase 1.1)
    # ------------------------------------------------------------------
    def extract_local_obs(self, joint_obs: np.ndarray, m: int) -> np.ndarray:
        """Return UAV m's local observation slice from a joint observation.

        Layout produced by MultiUAVSemComEnv._get_obs:
            [pos(2M), vel(2M), rel_pos(2MK), gains(MK), aosi(K), types(K), t(1)]
        """
        M, K = self.num_uavs, self.num_devices
        pos = joint_obs[2 * m : 2 * m + 2]
        vel = joint_obs[2 * M + 2 * m : 2 * M + 2 * m + 2]
        rel_off = 4 * M + 2 * K * m
        rel = joint_obs[rel_off : rel_off + 2 * K]
        gain_off = 4 * M + 2 * M * K + K * m
        gain = joint_obs[gain_off : gain_off + K]
        aosi_off = 4 * M + 2 * M * K + M * K
        aosi = joint_obs[aosi_off : aosi_off + K]
        type_off = aosi_off + K
        types = joint_obs[type_off : type_off + K]
        t_off = type_off + K
        t = joint_obs[t_off : t_off + 1]
        return np.concatenate([pos, vel, rel, gain, aosi, types, t]).astype(np.float32)

    def split_action(self, joint_action: np.ndarray) -> list[np.ndarray]:
        """Return list of M per-UAV action chunks from a flat joint action."""
        per = self.per_uav_action_dim
        return [joint_action[m * per : (m + 1) * per] for m in range(self.num_uavs)]

    # ------------------------------------------------------------------
    # Phase 1.2 hook (federated encoder sync) — currently a no-op
    # ------------------------------------------------------------------
    def sync_encoders(self) -> None:
        """FedAvg sync of per-UAV encoders. No-op in Phase 1.1 (shared encoder).

        Phase 1.2 implementation will: average φ_m across all UAVs, broadcast.
        """
        pass
