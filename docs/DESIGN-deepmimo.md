# Design Document: DeepMIMO Channel Replay Infrastructure

**Status**: Draft — pending reviewer approval
**Date**: 2026-05-03
**Author**: @claude (assistant) for issue #2 (P0.3)
**Roadmap item**: P0.3 → unblocks P2.6 (real-channel experiment in Tier-1 C.4.5)

---

## 0. Purpose and Scope

This document records the survey result of DeepMIMO for the journal-extension real-channel experiment (Task C.4.5). It covers:

- License audit (academic-use permitted, attribution required).
- UAV-applicability of the existing DeepMIMO v4 scenarios (limited — most scenarios are ground-BS).
- Recommended replay-infrastructure shim that lets `uav_semcom_env.py` consume DeepMIMO channel samples in place of its current closed-form path-loss model.
- A fallback plan (3GPP TR 38.901 air-to-ground synthetic) if DeepMIMO turns out to be blocked or insufficient.

---

## 1. License Audit

### 1.1 Code License — Apache 2.0

The DeepMIMO v4 codebase at <https://github.com/DeepMIMO/DeepMIMO> is released under the **Apache License, Version 2.0** (see the `LICENSE` file in the repository root). Apache 2.0 permits:

- Commercial and academic use without restriction.
- Modification and redistribution (including in proprietary works).
- Sublicensing under different terms in derivative works.

Required: keep the Apache 2.0 license notice in any redistribution, mark substantial modifications, and do not use DeepMIMO trademarks without permission. **Action required for our project**: include `LICENSE-DeepMIMO-Apache-2.0` in our repo if we vendor any DeepMIMO source; otherwise no action needed for pure pip-install usage.

### 1.2 Scenario Dataset License

The scenario datasets distributed via `dm.download(...)` are stored separately (object storage hosted by the DeepMIMO project at deepmimo.net). Per the DeepMIMO project page and `README.md`, scenario datasets are **freely shareable** and the project encourages re-use. Citation of the DeepMIMO paper is requested when datasets are used in publications.

**Required citation** (per README):

```bibtex
@article{Alkhateeb2019,
  author  = {A. Alkhateeb},
  title   = {{DeepMIMO}: A Generic Deep Learning Dataset for Millimeter Wave and Massive {MIMO} Applications},
  journal = {Proc.\ Information Theory and Applications Workshop (ITA)},
  year    = {2019},
  pages   = {1--8}
}
```

If we use a specific scenario in a paper, we should also cite the scenario's reference paper (listed on each scenario's page at <https://deepmimo.net/scenarios>).

### 1.3 Outcome

**No licensing blockers** for our intended academic use. The Apache 2.0 + freely-shareable scenario terms are strictly more permissive than what we need.

---

## 2. UAV-Applicability of DeepMIMO v4 Scenarios

### 2.1 Current Scenario Coverage (as of 2026-05-03)

DeepMIMO v4 ships 100+ scenarios via the [Scenarios Database](https://deepmimo.net/scenarios). Frequencies range from sub-6 GHz (e.g., `asu_campus_3p5` @ 3.5 GHz) to mmWave (e.g., I3 @ 60 GHz). Environments include outdoor urban (asu_campus, city), indoor (I-series), and intersection scenarios.

**However**: All scenarios I surveyed have **transmitters at terrestrial base-station heights** (typically 6 m for street-level small cells, up to ~25 m for macro cells on rooftops). I did not find a DeepMIMO-published scenario with the transmitter at UAV altitude (50–200 m) and free-space-dominant air-to-ground geometry.

This matches DeepMIMO's primary target audience — terrestrial 5G/6G ML research.

### 2.2 Three Strategies for UAV Channel Replay

Given the gap, we have three viable strategies, ordered by recommended preference:

#### Strategy A — Custom UAV scenario via Sionna RT + `dm.convert`

DeepMIMO v4 ships a `dm.convert(...)` API that ingests output from Sionna RT (NVIDIA's GPU-accelerated ray tracer) and converts it to the DeepMIMO scenario format. We can:

1. Build a Sionna RT scene with our target environment (e.g., urban canyon, suburban open) and a UAV-trajectory mesh of TX positions at 50–150 m altitude.
2. Run Sionna RT to produce per-link channel impulse responses for ~10⁴–10⁵ UE positions.
3. Run `dm.convert(...)` to produce a `mac_semmorl_uav_<scene>` DeepMIMO scenario.
4. Distribute the scenario inside our repo (or via `dm.upload(...)` to the public DB if license-cleared).

**Pros**: full control over UAV altitudes, trajectories, and frequency. Reproducible. Reusable by other groups.
**Cons**: requires Sionna RT + GPU (Sionna runs on TensorFlow + CUDA). Estimated 1–2 weeks of engineering for the Sionna scene setup + one-time ray-trace compute (~hours per scene).

#### Strategy B — Use ground-BS scenario, post-hoc lift TX height in SNR formula

Pick an existing outdoor scenario (e.g., `asu_campus_3p5`), download channels, and **post-process** the BS-to-UE path-loss to add a free-space LoS term that corresponds to lifting the BS to UAV altitude. Effectively:

```
PL_uav(d_3D) = PL_ground(d_2D) + 20·log10(d_3D / d_2D)
```

This is mathematically dubious — DeepMIMO scenarios bake in scattering geometry that does not survive lifting the TX 100 m. Recommend only as a last-resort sanity check; do not use as the primary real-channel experiment.

#### Strategy C — Skip DeepMIMO; use 3GPP TR 38.901 air-to-ground synthetic

If Strategies A and B are blocked (Sionna RT setup fails, GPU unavailable, scene setup intractable), fall back to the **3GPP TR 38.901 air-to-ground** path loss model with the LoS-probability mapping we already use in `uav_semcom_env.py`. This is what the conference paper uses.

**Trade-off vs DeepMIMO**: TR 38.901 is calibrated to measurement campaigns but is not "real-trace" — it is a stochastic model. TWC reviewers may accept it (TR 38.901 is a standardized reference) but the novelty score for the real-channel experiment drops. Recommend Strategy A unless infrastructure is impossible.

### 2.3 Recommended Path

**Strategy A** (custom Sionna RT + `dm.convert`) for the journal version. Begin Sionna RT scene setup in Phase 0 (parallel with multi-UAV refactor) so the scenario is ready by Phase 2 P2.6.

If we hit blockers — defined as ≥ 2 weeks of Sionna setup with no working scene — fall back to Strategy C (TR 38.901 synthetic) and clearly label the experiment in the paper as "stochastic-model real channel" rather than "ray-traced real channel".

---

## 3. Replay Infrastructure Shim

### 3.1 File: `src/channel/deepmimo_replay.py` (placeholder for now)

Shim signature, to be implemented in Phase 1 alongside the multi-UAV environment refactor:

```python
class DeepMIMOReplay:
    """Replays a DeepMIMO scenario as a channel oracle for uav_semcom_env."""

    def __init__(
        self,
        scenario_name: str,                    # e.g., "mac_semmorl_uav_urban"
        uav_altitude_m: float = 100.0,
        carrier_freq_ghz: float = 3.5,
        rx_antenna_idx: int = 0,
    ) -> None:
        import deepmimo as dm
        self.dataset = dm.load(scenario_name)
        self._uav_alt = uav_altitude_m
        # Extract per-UE pathloss tensor: shape (n_ue,)
        self._pl_db = dm.utils.compute_pathloss(self.dataset, rx_antenna_idx)

    def query(self, uav_xyz: np.ndarray, ue_xyz: np.ndarray) -> float:
        """Return path loss in dB for the (UAV, UE) link.

        Looks up the nearest UE position in the precomputed grid and
        applies a small free-space correction for the UAV altitude offset.
        """
        ue_idx = self._nearest_ue(ue_xyz)
        pl_base = self._pl_db[ue_idx]
        d_corr = self._altitude_correction(uav_xyz)
        return pl_base + d_corr
```

`uav_semcom_env.step()` accepts an optional `channel_model: ChannelModel = TR38901()`. To enable replay:

```python
env = UAVSemcomEnv(channel_model=DeepMIMOReplay("mac_semmorl_uav_urban"))
```

Default remains the closed-form TR 38.901 model used in the conference paper, so all existing experiments stay reproducible.

### 3.2 Validation Episode

Before declaring P0.3 complete, we will run one sanity-check episode:

1. Single-UAV trajectory at altitude 100 m over a fixed device layout (K = 5).
2. Compute throughput per device per slot using the DeepMIMO replay shim.
3. Compute throughput per device per slot using the closed-form TR 38.901 model.
4. Verify the two throughput series correlate (Pearson ρ > 0.7) — i.e., the real channel is not wildly different in qualitative shape, just in magnitude/details.

If correlation falls below 0.5, the scenario is unsuitable and we either pick a different scenario or revisit the altitude correction.

---

## 4. Acceptance Criteria

This design is considered complete and P0.3 ready to close when:

- [ ] `docs/DESIGN-deepmimo.md` merged to main.
- [ ] Decision recorded in `docs/DECISIONS.md` (new ADR-0005) selecting Strategy A (Sionna RT + `dm.convert`) as the default, with Strategy C (TR 38.901) as the fallback.
- [ ] `src/channel/deepmimo_replay.py` placeholder file created in a follow-up PR (not part of P0.3 scope; lives in P2.6).
- [ ] Reviewer (advisor / collaborator) approves the strategy choice in §2.3.

P0.3 is a planning task. The actual Sionna RT scene generation and replay validation belong to P2.6.

---

## 5. Open Questions

| ID | Question | Owner | Deadline | Notes |
|----|----------|-------|----------|-------|
| Q1 | Is a Sionna RT-capable GPU available on `qiankun@172.28.23.182`? | Lead engineer | 2026-08-15 | Sionna requires CUDA-compatible GPU with TensorFlow 2 |
| Q2 | Can we publish the custom UAV scenario to the public DeepMIMO DB? | Advisor | After Phase 2 P2.6 | Boosts citation impact; check license alignment with TWC pre-print policy |
| Q3 | Which carrier frequencies should the custom scenario cover? | Advisor | 2026-08-15 | Conference paper uses 3.5 GHz; consider adding mmWave (28 GHz) as ablation |

Each resolved decision should be recorded as an ADR addendum in `docs/DECISIONS.md`.

---

## 6. References

- DeepMIMO official site: <https://www.deepmimo.net>
- DeepMIMO v4 GitHub (Apache 2.0): <https://github.com/DeepMIMO/DeepMIMO>
- DeepMIMO scenarios database: <https://deepmimo.net/scenarios>
- Sionna RT (NVIDIA, ray tracer used by `dm.convert`): <https://nvlabs.github.io/sionna/api/rt.html>
- Original DeepMIMO paper: A. Alkhateeb, "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications," Proc. ITA, 2019. arXiv:1902.06435.
- 3GPP TR 38.901 (fallback): "Study on channel model for frequencies from 0.5 to 100 GHz," 3GPP TR 38.901 v17.0.0, 2022.
- Companion plan documents:
  - `docs/PLAN.md` §C.4.5 — real-channel trace requirement (Tier-1).
  - `docs/ROADMAP.md` P0.3, P2.6 — phasing.
  - `docs/SKETCHES.md` (no DeepMIMO section yet — TODO follow-up).
