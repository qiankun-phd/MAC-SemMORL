"""DeepMIMO replay shim.

This module is intentionally minimal at P0.3:
- Provide a stable interface for loading a DeepMIMO sample (or equivalent).
- Provide helpers to convert it into pathloss/SNR style metrics that the
  MAC-SemMORL environment expects.

DeepMIMO is a MATLAB-based dataset generator; Python users typically rely on
pre-exported arrays (e.g., `.npz`) or an external conversion script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeepMIMOSample:
    """Container for one channel snapshot.

    The only hard requirement for the replay pipeline is the complex channel
    coefficient tensor `h`.

    Expected shape: (..., n_rx, n_tx) or (n_sc, n_rx, n_tx). This shim is
    agnostic to how you choose to represent time/subcarriers; downstream helpers
    will reduce it to an average channel gain.
    """

    h: np.ndarray
    meta: dict[str, Any] | None = None


def load_npz(path: str | Path, *, key: str = "h") -> DeepMIMOSample:
    """Load a DeepMIMO-exported sample from `.npz`.

    Parameters
    ----------
    path:
        `.npz` file containing at least one complex-valued array.
    key:
        Name of the array for the complex channel tensor.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"Missing array {key!r} in {path}")
        h = data[key]

    if not np.iscomplexobj(h):
        raise TypeError(
            "Channel tensor must be complex-valued. "
            "Export from MATLAB with real+imag or complex dtype and save as npz."
        )

    return DeepMIMOSample(h=h)


def effective_channel_gain(sample: DeepMIMOSample) -> float:
    """Return average |h|^2 over all dimensions."""

    h = np.asarray(sample.h)
    if h.size == 0:
        raise ValueError("Empty channel tensor")
    return float(np.mean(np.abs(h) ** 2))


def snr_db(
    sample: DeepMIMOSample,
    *,
    tx_power_dbm: float,
    noise_figure_db: float = 7.0,
    bandwidth_hz: float = 20e6,
    noise_density_dbm_per_hz: float = -174.0,
) -> float:
    """Compute an SNR proxy in dB from one DeepMIMO sample.

    This is a pragmatic "replay" metric for the environment's PHY layer.
    It ignores array processing and treats the average channel gain as a scalar.
    """

    g = effective_channel_gain(sample)
    rx_power_dbm = tx_power_dbm + 10.0 * np.log10(g)
    noise_dbm = noise_density_dbm_per_hz + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
    return float(rx_power_dbm - noise_dbm)

