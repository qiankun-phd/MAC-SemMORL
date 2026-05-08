"""On-disk result schema for Issue #8 baselines (DESIGN-baselines.md §3).

Single source of truth for what a result npz must contain. The runner
``scripts/run_baseline.py`` calls ``write_result_npz`` after training; the
analysis pipeline (PR-G of Issue #8) reads via ``np.load`` directly using
the documented keys.

Default decision (DESIGN-baselines.md §7 Q3): one npz per
(method, config, seed). This makes resume-after-failure easier and lets
``analyze_results.py`` group on the (method, config) prefix when assembling
seed-mean-std rows.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Schema definition — every key listed here is required in the saved npz.
# Adding a new key is a breaking change to the analysis pipeline; remove or
# rename only with explicit reviewer approval.
# ---------------------------------------------------------------------------
SCHEMA_KEYS = (
    "eval_steps",
    "hv_trajectory",
    "ut_trajectory",
    "sparsity_trajectory",
    "final_ep_objs",
    "final_ep_prefs",
    "final_obj_means",
    "final_obj_stds",
    "c_violation_rates",
    "wallclock_seconds",
    "git_sha",
    "config_dict",
)


@dataclass
class NpzResult:
    """Container that mirrors the on-disk npz schema. Subclasses of
    Baseline build one of these and pass it to ``write_result_npz``."""

    eval_steps: np.ndarray
    hv_trajectory: np.ndarray
    ut_trajectory: np.ndarray
    sparsity_trajectory: np.ndarray
    final_ep_objs: np.ndarray
    final_ep_prefs: np.ndarray
    final_obj_means: np.ndarray
    final_obj_stds: np.ndarray
    c_violation_rates: np.ndarray  # shape (3,), NaN if unconstrained
    wallclock_seconds: float
    git_sha: str = ""
    config_dict: Dict[str, Any] = field(default_factory=dict)

    def to_npz_kwargs(self) -> Dict[str, np.ndarray]:
        """Convert to the dict np.savez expects. Scalars become 0-d arrays;
        the JSON-encoded config dict is stored as a length-1 string array."""
        return {
            "eval_steps": np.asarray(self.eval_steps, dtype=np.int64),
            "hv_trajectory": np.asarray(self.hv_trajectory, dtype=np.float64),
            "ut_trajectory": np.asarray(self.ut_trajectory, dtype=np.float64),
            "sparsity_trajectory": np.asarray(
                self.sparsity_trajectory, dtype=np.float64
            ),
            "final_ep_objs": np.asarray(self.final_ep_objs, dtype=np.float64),
            "final_ep_prefs": np.asarray(self.final_ep_prefs, dtype=np.float64),
            "final_obj_means": np.asarray(self.final_obj_means, dtype=np.float64),
            "final_obj_stds": np.asarray(self.final_obj_stds, dtype=np.float64),
            "c_violation_rates": np.asarray(
                self.c_violation_rates, dtype=np.float64
            ),
            "wallclock_seconds": np.array(float(self.wallclock_seconds)),
            "git_sha": np.array(self.git_sha, dtype=object),
            "config_dict": np.array(json.dumps(self.config_dict), dtype=object),
        }


def _detect_git_sha() -> str:
    """Return the current HEAD short SHA, or empty string if not in a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def make_config_tag(
    method: str,
    M: int,
    K: int,
    mobility: str = "none",
    channel: str = "analytical",
    perturbation: str = "none",
    constraint_handler: Optional[str] = None,
) -> str:
    """Build the canonical filename tag from §3 of the design doc.

    Example: ``C-MORL_M2_K5_mobnone_chanalytical_pertnone_constrlagrangian``
    """
    constr = constraint_handler or "none"
    return (
        f"{method}"
        f"_M{M}_K{K}"
        f"_mob{mobility}_ch{channel}"
        f"_pert{perturbation}_constr{constr}"
    )


def write_result_npz(
    result: NpzResult,
    output_dir: str,
    method: str,
    M: int,
    K: int,
    seed: int,
    mobility: str = "none",
    channel: str = "analytical",
    perturbation: str = "none",
    constraint_handler: Optional[str] = None,
) -> str:
    """Write `result` to `output_dir` using the schema-defined filename.

    Returns the full path to the saved file. Auto-fills ``git_sha`` if the
    caller left it empty.
    """
    tag = make_config_tag(
        method=method, M=M, K=K, mobility=mobility, channel=channel,
        perturbation=perturbation, constraint_handler=constraint_handler,
    )
    fname = f"{tag}_seed{seed}.npz"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fname)

    if not result.git_sha:
        result.git_sha = _detect_git_sha()

    np.savez(path, **result.to_npz_kwargs())
    return path


def validate_result_npz(path: str) -> Dict[str, Any]:
    """Load the npz at `path` and verify it contains every schema key.

    Returns the loaded dict (all keys present) on success; raises
    ``AssertionError`` listing missing keys on failure.
    """
    data = dict(np.load(path, allow_pickle=True))
    missing = [k for k in SCHEMA_KEYS if k not in data]
    if missing:
        raise AssertionError(
            f"npz file {path} is missing required schema keys: {missing}\n"
            f"present keys: {sorted(data.keys())}"
        )
    return data
