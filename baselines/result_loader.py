"""Load + group baseline result npz files for analysis (Issue #8 PR-G).

Sits between the on-disk schema (DESIGN-baselines.md §3) and the comparison
tooling in ``experiments/analysis/compare_baselines.py``. Knows nothing
about plotting or table formatting — that's the analyzer's job.

Filename convention from ``baselines.npz_schema.make_config_tag``:
    {method}_M{m}_K{k}_mob{mob}_ch{ch}_pert{pert}_constr{handler}_seed{seed}.npz

This module provides:

- ``RunResult`` — dataclass mirroring the schema for one (method, config, seed)
- ``ResultGroup`` — N runs that share (method, config); seed-mean/std summaries
- ``load_results_dir`` — scan a directory and return list[ResultGroup]
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Filename pattern: capture method (allow internal hyphens), tag, seed.
_FNAME_RE = re.compile(
    r"^(?P<method>[A-Za-z0-9-]+?)"
    r"_M(?P<M>\d+)_K(?P<K>\d+)"
    r"_mob(?P<mob>[A-Za-z0-9]+)"
    r"_ch(?P<ch>[A-Za-z0-9]+)"
    r"_pert(?P<pert>[A-Za-z0-9]+)"
    r"_constr(?P<constr>[A-Za-z0-9]+)"
    r"_seed(?P<seed>\d+)\.npz$"
)


@dataclass
class RunResult:
    """One ``(method, config, seed)`` result file loaded into memory."""

    path: str
    method: str
    M: int
    K: int
    mobility: str
    channel: str
    perturbation: str
    constraint_handler: str  # "none" if unconstrained
    seed: int

    eval_steps: np.ndarray
    hv_trajectory: np.ndarray
    ut_trajectory: np.ndarray
    sparsity_trajectory: np.ndarray
    final_ep_objs: np.ndarray
    final_ep_prefs: np.ndarray
    final_obj_means: np.ndarray
    final_obj_stds: np.ndarray
    c_violation_rates: np.ndarray
    wallclock_seconds: float
    git_sha: str
    config_dict: Dict[str, Any]

    @property
    def config_key(self) -> Tuple[str, int, int, str, str, str, str]:
        """Used to group runs from different seeds."""
        return (
            self.method, self.M, self.K, self.mobility, self.channel,
            self.perturbation, self.constraint_handler,
        )

    @property
    def final_hv(self) -> float:
        return (
            float(self.hv_trajectory[-1])
            if self.hv_trajectory.size > 0 else float("nan")
        )

    @property
    def final_ut(self) -> float:
        return (
            float(self.ut_trajectory[-1])
            if self.ut_trajectory.size > 0 else float("nan")
        )


@dataclass
class ResultGroup:
    """All seeds for one (method, config). Holds aggregate statistics."""

    method: str
    M: int
    K: int
    mobility: str
    channel: str
    perturbation: str
    constraint_handler: str
    runs: List[RunResult] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        return len(self.runs)

    def final_hv_mean_std(self) -> Tuple[float, float]:
        vals = np.array([r.final_hv for r in self.runs], dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    def final_ut_mean_std(self) -> Tuple[float, float]:
        vals = np.array([r.final_ut for r in self.runs], dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    def final_obj_means_stacked(self) -> np.ndarray:
        """Shape (n_seeds, N) of per-objective episode means at the final step."""
        if not self.runs:
            return np.zeros((0, 4))
        return np.stack([r.final_obj_means for r in self.runs])

    def violation_rates_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """Shape ((3,), (3,)). Returns all-NaN if every run is unconstrained
        (so callers can detect that case without parsing warnings)."""
        arr = np.stack([r.c_violation_rates for r in self.runs])
        if np.all(np.isnan(arr)):
            return np.full(3, np.nan), np.full(3, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    def all_final_ep_objs(self) -> np.ndarray:
        """Concatenate every seed's Pareto points into one (P_total, N) array.
        Used for cross-seed Pareto-front plotting."""
        if not self.runs:
            return np.zeros((0, 4))
        return np.concatenate([r.final_ep_objs for r in self.runs], axis=0)


def parse_fname(fname: str) -> Optional[Dict[str, Any]]:
    """Returns the parsed components of ``fname`` if it matches the schema,
    else None."""
    m = _FNAME_RE.match(os.path.basename(fname))
    if m is None:
        return None
    g = m.groupdict()
    return {
        "method": g["method"],
        "M": int(g["M"]),
        "K": int(g["K"]),
        "mobility": g["mob"],
        "channel": g["ch"],
        "perturbation": g["pert"],
        "constraint_handler": g["constr"],
        "seed": int(g["seed"]),
    }


def load_run(path: str) -> RunResult:
    """Load a single npz into a RunResult. Raises if the schema is malformed."""
    parts = parse_fname(path)
    if parts is None:
        raise ValueError(
            f"filename does not match schema: {os.path.basename(path)}"
        )
    data = dict(np.load(path, allow_pickle=True))
    cfg_raw = data.get("config_dict")
    cfg = json.loads(str(cfg_raw)) if cfg_raw is not None else {}
    return RunResult(
        path=path,
        **parts,
        eval_steps=np.asarray(data["eval_steps"]),
        hv_trajectory=np.asarray(data["hv_trajectory"]),
        ut_trajectory=np.asarray(data["ut_trajectory"]),
        sparsity_trajectory=np.asarray(data["sparsity_trajectory"]),
        final_ep_objs=np.asarray(data["final_ep_objs"]),
        final_ep_prefs=np.asarray(data["final_ep_prefs"]),
        final_obj_means=np.asarray(data["final_obj_means"]),
        final_obj_stds=np.asarray(data["final_obj_stds"]),
        c_violation_rates=np.asarray(data["c_violation_rates"]),
        wallclock_seconds=float(data["wallclock_seconds"]),
        git_sha=str(data["git_sha"]),
        config_dict=cfg,
    )


def load_results_dir(
    results_dir: str,
    method_filter: Optional[Iterable[str]] = None,
) -> List[ResultGroup]:
    """Scan results_dir for npz files matching the schema, group by config.

    Files that don't match the filename schema are silently skipped; raise on
    schema-malformed npz contents (missing keys).
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")
    method_set = set(method_filter) if method_filter else None

    groups: Dict[Tuple, ResultGroup] = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(results_dir, fname)
        parts = parse_fname(fname)
        if parts is None:
            continue
        if method_set is not None and parts["method"] not in method_set:
            continue
        run = load_run(path)
        key = run.config_key
        if key not in groups:
            groups[key] = ResultGroup(
                method=parts["method"], M=parts["M"], K=parts["K"],
                mobility=parts["mobility"], channel=parts["channel"],
                perturbation=parts["perturbation"],
                constraint_handler=parts["constraint_handler"],
            )
        groups[key].runs.append(run)
    return list(groups.values())
