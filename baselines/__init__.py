"""Baseline registry for the journal extension (Issue #8 PR-B).

Two responsibilities:

1. Backward compatibility — re-export every top-level symbol from the original
   ``baselines.py`` (now ``heuristics.py``) so callers like
   ``experiments/plot_uav_trajectory.py``, ``experiments/mobility_ablation_eval.py``,
   ``experiments/analysis/reward_diagnosis.py``, and
   ``experiments/analysis/calibrate_rewards.py`` keep working without edits.

2. New baseline framework — the ``Baseline`` ABC, the ``register_baseline`` /
   ``get_baseline`` factory, and the on-disk npz schema utilities described in
   ``docs/DESIGN-baselines.md`` §3-§4.

External baselines (MO-PPO, Pareto-PG, Pareto-Q, C-MORL, PSL-MORL) land in
follow-up PRs and import from this package using ``register_baseline``.
"""
from __future__ import annotations

# --- Backward-compat re-exports from the heuristics module --------------
from .heuristics import (  # noqa: F401
    make_uav_semcom_env,
    evaluate_policy,
    evaluate_comm_metrics,
    fixed_trajectory_greedy_power,
    greedy_aosi,
    random_policy,
    nearest_device_round_robin,
)

# --- New framework ------------------------------------------------------
from .base import Baseline
from .registry import register_baseline, get_baseline, list_baselines
from .npz_schema import write_result_npz, validate_result_npz, NpzResult
from .result_loader import RunResult, ResultGroup, load_run, load_results_dir, parse_fname

__all__ = [
    # backward-compat
    "make_uav_semcom_env",
    "evaluate_policy",
    "evaluate_comm_metrics",
    "fixed_trajectory_greedy_power",
    "greedy_aosi",
    "random_policy",
    "nearest_device_round_robin",
    # new framework
    "Baseline",
    "register_baseline",
    "get_baseline",
    "list_baselines",
    "write_result_npz",
    "validate_result_npz",
    "NpzResult",
    "RunResult",
    "ResultGroup",
    "load_run",
    "load_results_dir",
    "parse_fname",
]

# Auto-register the no-op baseline so ``run_baseline.py --baseline noop`` works
# without any further setup. Each external baseline added in a later PR will
# register itself by importing it here.
from . import noop  # noqa: F401
from . import mo_ppo  # noqa: F401  -- registers "mo-ppo"
from . import pareto_pg  # noqa: F401  -- registers "pareto-pg"
from . import pareto_q  # noqa: F401  -- registers "pareto-q"
from . import c_morl  # noqa: F401  -- registers "c-morl" (stub)
from . import psl_morl  # noqa: F401  -- registers "psl-morl" (stub)
