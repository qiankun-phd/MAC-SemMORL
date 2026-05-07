"""Lightweight MORL eval helpers used by external baselines.

The original agent.py defines ``generate_w_batch_test`` and
``evluate_Hv_UT_and_spa`` but importing it pulls in heavy training-only
dependencies (tensorboard, visdom, rltorch). Baselines only need the
preference grid + HV metric, so this module re-implements those two
functions with the smallest possible dependency footprint (numpy +
the existing hypervolume.py).
"""
from __future__ import annotations

import itertools
from copy import deepcopy

import numpy as np

from hypervolume import InnerHyperVolume


def generate_w_batch_test(reward_num: int, step_size: float) -> np.ndarray:
    """Uniform simplex grid; matches agent.py:262 verbatim."""
    mesh_array = [np.arange(0, 1 + step_size, step_size) for _ in range(reward_num)]
    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1, :]
    w_batch_test = np.unique(w_batch_test, axis=0)
    return w_batch_test


def _sparsity(obj_batch: np.ndarray) -> float:
    if len(obj_batch) <= 1:
        return 0.0
    sparsity = 0.0
    m = len(obj_batch[0])
    for dim in range(m):
        objs_i = np.sort(deepcopy(obj_batch.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    return float(sparsity / max(len(obj_batch) - 1, 1))


def evluate_Hv_UT_and_spa(
    obj_num: int, obj_batch: np.ndarray, PREF_: np.ndarray
) -> tuple:
    """Mirrors agent.py:145 — HV against zero ref point, mean utility over
    PREF_, and sparsity. Returns (hv, sparsity, ut). Spelling matches the
    original codebase to keep call sites identical (it's "evaluate" with a
    typo in agent.py)."""
    obj_batch = np.asarray(obj_batch, dtype=np.float64)
    ref_point = np.zeros(obj_num)
    HV = InnerHyperVolume(-ref_point)
    hv = HV.compute(obj_batch)
    sparsity = _sparsity(obj_batch)
    ut = 0.0
    for ref in PREF_:
        ut += float(np.max(np.sum(np.array(ref) * obj_batch, axis=-1)))
    ut /= max(len(PREF_), 1)
    return hv, sparsity, ut
