"""Factory and registry for baselines (Issue #8 PR-B).

External baselines (MO-PPO, Pareto-PG, Pareto-Q, C-MORL, PSL-MORL) self-register
on import via ``register_baseline``. The runner script
``scripts/run_baseline.py`` looks up the right class via ``get_baseline(name)``.

Why a registry instead of direct imports: per-baseline ports may have heavyweight
dependencies (mo-gymnasium, mujoco-py, custom forks). Lazy-importing on demand
avoids paying the import cost when running a different baseline.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import Baseline


_REGISTRY: Dict[str, Type[Baseline]] = {}


def register_baseline(name: str):
    """Class decorator: register a Baseline subclass under ``name``.

    Example::

        @register_baseline("mo-ppo")
        class MoPPOBaseline(Baseline):
            name = "mo-ppo"
            ...
    """

    def deco(cls: Type[Baseline]) -> Type[Baseline]:
        if not issubclass(cls, Baseline):
            raise TypeError(f"{cls.__name__} is not a Baseline subclass")
        if name in _REGISTRY:
            raise ValueError(
                f"baseline {name!r} already registered to "
                f"{_REGISTRY[name].__name__}; cannot re-register {cls.__name__}"
            )
        _REGISTRY[name] = cls
        return cls

    return deco


def get_baseline(name: str) -> Type[Baseline]:
    """Lookup a registered Baseline class. Raises KeyError if missing."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown baseline {name!r}; registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_baselines() -> list:
    """Sorted list of registered baseline names. Useful for CLI choices."""
    return sorted(_REGISTRY.keys())
