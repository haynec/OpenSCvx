"""Discretization methods for trajectory optimization.

This module provides implementations of discretization schemes that convert
continuous-time optimal control problems into discrete-time approximations
suitable for numerical optimization.

Discretization and linearization are combined into a single interface
(:class:`Discretizer`) because different schemes may linearize then discretize,
discretize then linearize, or use other approaches. The ordering changes the
intermediate types, but the input (continuous nonlinear dynamics + reference
trajectory) and output (discrete-time linear matrices A_d, B_d, C_d) are
always consistent.

The default implementation is :class:`LinearizeDiscretize`, which computes
continuous-time Jacobians via JAX autodiff and integrates them alongside the
nonlinear dynamics through an augmented state vector.
"""

import inspect
from typing import Any

from .base import Discretizer
from .linearize_discretize import LinearizeDiscretize

# ---------------------------------------------------------------------------
# Spec resolver — turn a dict into a Discretizer instance
# ---------------------------------------------------------------------------

_DISCRETIZER_MAP = {
    "LinearizeDiscretize": LinearizeDiscretize,
}


def _resolve_discretizer(val: Any) -> Discretizer:
    """Resolve a discretizer specification into an instance.

    Accepted forms:

    * **instance** — already-constructed :class:`Discretizer` (pass-through).
    * **dict** — keyword arguments passed to :class:`LinearizeDiscretize`.
      An optional ``"type"`` key selects the class (currently only
      ``"LinearizeDiscretize"``).

    Examples::

        # Dict with keyword overrides (default class)
        _resolve_discretizer({"dis_type": "ZOH", "solver": "Dopri8"})

        # Dict with explicit type
        _resolve_discretizer({"type": "LinearizeDiscretize", "dis_type": "ZOH"})

        # Instance pass-through
        _resolve_discretizer(LinearizeDiscretize(dis_type="ZOH"))
    """
    if isinstance(val, Discretizer):
        return val

    if not isinstance(val, dict):
        raise TypeError(f"Expected a Discretizer instance or dict, got {type(val).__name__}")

    kwargs = dict(val)  # copy to avoid mutating caller's dict
    name = kwargs.pop("type", "LinearizeDiscretize")

    cls = _DISCRETIZER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown discretizer {name!r}; expected one of {sorted(_DISCRETIZER_MAP)}"
        )

    try:
        return cls(**kwargs)
    except TypeError as e:
        valid = list(inspect.signature(cls.__init__).parameters.keys())
        valid.remove("self")
        raise TypeError(f"Invalid discretizer keyword argument: {e}. Valid keys: {valid}") from None


__all__ = [
    "Discretizer",
    "LinearizeDiscretize",
    "_resolve_discretizer",
]
