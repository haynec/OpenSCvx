"""Base class for external-backend dynamics adapters.

A `DynamicsAdapter` is the easy on-ramp for users who want to plug an
external physics backend (MuJoCo MJX, Brax, Drake, ...) into OpenSCvx without
manually constructing State/Control objects with matching shapes or routing
raw JAX callables through the expert ``byof`` channel.

The intended call site is::

    dyn = ox.MjxDynamics(mjx_model)
    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        ...
    )

Internally, `Problem` detects the adapter, calls `DynamicsAdapter.expand`,
and merges the resulting BYOF callables into the user's ``byof`` dict (if
any). Everything downstream sees ordinary ``dynamics`` and ``byof`` dicts.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


class DynamicsAdapter(ABC):
    """Abstract base class for external-backend dynamics adapters.

    Subclasses describe the State/Control objects they synthesize on
    ``.states`` / ``.controls`` and implement `expand` to return the
    two-channel ``(dynamics_dict, byof_dict)`` representation consumed by
    `Problem`.

    The split mirrors the existing two-channel API: ``dynamics_dict`` carries
    symbolic Expr entries (e.g. ``{"qpos": qvel}``) while ``byof_dict`` carries
    raw JAX callables under the ``"dynamics"`` key. Either or both may be
    empty, but ``expand()`` should never silently produce overlapping keys.
    """

    states: list["State"]
    controls: list["Control"]

    @abstractmethod
    def expand(self) -> Tuple[dict, dict]:
        """Return ``(dynamics_dict, byof_dict)`` in OpenSCvx's internal form.

        ``dynamics_dict`` maps state names to symbolic ``Expr`` derivatives
        (the same shape as the ``dynamics=`` argument to ``Problem``).
        ``byof_dict`` has the same shape as the ``byof=`` argument: its
        ``"dynamics"`` key (if present) maps state names to raw JAX callables.
        """


def _merge_byof(user_byof: dict | None, extra_byof: dict) -> dict:
    """Merge an adapter-synthesized BYOF dict into a user-provided one.

    Only the ``"dynamics"`` sub-dict is deep-merged; other keys are taken
    verbatim from whichever side provides them. Raises ``ValueError`` on any
    key collision under ``"dynamics"`` — a user passing both
    ``dynamics=ox.MjxDynamics(...)`` and ``byof={"dynamics": {"qvel": ...}}``
    almost certainly has a bug, and silent override would mask it.
    """
    if not user_byof:
        return copy.copy(extra_byof)

    merged = dict(user_byof)
    extra_dyn = extra_byof.get("dynamics", {})
    user_dyn = user_byof.get("dynamics", {})

    if extra_dyn:
        collisions = set(user_dyn) & set(extra_dyn)
        if collisions:
            raise ValueError(
                "DynamicsAdapter produced byof['dynamics'] entries that "
                f"collide with user-provided byof['dynamics']: {sorted(collisions)}. "
                "Drop the duplicate keys from your byof dict, or drop the adapter "
                "and assemble byof['dynamics'] manually for full control."
            )
        merged["dynamics"] = {**user_dyn, **extra_dyn}

    return merged
