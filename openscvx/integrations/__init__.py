"""External-backend dynamics adapters for OpenSCvx.

Two backends are currently supported. `MjxDynamics` wraps a MuJoCo MJX model;
`FraxDynamics` wraps a `frax.Robot` (Fast Robot Kinematics and Dynamics in
JAX). Both go directly into the ``dynamics=`` slot of `Problem` and construct
the matching State/Control objects for the user::

    from openscvx.integrations import FraxDynamics
    from frax.robots.franka_panda import load_panda

    dyn = FraxDynamics(load_panda())
    problem = ox.Problem(dynamics=dyn, states=dyn.states, controls=dyn.controls, ...)

For MJX, free-joint quaternion kinematics for floating-base models (drones,
humanoids) are detected and handled automatically::

    from openscvx.integrations import MjxDynamics

    dyn = MjxDynamics(mjx_model)
    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        ...
    )

For advanced users who need custom State/Control names (or to interleave
them with extra custom states), `mjx_dynamics` is exposed as the underlying
BYOF callable factory — assemble your own ``byof["dynamics"]`` dict from it.

All MJX symbols delegate lazily so ``mujoco.mjx`` is only imported when
actually used. The ``menagerie`` submodule is also loaded lazily.

Example — cartpole (``nq == nv``)::

    from openscvx.integrations import MjxDynamics

    dyn = MjxDynamics(mjx_model)
    problem = ox.Problem(dynamics=dyn, states=dyn.states, controls=dyn.controls, ...)

Example — quadrotor with free joint (``nq=7``, ``nv=6``)::

    from openscvx.integrations import MjxDynamics

    dyn = MjxDynamics(mjx_model)
    problem = ox.Problem(dynamics=dyn, states=dyn.states, controls=dyn.controls, ...)
"""

from typing import Any

from .base import DynamicsAdapter
from .frax import FraxDynamics
from .frax_cito import (
    CitoFraxDynamics,
    ContactModelConfig,
    DfohControlLayout,
    load_monoped_3d,
)
from .mjx import MjxDynamics


def mjx_dynamics(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate; imports ``mujoco.mjx`` on first call."""
    from .mjx import mjx_dynamics as _mjx_dynamics

    return _mjx_dynamics(*args, **kwargs)


def frax_dynamics(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate to the low-level frax BYOF factory."""
    from .frax import frax_dynamics as _frax_dynamics

    return _frax_dynamics(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "menagerie":
        from . import menagerie

        return menagerie
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DynamicsAdapter",
    "MjxDynamics",
    "mjx_dynamics",
    "FraxDynamics",
    "frax_dynamics",
    "CitoFraxDynamics",
    "ContactModelConfig",
    "DfohControlLayout",
    "load_monoped_3d",
    "menagerie",
]
