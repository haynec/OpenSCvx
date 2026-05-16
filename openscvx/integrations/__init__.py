"""External-backend dynamics adapters for OpenSCvx.

The recommended entry-point is `MjxDynamics`, which goes directly into the
``dynamics=`` slot of `Problem` and constructs the matching State/Control
objects for the user. Free-joint quaternion kinematics for floating-base
models (drones, humanoids) are detected and handled automatically::

    from openscvx.integrations import MjxDynamics

    dyn = MjxDynamics(mjx_model)
    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        ...
    )

The legacy `mjx_byof` helper is retained for users who need to supply their
own State/Control objects (e.g. to interleave with extra custom states or
rename them). `mjx_dynamics` is the underlying BYOF callable factory used by
both, exposed publicly for advanced users.

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
from .mjx import MjxDynamics


def mjx_byof(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate; imports ``mujoco.mjx`` on first call."""
    from .mjx import mjx_byof as _mjx_byof

    return _mjx_byof(*args, **kwargs)


def mjx_dynamics(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate; imports ``mujoco.mjx`` on first call."""
    from .mjx import mjx_dynamics as _mjx_dynamics

    return _mjx_dynamics(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "menagerie":
        from . import menagerie

        return menagerie
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DynamicsAdapter",
    "MjxDynamics",
    "mjx_byof",
    "mjx_dynamics",
    "menagerie",
]
