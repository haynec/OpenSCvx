"""Adapters for MuJoCo MJX dynamics in OpenSCvx BYOF.

The recommended entry-point is :func:`mjx_byof`, which returns a complete
``byof["dynamics"]`` dict and automatically handles free-joint quaternion
kinematics for floating-base models (drones, humanoids, etc.):

    from openscvx.integrations import mjx_byof

    byof = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

For models without free joints (cartpoles, manipulators) the returned dict
contains only ``"qvel"`` and ``dynamics={"qpos": qvel}`` should still be
provided to :class:`~openscvx.Problem`.  For models with free joints
(``nq > nv``) ``"qpos"`` is included automatically — no extra imports needed.

Lower-level building blocks (advanced use)
------------------------------------------
:func:`mjx_dynamics`
    Returns a single BYOF callable for the ``qvel`` derivative (qacc).

:func:`free_joint_qpos_dynamics`
    Returns a BYOF callable for ``qpos`` when the model has quaternion free
    joints (``nq > nv``).

All symbols delegate lazily so ``mujoco.mjx`` is only imported when used.
The :mod:`menagerie` submodule is loaded lazily via attribute access.

Example — cartpole (nq == nv)::

    from openscvx.integrations import mjx_byof

    byof = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}
    problem = ox.Problem(dynamics={"qpos": qvel}, byof=byof, ...)

Example — quadrotor with free joint (nq=7, nv=6)::

    from openscvx.integrations import mjx_byof

    byof = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}
    problem = ox.Problem(dynamics={}, byof=byof, ...)
"""

from typing import Any


def mjx_byof(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate; imports ``mujoco.mjx`` on first call."""
    from .mjx import mjx_byof as _mjx_byof

    return _mjx_byof(*args, **kwargs)


def mjx_dynamics(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate; imports ``mujoco.mjx`` on first call."""
    from .mjx import mjx_dynamics as _mjx_dynamics

    return _mjx_dynamics(*args, **kwargs)


def free_joint_qpos_dynamics(*args: Any, **kwargs: Any) -> Any:
    """Lazy delegate; imports ``mujoco.mjx`` on first call."""
    from .mjx import free_joint_qpos_dynamics as _free_joint_qpos_dynamics

    return _free_joint_qpos_dynamics(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "menagerie":
        from . import menagerie

        return menagerie
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["mjx_byof", "mjx_dynamics", "free_joint_qpos_dynamics", "menagerie"]
