"""Adapters for MuJoCo MJX dynamics in OpenSCvx BYOF.

Bridges OpenSCvx's BYOF dynamics interface to MuJoCo MJX, a JAX-native
rigid-body physics backend. The adapters return functions matching the BYOF
dynamics signature ``(x, u, node, params) -> xdot`` so they can be plugged
directly into :class:`~openscvx.expert.ByofSpec`.

Available adapters
------------------
:func:`mjx_dynamics`
    Wraps a MuJoCo MJX model to provide generalized accelerations (``qacc``)
    for use as the ``qvel`` state derivative.

:func:`free_joint_qpos_dynamics`
    Pure-JAX quaternion kinematics for floating-base systems where ``nq > nv``.
    Use instead of the symbolic ``"qpos": qvel`` shorthand whenever the model
    contains a free/floating-base joint.

:func:`mjx_dynamics` and :func:`free_joint_qpos_dynamics` delegate to
``openscvx.integrations.mjx`` on first call so ``mujoco.mjx`` is only imported
when used. The :mod:`menagerie` submodule is loaded lazily via attribute access.

Example
-------
Wrapping MuJoCo MJX dynamics for a cartpole::

    import mujoco
    import mujoco.mjx as mjx
    import openscvx as ox
    from openscvx.integrations import mjx_dynamics

    mj_model = mujoco.MjModel.from_xml_path("cartpole.xml")
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mjx_model = mjx.put_model(mj_model)

    qpos = ox.State("qpos", shape=(mjx_model.nq,))
    qvel = ox.State("qvel", shape=(mjx_model.nv,))
    ctrl = ox.Control("ctrl", shape=(mjx_model.nu,))

    problem = ox.Problem(
        dynamics={"qpos": qvel},
        states=[qpos, qvel],
        controls=[ctrl],
        byof={"dynamics": {"qvel": mjx_dynamics(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}},
        ...
    )

For a quadrotor with a free joint (nq=7, nv=6)::

    from openscvx.integrations import free_joint_qpos_dynamics, mjx_dynamics

    byof = {
        "dynamics": {
            "qpos": free_joint_qpos_dynamics(qpos=qpos, qvel=qvel),
            "qvel": mjx_dynamics(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl),
        }
    }
"""

from typing import Any


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


__all__ = ["mjx_dynamics", "free_joint_qpos_dynamics", "menagerie"]
