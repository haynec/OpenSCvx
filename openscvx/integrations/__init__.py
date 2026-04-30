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

Both adapters are imported lazily so that ``mujoco.mjx`` only needs to be
installed when they are actually used.

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


def __getattr__(name):
    if name == "mjx_dynamics":
        from .mjx import mjx_dynamics
        return mjx_dynamics
    if name == "free_joint_qpos_dynamics":
        from .mjx import free_joint_qpos_dynamics
        return free_joint_qpos_dynamics
    if name == "menagerie":
        from . import menagerie
        return menagerie
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["mjx_dynamics", "free_joint_qpos_dynamics", "menagerie"]
