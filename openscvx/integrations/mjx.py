"""MuJoCo MJX dynamics adapters for OpenSCvx BYOF.

The recommended entry-point is :func:`mjx_byof`, which returns a complete
``byof["dynamics"]`` dict and automatically handles free-joint quaternion
kinematics — no separate imports required:

    byof = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

For models **without** free joints (cartpoles, manipulators, etc.) the
returned dict contains only ``"qvel"``, and qpos kinematics must still be
specified symbolically via ``dynamics={"qpos": qvel}``.  For models **with**
free joints (drones, humanoids) ``"qpos"`` is included automatically and no
symbolic dynamics entry is needed.

Lower-level building blocks (for advanced users):

* :func:`mjx_dynamics` — returns a single BYOF callable for ``qvel`` (qacc).
* :func:`free_joint_qpos_dynamics` — returns a BYOF callable for ``qpos``
  when ``nq > nv`` (quaternion free-joint kinematics).

Note:
    Time dilation is handled automatically by the BYOF lowering pipeline; all
    functions return physical (un-dilated) quantities.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

import jax.numpy as jnp

if TYPE_CHECKING:
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


def _resolve_slice(arg, name: str) -> slice:
    """Accept either a State/Control or a slice and return the slice."""
    if hasattr(arg, "slice"):
        sl = arg.slice
        if sl is None:
            raise ValueError(
                f"{name} has no .slice yet — pass it after Problem construction has called "
                "preprocessing, or pass an explicit slice."
            )
        return sl
    if isinstance(arg, slice):
        return arg
    raise TypeError(f"{name} must be a State/Control or slice, got {type(arg).__name__}")


def mjx_dynamics(
    mjx_model: Any,
    *,
    qpos: "State | slice",
    qvel: "State | slice",
    ctrl: "Control | slice",
    return_component: str = "qacc",
    extra_postprocess: Optional[Callable[[Any], Any]] = None,
) -> Callable:
    """Wrap a MuJoCo MJX model as a BYOF dynamics function.

    Args:
        mjx_model: A model produced by :func:`mujoco.mjx.put_model`. Must be a
            JAX pytree (the standard MJX representation).
        qpos: Position state (or slice into the unified ``x`` vector).
            Length must equal ``mjx_model.nq``.
        qvel: Velocity state (or slice). Length must equal ``mjx_model.nv``.
        ctrl: Control variable (or slice into the unified ``u`` vector).
            Length must equal ``mjx_model.nu``.
        return_component: Which MJX field to return. ``"qacc"`` (default)
            returns the generalized acceleration ``qacc`` for use as the
            ``qvel`` state derivative. ``"qvel"`` returns ``qvel`` for use as
            the ``qpos`` state derivative (rarely needed because that is
            already symbolic).
        extra_postprocess: Optional callable applied to the MJX ``data`` after
            ``mjx.forward``. Useful for computing custom outputs (e.g. site
            positions) used elsewhere.

    Returns:
        A function ``f(x, u, node, params) -> jnp.ndarray`` matching the BYOF
        dynamics signature.

    Raises:
        ImportError: If ``mujoco.mjx`` is not installed.
        ValueError: If ``return_component`` is not one of the allowed values.

    Note:
        MJX's contact solver uses ``lax.while_loop``, which is **not**
        reverse-mode differentiable. For contact-free systems (manipulators,
        cartpoles, etc.) disable contacts before uploading the model::

            mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            mjx_model = mjx.put_model(mj_model)

    Example:
        Cartpole swing-up dynamics::

            import mujoco
            import mujoco.mjx as mjx
            import openscvx as ox
            from openscvx.integrations import mjx_dynamics

            mj_model = mujoco.MjModel.from_xml_path("cartpole.xml")
            mjx_model = mjx.put_model(mj_model)

            qpos = ox.State("qpos", shape=(mjx_model.nq,))
            qvel = ox.State("qvel", shape=(mjx_model.nv,))
            ctrl = ox.Control("ctrl", shape=(mjx_model.nu,))

            qvel_dynamics = mjx_dynamics(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

            problem = ox.Problem(
                dynamics={"qpos": qvel},
                byof={"dynamics": {"qvel": qvel_dynamics}},
                states=[qpos, qvel],
                controls=[ctrl],
                ...
            )
    """
    try:
        import mujoco.mjx as mjx
    except ImportError as e:
        raise ImportError(
            "mujoco.mjx is required for mjx_dynamics. Install with: pip install openscvx[mjx]"
        ) from e

    if return_component not in ("qacc", "qvel"):
        raise ValueError(f"return_component must be 'qacc' or 'qvel', got {return_component!r}")

    # Store the raw args; slices are resolved lazily on first call so that
    # mjx_dynamics() can be called before Problem construction assigns .slice.
    _qpos_arg = qpos
    _qvel_arg = qvel
    _ctrl_arg = ctrl
    _resolved: list = []  # populated on first call: [qpos_slice, qvel_slice, ctrl_slice]

    def f(x, u, node, params):
        del node, params  # MJX dynamics are stateless w.r.t. node and OpenSCvx params
        if not _resolved:
            _resolved.append(_resolve_slice(_qpos_arg, "qpos"))
            _resolved.append(_resolve_slice(_qvel_arg, "qvel"))
            _resolved.append(_resolve_slice(_ctrl_arg, "ctrl"))
        qpos_slice, qvel_slice, ctrl_slice = _resolved

        qpos_val = x[qpos_slice]
        qvel_val = x[qvel_slice]
        ctrl_val = u[ctrl_slice]

        data = mjx.make_data(mjx_model)
        data = data.replace(qpos=qpos_val, qvel=qvel_val, ctrl=ctrl_val)
        data = mjx.forward(mjx_model, data)

        if extra_postprocess is not None:
            data = extra_postprocess(data)

        if return_component == "qacc":
            return jnp.asarray(data.qacc)
        return jnp.asarray(data.qvel)

    return f


def free_joint_qpos_dynamics(
    *,
    qpos: "State | slice",
    qvel: "State | slice",
    n_free_joints: int = 1,
) -> Callable:
    """BYOF dynamics for ``qpos`` when the system has quaternion free joints.

    When a MuJoCo model has a floating-base free joint, ``nq > nv`` because
    each quaternion orientation contributes 4 position DOF but only 3 angular
    velocity DOF. The simple symbolic shorthand ``"qpos": qvel`` therefore
    fails a shape check. This function returns a BYOF dynamics callable that
    correctly computes ``qdot`` from ``(q, qd)`` by applying the quaternion
    kinematic equation::

        [x_dot; q_dot; joint_dot] = [v; 0.5 * Xi(q) @ omega; qdot_joints]

    Args:
        qpos: Position state (or slice). Expected shape
            ``(7 * n_free + n_joints,)``.
        qvel: Velocity state (or slice). Expected shape
            ``(6 * n_free + n_joints,)``.
        n_free_joints: Number of free / floating-base joints (default 1). Each
            contributes 7 position DOF (3 translation + 4 quaternion) and 6
            velocity DOF (3 linear + 3 angular).

    Returns:
        A BYOF dynamics function ``f(x, u, node, params) -> qdot`` whose
        output has the same shape as ``qpos``.

    Note:
        The quaternion convention assumed here is ``[qw, qx, qy, qz]`` with
        angular velocity expressed in the **world frame**, which matches
        MuJoCo MJX's free-joint convention. If your system expresses angular
        velocity in the body frame, negate the off-diagonal terms in the
        integration matrix.

    Example:
        Quadrotor with one free joint (position + attitude)::

            from openscvx.integrations import free_joint_qpos_dynamics, mjx_dynamics

            qpos = ox.State("qpos", shape=(mjx_model.nq,))  # nq = 7
            qvel = ox.State("qvel", shape=(mjx_model.nv,))  # nv = 6

            byof = {
                "dynamics": {
                    "qpos": free_joint_qpos_dynamics(qpos=qpos, qvel=qvel),
                    "qvel": mjx_dynamics(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl),
                }
            }
    """
    _qpos_arg = qpos
    _qvel_arg = qvel
    _resolved: list = []

    def f(x, u, node, params):
        del u, node, params
        if not _resolved:
            _resolved.append(_resolve_slice(_qpos_arg, "qpos"))
            _resolved.append(_resolve_slice(_qvel_arg, "qvel"))
        qpos_sl, qvel_sl = _resolved

        q = x[qpos_sl]  # (7 * n_free + n_joints,)
        qd = x[qvel_sl]  # (6 * n_free + n_joints,)

        parts_q: list = []
        q_offset = 0
        qd_offset = 0

        for _ in range(n_free_joints):
            # Translation: qdot[:3] = qd[:3]
            parts_q.append(qd[qd_offset : qd_offset + 3])

            # Quaternion kinematics: qdot[3:7] = 0.5 * Xi(q[3:7]) @ omega
            #
            #   Xi(q) = [[-qx, -qy, -qz],
            #             [ qw, -qz,  qy],
            #             [ qz,  qw, -qx],
            #             [-qy,  qx,  qw]]
            qw = q[q_offset + 3]
            qx = q[q_offset + 4]
            qy = q[q_offset + 5]
            qz = q[q_offset + 6]
            wx = qd[qd_offset + 3]
            wy = qd[qd_offset + 4]
            wz = qd[qd_offset + 5]
            quat_dot = 0.5 * jnp.array(
                [
                    -qx * wx - qy * wy - qz * wz,
                    qw * wx + qy * wz - qz * wy,
                    qw * wy - qx * wz + qz * wx,
                    qw * wz + qx * wy - qy * wx,
                ]
            )
            parts_q.append(quat_dot)

            q_offset += 7
            qd_offset += 6

        # Remaining revolute / prismatic joints: 1-to-1 pass-through
        parts_q.append(qd[qd_offset:])

        return jnp.concatenate(parts_q)

    return f


def mjx_byof(
    mjx_model: Any,
    *,
    qpos: "State | slice",
    qvel: "State | slice",
    ctrl: "Control | slice",
    return_component: str = "qacc",
    extra_postprocess: Optional[Callable[[Any], Any]] = None,
) -> dict:
    """Return a complete ``byof["dynamics"]`` dict for a MuJoCo MJX model.

    This is the recommended high-level entry-point.  It inspects the model's
    ``nq`` and ``nv`` to detect free joints and automatically includes the
    quaternion kinematics callable for ``qpos`` when needed — no separate
    import of :func:`free_joint_qpos_dynamics` is required.

    Args:
        mjx_model: A model produced by :func:`mujoco.mjx.put_model`.
        qpos: Position state (or slice). Length must equal ``mjx_model.nq``.
        qvel: Velocity state (or slice). Length must equal ``mjx_model.nv``.
        ctrl: Control variable (or slice). Length must equal ``mjx_model.nu``.
        return_component: Passed to :func:`mjx_dynamics`. ``"qacc"``
            (default) uses the generalized acceleration as the ``qvel``
            derivative; ``"qvel"`` returns qvel directly (rarely needed).
        extra_postprocess: Optional callable applied to the MJX ``data``
            object after ``mjx.forward``. Passed through to
            :func:`mjx_dynamics`.

    Returns:
        A dict suitable for use as ``byof["dynamics"]``.
        For models **without** free joints (``nq == nv``) only ``"qvel"`` is
        included; position kinematics should still be provided symbolically
        via ``dynamics={"qpos": qvel}``.
        For models **with** free joints (``nq > nv``) both ``"qpos"`` and
        ``"qvel"`` are included and no symbolic ``dynamics`` entry is needed.

    Example:
        Cartpole (nq == nv, no free joint)::

            byof = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}
            problem = ox.Problem(
                dynamics={"qpos": qvel},   # still required for non-free models
                byof=byof, ...
            )

        Quadrotor / drone (nq > nv, one free joint)::

            byof = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}
            problem = ox.Problem(
                dynamics={},               # qpos handled automatically
                byof=byof, ...
            )
    """
    nq = int(mjx_model.nq)
    nv = int(mjx_model.nv)

    result: dict = {
        "qvel": mjx_dynamics(
            mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            return_component=return_component,
            extra_postprocess=extra_postprocess,
        ),
    }

    n_free = nq - nv  # each free joint contributes exactly 1 extra position DOF
    if n_free > 0:
        result["qpos"] = free_joint_qpos_dynamics(
            qpos=qpos,
            qvel=qvel,
            n_free_joints=n_free,
        )

    return result
