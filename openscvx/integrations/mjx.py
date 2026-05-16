"""MuJoCo MJX dynamics adapters for OpenSCvx.

The recommended entry-point is `MjxDynamics`, a `DynamicsAdapter` that goes
directly into the ``dynamics=`` slot of `Problem` and exposes the synthesized
State/Control objects on ``.states`` / ``.controls``::

    dyn = ox.MjxDynamics(mjx_model)
    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        ...
    )

Free-joint quaternion kinematics (``nq > nv`` models such as drones or
humanoids) are detected and handled automatically.

For advanced users, the lower-level `mjx_dynamics` callable factory and the
legacy `mjx_byof` helper remain available.

Note:
    Time dilation is handled automatically by the BYOF lowering pipeline; all
    functions return physical (un-dilated) quantities.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import jax.numpy as jnp

from openscvx.integrations.base import DynamicsAdapter

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


def _free_joint_qpos_dynamics(
    *,
    qpos: "State | slice",
    qvel: "State | slice",
    n_free_joints: int = 1,
) -> Callable:
    """BYOF callable for ``qpos`` when the model has quaternion free joints.

    Used internally by :func:`mjx_byof`.  When a MuJoCo model has a
    floating-base free joint, ``nq > nv`` because each quaternion orientation
    contributes 4 position DOF but only 3 angular velocity DOF. The simple
    symbolic shorthand ``"qpos": qvel`` therefore fails a shape check. This
    function returns a BYOF dynamics callable that correctly computes ``qdot``
    from ``(q, qd)`` by applying the quaternion kinematic equation::

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
        Quaternions follow MuJoCo's scalar-first ``[qw, qx, qy, qz]`` layout and
        describe the **body** frame **relative to the world** frame.
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

    Note:
        `MjxDynamics` is the preferred entry-point — it goes directly in the
        ``dynamics=`` slot of `Problem` and constructs the matching
        State/Control objects for you. Use `mjx_byof` only when you need to
        supply custom State/Control objects (e.g. interleave them with other
        states) or otherwise want full control over names.

    It inspects the model's ``nq`` and ``nv`` to detect free joints and
    automatically includes the quaternion kinematics callable for ``qpos``
    when needed.

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
        result["qpos"] = _free_joint_qpos_dynamics(
            qpos=qpos,
            qvel=qvel,
            n_free_joints=n_free,
        )

    return result


class MjxDynamics(DynamicsAdapter):
    """First-class MJX dynamics adapter for `Problem`.

    Wraps a ``mujoco.mjx`` model so it can be passed directly to the
    ``dynamics=`` argument of `Problem`. The adapter
    constructs default ``qpos`` / ``qvel`` State objects and a ``ctrl``
    Control matching the model's ``nq`` / ``nv`` / ``nu``, exposes them via
    ``.states`` / ``.controls``, and routes the MJX forward dynamics through
    the BYOF channel internally — without requiring the user to know about
    BYOF at all.

    Example:
        Cartpole (``nq == nv``)::

            mj_model = mujoco.MjModel.from_xml_path("cartpole.xml")
            mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            mjx_model = mjx.put_model(mj_model)

            dyn = ox.MjxDynamics(mjx_model)
            problem = ox.Problem(
                dynamics=dyn,
                states=dyn.states,
                controls=dyn.controls,
                ...
            )

        Quadrotor with a free joint (``nq > nv``) — quaternion kinematics
        are inserted automatically::

            dyn = ox.MjxDynamics(mjx_model)  # nq=7, nv=6
            problem = ox.Problem(
                dynamics=dyn, states=dyn.states, controls=dyn.controls, ...
            )

    Custom State/Control names or shapes are *not* supported here on purpose
    — the whole point of the adapter is "I don't want to think about names."
    Drop to the lower-level `mjx_byof` helper if you need that control.
    """

    def __init__(
        self,
        mjx_model: Any,
        *,
        return_component: str = "qacc",
        extra_postprocess: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.state import State

        self.mjx_model = mjx_model
        self.return_component = return_component
        self.extra_postprocess = extra_postprocess

        nq = int(mjx_model.nq)
        nv = int(mjx_model.nv)
        nu = int(mjx_model.nu)

        self._qpos = State("qpos", shape=(nq,))
        self._qvel = State("qvel", shape=(nv,))
        self._ctrl = Control("ctrl", shape=(nu,))

        self.states: list[State] = [self._qpos, self._qvel]
        self.controls: list[Control] = [self._ctrl]

    def expand(self) -> Tuple[dict, dict]:
        """Return ``(dynamics_dict, byof_dict)`` for this MJX model.

        - ``nq == nv``: ``dynamics_dict = {"qpos": qvel}`` (symbolic
          kinematic identity), ``byof_dict["dynamics"] = {"qvel": ...}``.
        - ``nq > nv``: ``dynamics_dict = {}``, ``byof_dict["dynamics"]``
          contains both ``"qpos"`` (quaternion kinematics) and ``"qvel"``.
        """
        nq = int(self.mjx_model.nq)
        nv = int(self.mjx_model.nv)

        byof_dynamics: dict = {
            "qvel": mjx_dynamics(
                self.mjx_model,
                qpos=self._qpos,
                qvel=self._qvel,
                ctrl=self._ctrl,
                return_component=self.return_component,
                extra_postprocess=self.extra_postprocess,
            ),
        }

        n_free = nq - nv
        if n_free > 0:
            byof_dynamics["qpos"] = _free_joint_qpos_dynamics(
                qpos=self._qpos,
                qvel=self._qvel,
                n_free_joints=n_free,
            )
            dynamics_dict: dict = {}
        else:
            dynamics_dict = {"qpos": self._qvel}

        return dynamics_dict, {"dynamics": byof_dynamics}
