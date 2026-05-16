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

The lower-level `mjx_dynamics` callable factory is also public for advanced
users who need to assemble their own BYOF dynamics dict (e.g. with custom
State/Control names or interleaved with other states).

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

    Used internally by `MjxDynamics`.  When a MuJoCo model has a
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


# MuJoCo joint type enum (matches mujoco.mjtJoint): 0=free, 1=ball, 2=slide, 3=hinge.
# Inlined here so we don't require `mujoco` to be importable just for type validation —
# the user already needs mujoco to have constructed the mjx_model, but keeping the
# numeric constants local makes this file self-contained.
_MJ_JNT_FREE = 0
_MJ_JNT_BALL = 1
_MJ_JNT_SLIDE = 2
_MJ_JNT_HINGE = 3


def _initial_bounds_from_model(
    mjx_model: Any, nq: int, nv: int, nu: int
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Pull qpos / ctrl bounds out of the MJX model.

    Returns ``(qpos_min, qpos_max, qvel_min, qvel_max, ctrl_min, ctrl_max)``.

    - ``qpos`` bounds come from ``mjx_model.jnt_range`` for slide/hinge joints
      flagged ``jnt_limited=True``. All other qpos slots — free-joint
      translations and quaternion components, unlimited slide/hinge joints —
      get ``±inf``.
    - ``ctrl`` bounds come from ``mjx_model.actuator_ctrlrange`` for actuators
      flagged ``actuator_ctrllimited=True``. Unlimited actuators get ``±inf``.
    - ``qvel`` bounds are always ``±inf`` because MuJoCo has no per-joint
      velocity-limit concept; users override as needed.
    """
    import numpy as _np

    qpos_min = _np.full(nq, -_np.inf)
    qpos_max = _np.full(nq, _np.inf)
    qvel_min = _np.full(nv, -_np.inf)
    qvel_max = _np.full(nv, _np.inf)
    ctrl_min = _np.full(nu, -_np.inf)
    ctrl_max = _np.full(nu, _np.inf)

    # Per-joint qpos bounds — only slide/hinge can be range-limited; free
    # joints always have jnt_limited=False so we skip them safely.
    jnt_type = _np.asarray(mjx_model.jnt_type).astype(int)
    jnt_qposadr = _np.asarray(mjx_model.jnt_qposadr).astype(int)
    jnt_limited = _np.asarray(mjx_model.jnt_limited).astype(bool)
    jnt_range = _np.asarray(mjx_model.jnt_range).astype(float)
    for i, jtype in enumerate(jnt_type):
        if jtype in (_MJ_JNT_SLIDE, _MJ_JNT_HINGE) and jnt_limited[i]:
            adr = int(jnt_qposadr[i])
            qpos_min[adr] = jnt_range[i, 0]
            qpos_max[adr] = jnt_range[i, 1]

    if nu > 0:
        act_limited = _np.asarray(mjx_model.actuator_ctrllimited).astype(bool)
        act_range = _np.asarray(mjx_model.actuator_ctrlrange).astype(float)
        for i in range(nu):
            if act_limited[i]:
                ctrl_min[i] = act_range[i, 0]
                ctrl_max[i] = act_range[i, 1]

    return qpos_min, qpos_max, qvel_min, qvel_max, ctrl_min, ctrl_max


def _validate_supported_joints(mjx_model: Any) -> None:
    """Refuse models whose joint layout the adapter cannot correctly handle.

    `MjxDynamics` only supports models composed of free / slide / hinge joints
    where all free joints precede the others in the state layout. Anything else
    (ball joints, custom joint orderings) silently breaks the
    `_free_joint_qpos_dynamics` arithmetic, so we refuse with a clear error
    rather than producing wrong dynamics.
    """
    import numpy as _np

    jnt_type = _np.asarray(mjx_model.jnt_type).astype(int)
    supported = {_MJ_JNT_FREE, _MJ_JNT_SLIDE, _MJ_JNT_HINGE}
    bad = sorted(set(jnt_type.tolist()) - supported)
    if bad:
        if _MJ_JNT_BALL in bad:
            raise NotImplementedError(
                "MjxDynamics does not support ball joints (mjJNT_BALL): they "
                "share nq=4, nv=3 with free joints but use different "
                "kinematics, and the current quaternion-kinematics callable "
                "would silently produce wrong dynamics. Use `mjx_dynamics` "
                "directly and assemble byof['dynamics'] manually."
            )
        raise NotImplementedError(
            f"MjxDynamics only supports free, slide, and hinge joints; "
            f"model contains unsupported joint types {bad}. Use "
            "`mjx_dynamics` directly and assemble byof['dynamics'] manually."
        )

    # _free_joint_qpos_dynamics assumes all free joints come first in the
    # state vector. If a slide/hinge precedes a free joint, the quaternion
    # offsets would be off.
    free_mask = jnt_type == _MJ_JNT_FREE
    n_free = int(free_mask.sum())
    if n_free and not free_mask[:n_free].all():
        raise NotImplementedError(
            "MjxDynamics requires all free joints to come before any "
            "slide/hinge joints in the MuJoCo model. Reorder the joints in "
            "your MJCF/URDF, or use `mjx_dynamics` directly to assemble "
            "byof['dynamics'] yourself."
        )


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
    Drop to the lower-level `mjx_dynamics` helper if you need that control —
    construct your own State/Control objects, pass them in, and assemble the
    BYOF dict yourself.

    Supported joint structure:
        * Free (``mjJNT_FREE``), slide (``mjJNT_SLIDE``), and hinge
          (``mjJNT_HINGE``) joints only.
        * If the model contains any free joints, they must all come
          *before* any slide/hinge joints in the MuJoCo layout.
        * Ball joints (``mjJNT_BALL``) are explicitly refused — they share
          ``nq=4, nv=3`` with free joints but use different kinematics, and
          would silently produce wrong dynamics.

        Construction raises ``NotImplementedError`` if any of these
        conditions are violated; fall back to `mjx_dynamics` for those
        cases.

    Auto-populated bounds:
        * ``qpos.min`` / ``qpos.max`` are read from ``mjx_model.jnt_range``
          for slide / hinge joints flagged ``jnt_limited=True``; free-joint
          slots and unlimited joints default to ``±inf``.
        * ``ctrl.min`` / ``ctrl.max`` are read from ``actuator_ctrlrange``
          for actuators flagged ``actuator_ctrllimited=True``; otherwise
          ``±inf``.
        * ``qvel`` bounds default to ``±inf`` (MuJoCo has no per-joint
          velocity-limit concept).

        Override any of these after construction if you want tighter
        problem-specific bounds.
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

        _validate_supported_joints(mjx_model)

        self.mjx_model = mjx_model
        self.return_component = return_component
        self.extra_postprocess = extra_postprocess

        nq = int(mjx_model.nq)
        nv = int(mjx_model.nv)
        nu = int(mjx_model.nu)

        self._qpos = State("qpos", shape=(nq,))
        self._qvel = State("qvel", shape=(nv,))
        self._ctrl = Control("ctrl", shape=(nu,))

        # Auto-populate bounds from the model so the user doesn't have to
        # re-type joint / actuator limits already declared in MJCF. Users
        # can still override any of these after construction.
        qpos_min, qpos_max, qvel_min, qvel_max, ctrl_min, ctrl_max = _initial_bounds_from_model(
            mjx_model, nq, nv, nu
        )
        self._qpos.min = qpos_min
        self._qpos.max = qpos_max
        self._qvel.min = qvel_min
        self._qvel.max = qvel_max
        self._ctrl.min = ctrl_min
        self._ctrl.max = ctrl_max

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
