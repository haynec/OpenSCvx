"""frax dynamics adapters for OpenSCvx.

The recommended entry-point is `FraxDynamics`, a `DynamicsAdapter` that goes
directly into the ``dynamics=`` slot of `Problem` and exposes the synthesized
State/Control objects on ``.states`` / ``.controls``::

    from frax.robots.franka_panda import load_panda
    import openscvx as ox

    dyn = ox.FraxDynamics(load_panda())
    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        ...
    )

Joint limits and torque bounds are read automatically from the robot's URDF
and set on the State/Control objects. Override them after construction if
needed::

    q, qd = dyn.states
    q.initial = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])

Because `frax.Robot` is registered as a JAX static pytree
(``@jax.tree_util.register_static``), it is captured safely inside the BYOF
callable without special handling.

The lower-level `frax_dynamics` factory is also public for advanced users who
need to assemble their own BYOF dynamics dict::

    from openscvx.integrations import frax_dynamics

    q   = ox.State("q",   shape=(robot.num_joints,))
    qd  = ox.State("qd",  shape=(robot.num_joints,))
    tau = ox.Control("tau", shape=(robot.num_actuated_joints,))

    problem = ox.Problem(
        dynamics={"q": qd},
        byof={"dynamics": {"qd": frax_dynamics(robot, q=q, qd=qd, tau=tau)}},
        states=[q, qd],
        controls=[tau],
        ...
    )

Note:
    frax uses Euler-angle / rotation-matrix representation for all joint DOFs,
    including floating-base joints. There is no ``nq != nv`` split, so
    ``dynamics_dict`` is always ``{"q": qd}`` regardless of robot type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np
import jax.numpy as jnp

from openscvx.integrations._utils import _resolve_slice
from openscvx.integrations.base import DynamicsAdapter

if TYPE_CHECKING:
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


# frax encodes an "unlimited" joint DOF as ±1e6 (not 0.0 and not inf). This
# is emitted by frax's URDF / floating-base construction — e.g. every entry of
# a 6-DOF floating base's position / velocity / effort limits is ±1e6. We map
# entries at or beyond this magnitude to ±inf so that unbounded DOFs become
# genuinely unbounded box constraints (matching the MJX adapter's convention)
# rather than leaking a hard ±1e6 box into the optimization problem.
_FRAX_UNLIMITED = 1e6


def _desentinel(values: np.ndarray) -> np.ndarray:
    """Map frax's ±1e6 "no limit" sentinel to signed ``±inf``.

    Entries whose magnitude is at least ``_FRAX_UNLIMITED`` are replaced with
    ``+inf`` (if positive) or ``-inf`` (if negative); all other entries are
    left untouched.
    """
    return np.where(np.abs(values) >= _FRAX_UNLIMITED, np.sign(values) * np.inf, values)


def _bounds_from_robot(robot) -> tuple:
    """Extract State/Control bounds from a ``frax.Robot``.

    Reads ``joint_lower_limits``, ``joint_upper_limits``,
    ``joint_max_velocities``, and ``joint_max_forces`` from the robot. frax
    encodes "no limit" as ``±1e6`` (e.g. for floating-base DOFs); those entries
    are mapped to ``±inf`` via :func:`_desentinel`.

    For floating-base robots, the first 6 entries of ``joint_max_forces``
    correspond to the unactuated floating-base DOFs; only the remaining
    ``num_actuated_joints`` entries form the ``tau`` bounds.

    Args:
        robot: A ``frax.Robot`` (or subclass) instance.

    Returns:
        Tuple ``(q_min, q_max, qd_min, qd_max, tau_min, tau_max)``, each a
        NumPy array of the appropriate length.
    """
    nj = robot.num_joints
    na = robot.num_actuated_joints

    q_min = _desentinel(np.asarray(robot.joint_lower_limits, dtype=float))
    q_max = _desentinel(np.asarray(robot.joint_upper_limits, dtype=float))

    vel = _desentinel(np.asarray(robot.joint_max_velocities, dtype=float))

    forces = _desentinel(np.asarray(robot.joint_max_forces, dtype=float))
    # For floating-base robots the first 6 entries are the unactuated base;
    # take the last na entries as the actuated-joint torque limits.
    tau_max = forces[nj - na :]

    return q_min, q_max, -vel, vel, -tau_max, tau_max


def frax_dynamics(
    robot,
    *,
    q: "State | slice",
    qd: "State | slice",
    tau: "Control | slice",
) -> Callable:
    """Wrap a ``frax.Robot`` as a BYOF dynamics function.

    Returns a callable ``f(x, u, node, params) -> qdd`` (shape
    ``(robot.num_joints,)``) suitable for ``byof["dynamics"]["qd"]``.

    For floating-base robots (``robot.includes_floating_dof = True``), the
    ``tau`` control covers only the ``num_actuated_joints`` actuated DOFs.
    Internally, the callable prepends six zeros to form the full
    ``(num_joints,)`` torque vector before calling
    ``robot.forward_dynamics``. The floating-base DOFs are assumed to appear
    first in the joint ordering (standard robotics convention, verified for
    ``frax.load_g1``).

    ``robot`` is captured in the closure safely because ``frax.Robot`` is
    decorated with ``@jax.tree_util.register_static``, making it a hashable
    JAX static type.

    Args:
        robot: A ``frax.Robot`` (or subclass) instance.
        q: Position state (or slice into the unified ``x`` vector).
            Length must equal ``robot.num_joints``.
        qd: Velocity state (or slice). Length must equal ``robot.num_joints``.
        tau: Control variable (or slice into the unified ``u`` vector).
            Length must equal ``robot.num_actuated_joints``.

    Returns:
        A function ``f(x, u, node, params) -> jnp.ndarray`` matching the BYOF
        dynamics signature. The output is ``qdd``, the joint acceleration
        vector of shape ``(robot.num_joints,)``.

    Example:
        Franka Panda joint-space dynamics::

            from frax.robots.franka_panda import load_panda
            import openscvx as ox
            from openscvx.integrations import frax_dynamics

            robot = load_panda()
            q   = ox.State("q",   shape=(robot.num_joints,))
            qd  = ox.State("qd",  shape=(robot.num_joints,))
            tau = ox.Control("tau", shape=(robot.num_actuated_joints,))

            qdd = frax_dynamics(robot, q=q, qd=qd, tau=tau)

            problem = ox.Problem(
                dynamics={"q": qd},
                byof={"dynamics": {"qd": qdd}},
                states=[q, qd],
                controls=[tau],
                ...
            )
    """
    _q_arg = q
    _qd_arg = qd
    _tau_arg = tau
    # Populated lazily on first call so the factory can be called before
    # Problem preprocessing assigns .slice to the State/Control objects.
    _resolved: list = []

    def f(x, u, node, params):
        del node, params
        if not _resolved:
            _resolved.append(_resolve_slice(_q_arg, "q"))
            _resolved.append(_resolve_slice(_qd_arg, "qd"))
            _resolved.append(_resolve_slice(_tau_arg, "tau"))
        q_sl, qd_sl, tau_sl = _resolved

        q_val = x[q_sl]
        qd_val = x[qd_sl]
        tau_val = u[tau_sl]

        if robot.includes_floating_dof:
            # First 6 DOF are the unactuated floating base; user ctrl covers
            # only the remaining num_actuated_joints.
            tau_full = jnp.concatenate([jnp.zeros(6, dtype=tau_val.dtype), tau_val])
        else:
            tau_full = tau_val

        # frax's forward_dynamics declares fext with no default; pass it
        # explicitly.
        return robot.forward_dynamics(q_val, qd_val, tau_full, fext=None)

    return f


class FraxDynamics(DynamicsAdapter):
    """First-class frax dynamics adapter for `Problem`.

    Wraps a ``frax.Robot`` so it can be passed directly to the ``dynamics=``
    argument of `Problem`. The adapter constructs default ``q`` / ``qd`` State
    objects and a ``tau`` Control matching the robot's joint count, exposes
    them via ``.states`` / ``.controls``, and routes the frax forward dynamics
    through the BYOF channel internally — without requiring the user to know
    about BYOF.

    Joint limits and torque bounds are read from the robot's URDF and
    auto-populated on the State/Control objects. Override any of them after
    construction if you want tighter problem-specific bounds::

        dyn = ox.FraxDynamics(load_panda())
        q, qd = dyn.states
        q.initial = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
        # tighten velocity limits for a slow approach
        qd.max = np.full(7, 1.0)

    Example:
        Franka Panda::

            from frax.robots.franka_panda import load_panda
            import openscvx as ox

            dyn = ox.FraxDynamics(load_panda())
            problem = ox.Problem(
                dynamics=dyn,
                states=dyn.states,
                controls=dyn.controls,
                constraints=[...],
                N=50,
                time=ox.Time(...),
            )

    Floating-base robots (``robot.includes_floating_dof = True``) are
    supported. The ``tau`` Control covers only the ``num_actuated_joints``
    actuated DOFs; the adapter inserts the required zero-torque entries for the
    unactuated floating-base DOFs internally.

    Custom State/Control names are *not* supported — drop to the lower-level
    `frax_dynamics` helper for that.

    Auto-populated bounds:
        * ``q.min`` / ``q.max``: from the URDF ``<limit lower/upper>``.
        * ``qd.min`` / ``qd.max``: from the URDF ``<limit velocity>``
          (mirrored as ``±max_vel``).
        * ``tau.min`` / ``tau.max``: from the URDF ``<limit effort>`` for the
          actuated joints.

        frax's ``±1e6`` "no limit" sentinel (used for floating-base DOFs) is
        mapped to ``±inf`` on all of the above.
    """

    def __init__(self, robot) -> None:
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.state import State

        self._robot = robot
        nj = robot.num_joints
        na = robot.num_actuated_joints

        self._q = State("q", shape=(nj,))
        self._qd = State("qd", shape=(nj,))
        self._tau = Control("tau", shape=(na,))

        q_min, q_max, qd_min, qd_max, tau_min, tau_max = _bounds_from_robot(robot)
        self._q.min = q_min
        self._q.max = q_max
        self._qd.min = qd_min
        self._qd.max = qd_max
        self._tau.min = tau_min
        self._tau.max = tau_max

        self.states: list[State] = [self._q, self._qd]
        self.controls: list[Control] = [self._tau]

    def expand(self) -> Tuple[dict, dict]:
        """Return ``(dynamics_dict, byof_dict)`` for this frax robot.

        frax always has ``nq == nv`` (no quaternion floating-base split), so
        ``dynamics_dict`` is always ``{"q": qd}`` and the full joint
        acceleration goes through BYOF.
        """
        dynamics_dict = {"q": self._qd}
        byof_dict = {
            "dynamics": {
                "qd": frax_dynamics(
                    self._robot,
                    q=self._q,
                    qd=self._qd,
                    tau=self._tau,
                )
            }
        }
        return dynamics_dict, byof_dict
