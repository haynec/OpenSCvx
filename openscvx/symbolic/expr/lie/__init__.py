"""Lie algebra operations for rigid body dynamics.

This module provides symbolic expression nodes for Lie algebra operations
commonly used in 6-DOF rigid body dynamics, robotics, and geometric mechanics.
These operations enable elegant formulations of Newton-Euler dynamics using
spatial vectors (twists and wrenches).

The module provides two tiers of functionality:

**Built-in operators** work out of the box and include adjoint/coadjoint
operators for dynamics (_e.g._ ``Adjoint``, ``AdjointDual``) and frame transformations
(_e.g._ ``SE3Adjoint``, ``SE3AdjointDual``).

**jaxlie-backed operators** require ``pip install openscvx[lie]`` and provide
exponential/logarithm maps for SO(3) and SE(3) groups (_e.g._ ``SO3Exp``, ``SO3Log``,
``SE3Exp``, ``SE3Log``).

Conventions:
    - Twist (spatial velocity): ξ = [v; ω] where v ∈ ℝ³ is linear velocity
      and ω ∈ ℝ³ is angular velocity (both in body frame)
    - Wrench (spatial force): F = [f; τ] where f ∈ ℝ³ is force and τ ∈ ℝ³
      is torque (both in body frame)

Note:
    The twist convention [v; ω] (linear first, angular second) matches jaxlie's
    SE3 tangent parameterization, so no reordering is needed during lowering.

Example:
    Newton-Euler dynamics for a rigid body using the coadjoint operator::

        import openscvx as ox

        twist = ox.State("twist", shape=(6,))
        M = ox.Parameter("M", shape=(6, 6), value=spatial_inertia)
        wrench = ox.Control("wrench", shape=(6,))

        momentum = M @ twist
        bias_force = ox.lie.AdjointDual(twist, momentum)
        twist_dot = M_inv @ (wrench - bias_force)

    Product of Exponentials forward kinematics (requires jaxlie)::

        screw_axis = ox.Constant(np.array([0, 0, 0, 0, 0, 1]))
        theta = ox.State("theta", shape=(1,))
        T_joint = ox.lie.SE3Exp(screw_axis * theta)  # 4×4 matrix

References:
    - Murray, Li, Sastry: "A Mathematical Introduction to Robotic Manipulation"
    - Featherstone: "Rigid Body Dynamics Algorithms"
    - Sola et al.: "A micro Lie theory for state estimation in robotics"
"""

# Core operators - no dependencies
from .adjoint import Adjoint, AdjointDual, SE3Adjoint, SE3AdjointDual

# jaxlie-backed operators - optional dependency.
# These AST nodes don't import jaxlie themselves (lowering does), so we probe
# for jaxlie explicitly to gate construction with a clear install hint instead
# of letting users build nodes that blow up later inside the lowerer.
try:
    import jaxlie  # noqa: F401

    from .se3 import SE3Exp, SE3Log
    from .so3 import SO3Exp, SO3Log

    _JAXLIE_AVAILABLE = True
except ImportError:
    _JAXLIE_AVAILABLE = False

    def _make_stub(name: str):
        """Create a stub class that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(f"{name} requires jaxlie. Install with: pip install openscvx[lie]")

        return type(name, (), {"__init__": __init__})

    SO3Exp = _make_stub("SO3Exp")
    SO3Log = _make_stub("SO3Log")
    SE3Exp = _make_stub("SE3Exp")
    SE3Log = _make_stub("SE3Log")

__all__ = [
    "AdjointDual",
    "Adjoint",
    "SE3Adjoint",
    "SE3AdjointDual",
    "SO3Exp",
    "SO3Log",
    "SE3Exp",
    "SE3Log",
]
