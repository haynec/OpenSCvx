"""CVXPy visitors for constraint expressions.

Visitors: Equality, Inequality, CrossNodeConstraint, CTCS
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.constraint import (
    CTCS,
    CrossNodeConstraint,
    Equality,
    Inequality,
)
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Equality)
def _visit_equality(lowerer, node: Equality) -> cp.Constraint:
    """Lower equality constraint to CVXPy constraint (lhs == rhs).

    Equality constraints require affine expressions on both sides for
    DCP compliance.

    Args:
        node: Equality constraint node

    Returns:
        CVXPy equality constraint object

    Note:
        For DCP compliance, both lhs and rhs must be affine. CVXPy will
        raise a DCP error if either side is non-affine.
    """
    left = lowerer.lower(node.lhs)
    right = lowerer.lower(node.rhs)
    return left == right


@visitor(Inequality)
def _visit_inequality(lowerer, node: Inequality) -> cp.Constraint:
    """Lower inequality constraint to CVXPy constraint (lhs <= rhs).

    Inequality constraints must satisfy DCP rules: convex <= concave.

    Args:
        node: Inequality constraint node

    Returns:
        CVXPy inequality constraint object

    Note:
        For DCP compliance: lhs must be convex and rhs must be concave.
        Common form: convex_expr(x) <= constant
    """
    left = lowerer.lower(node.lhs)
    right = lowerer.lower(node.rhs)
    return left <= right


@visitor(CrossNodeConstraint)
def _visit_cross_node_constraint(lowerer, node: CrossNodeConstraint) -> cp.Constraint:
    """Lower CrossNodeConstraint to CVXPy constraint.

    CrossNodeConstraint wraps constraints that reference multiple trajectory
    nodes via NodeReference (e.g., rate limits like x.at(k) - x.at(k-1) <= r).

    For CVXPy lowering, this simply lowers the inner constraint. The NodeReference
    nodes within the constraint will handle extracting values from the full
    trajectory arrays (which must be provided in variable_map as "x" and "u").

    Args:
        node: CrossNodeConstraint expression wrapping the inner constraint

    Returns:
        CVXPy constraint object

    Note:
        The variable_map must contain full trajectory arrays:
            - "x": (N, n_x) CVXPy expression (e.g., cp.vstack(x_nonscaled))
            - "u": (N, n_u) CVXPy expression (e.g., cp.vstack(u_nonscaled))

        NodeReference visitors will index into these arrays using the fixed
        node indices baked into the expression.

    Example:
        For constraint: position.at(5) - position.at(4) <= max_step

        With variable_map = {"x": cp.vstack([x[k] for k in range(N)])}

        The lowered constraint evaluates:
            x[5, pos_slice] - x[4, pos_slice] <= max_step
    """
    # Simply lower the inner constraint - NodeReference handles indexing
    return lowerer.lower(node.constraint)


@visitor(CTCS)
def _visit_ctcs(lowerer, node: CTCS) -> cp.Expression:
    """Raise NotImplementedError for CTCS constraints.

    CTCS (Continuous-Time Constraint Satisfaction) constraints are handled
    through dynamics augmentation using JAX, not CVXPy. They represent
    non-convex continuous-time constraints.

    Args:
        node: CTCS constraint node

    Raises:
        NotImplementedError: Always raised since CTCS uses JAX, not CVXPy

    Note:
        CTCS constraints are lowered to JAX during dynamics augmentation.
        They add virtual states and controls to enforce constraints over
        continuous time intervals. See JaxLowerer.visit_ctcs() instead.
    """
    raise NotImplementedError(
        "CTCS constraints are for continuous-time constraint satisfaction and "
        "should be handled through dynamics augmentation with JAX lowering, "
        "not CVXPy lowering. CTCS constraints represent non-convex dynamics "
        "augmentation."
    )
