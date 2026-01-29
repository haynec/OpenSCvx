"""JAX visitors for constraint expressions.

Visitors: Equality, Inequality, NodalConstraint, CrossNodeConstraint, CTCS
"""

import jax.numpy as jnp
from jax.lax import cond

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.constraint import (
    CTCS,
    Constraint,
    CrossNodeConstraint,
    Equality,
    Inequality,
    NodalConstraint,
)
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Equality)
@visitor(Inequality)
def _visit_constraint(lowerer, node: Constraint):
    """Lower constraint to residual function.

    Both equality (lhs == rhs) and inequality (lhs <= rhs) constraints are
    lowered to their residual form: lhs - rhs. The constraint is satisfied
    when the residual equals zero (equality) or is non-positive (inequality).

    Args:
        node: Equality or Inequality constraint node

    Returns:
        Function (x, u, node, params) -> lhs - rhs (constraint residual)

    Note:
        The returned residual is used in penalty methods and Lagrangian terms.
        For equality: residual should be 0
        For inequality: residual should be <= 0
    """
    fL = lowerer.lower(node.lhs)
    fR = lowerer.lower(node.rhs)
    return lambda x, u, node, params: fL(x, u, node, params) - fR(x, u, node, params)


@visitor(NodalConstraint)
def _visit_nodal_constraint(lowerer, node: NodalConstraint):
    """Lower a NodalConstraint by lowering its underlying constraint.

    NodalConstraint is a wrapper that specifies which nodes a constraint
    applies to. The lowering just unwraps and lowers the inner constraint.

    Args:
        node: NodalConstraint wrapper

    Returns:
        Function from lowering the wrapped constraint expression
    """
    return lowerer.lower(node.constraint)


@visitor(CrossNodeConstraint)
def _visit_cross_node_constraint(lowerer, node: CrossNodeConstraint):
    """Lower CrossNodeConstraint to trajectory-level function.

    CrossNodeConstraint wraps constraints that reference multiple trajectory
    nodes via NodeReference (e.g., rate limits like x.at(k) - x.at(k-1) <= r).

    Unlike regular nodal constraints which have signature (x, u, node, params)
    and are vmapped across nodes, cross-node constraints operate on full
    trajectory arrays and return a scalar residual.

    Args:
        node: CrossNodeConstraint expression wrapping the inner constraint

    Returns:
        Function with signature (X, U, params) -> scalar residual
            - X: Full state trajectory, shape (N, n_x)
            - U: Full control trajectory, shape (N, n_u)
            - params: Dictionary of problem parameters
            - Returns: Scalar constraint residual (g <= 0 convention)

    Note:
        The inner constraint is lowered first (producing a function with the
        standard (x, u, node, params) signature), then wrapped to provide the
        trajectory-level (X, U, params) signature. The `node` parameter is
        unused since NodeReference nodes have fixed indices baked in.

    Example:
        For constraint: position.at(5) - position.at(4) <= max_step

        The lowered function evaluates:
            X[5, pos_slice] - X[4, pos_slice] - max_step

        And returns a scalar residual.
    """
    # Lower the inner constraint expression
    inner_fn = lowerer.lower(node.constraint)

    # Wrap to provide trajectory-level signature
    # The `node` parameter is unused for cross-node constraints since
    # NodeReference nodes have fixed indices baked in at construction time
    def trajectory_constraint(X, U, params):
        return inner_fn(X, U, 0, params)

    return trajectory_constraint


# TODO: (norrisg) CTCS is playing 2 roles here: both as a constraint wrapper and as the penalty
# expression w/ conditional logic. Consider adding conditional logic as separate AST nodes.
# Then, CTCS remains a wrapper and we just wrap the penalty expression with the conditional
# logic when we lower it.
@visitor(CTCS)
def _visit_ctcs(lowerer, node: CTCS):
    """Lower CTCS (Continuous-Time Constraint Satisfaction) to JAX function.

    CTCS constraints use penalty methods to enforce constraints over continuous
    time intervals. The lowered function includes conditional logic to activate
    the penalty only within the specified node interval.

    Args:
        node: CTCS constraint node with penalty expression and optional node range

    Returns:
        Function (x, u, current_node, params) -> penalty value or 0

    Note:
        Uses jax.lax.cond for JAX-traceable conditional evaluation. The penalty
        is active only when current_node is in [start_node, end_node).
        If no node range is specified, the penalty is always active.

    See Also:
        - CTCS: The symbolic CTCS constraint class
        - penalty functions: PositivePart, Huber, SmoothReLU
    """
    # Lower the penalty expression (which includes the constraint residual)
    penalty_expr_fn = lowerer.lower(node.penalty_expr())

    def ctcs_fn(x, u, current_node, params):
        # Check if constraint is active at this node
        if node.nodes is not None:
            start_node, end_node = node.nodes
            # Extract scalar value from current_node (which may be array or scalar)
            # Keep as JAX array for tracing compatibility
            node_scalar = jnp.atleast_1d(current_node)[0]
            is_active = (start_node <= node_scalar) & (node_scalar < end_node)

            # Use jax.lax.cond for conditional evaluation
            return cond(
                is_active,
                lambda _: penalty_expr_fn(x, u, current_node, params),
                lambda _: 0.0,
                operand=None,
            )
        else:
            # Always active if no node range specified
            return penalty_expr_fn(x, u, current_node, params)

    return ctcs_fn
