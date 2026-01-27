"""JAX visitors for constraint expressions.

Visitors: Equality, Inequality, NodalConstraint, CrossNodeConstraint, CTCS
"""

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.constraint import (
    CTCS,
    Constraint,
    CrossNodeConstraint,
    Equality,
    Inequality,
    NodalConstraint,
)
from openscvx.symbolic.expr.logic import Cond
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


@visitor(CTCS)
def _visit_ctcs(lowerer, node: CTCS):
    """Lower CTCS (Continuous-Time Constraint Satisfaction) to JAX function.

    CTCS constraints use penalty methods to enforce constraints over continuous
    time intervals. When a node range is specified, the penalty expression is
    wrapped in a Cond node to activate it only within that interval.

    Args:
        node: CTCS constraint node with penalty expression and optional node range

    Returns:
        Function (x, u, current_node, params) -> penalty value or 0

    Note:
        The penalty is active only when current_node is in [start_node, end_node).
        If no node range is specified, the penalty is always active.
        Conditional node-range logic is delegated to the Cond AST node.

    See Also:
        - CTCS: The symbolic CTCS constraint class
        - Cond: Conditional AST node used for node-range gating
        - penalty functions: PositivePart, Huber, SmoothReLU
    """
    penalty_expr = node.penalty_expr()

    if node.nodes is not None:
        # Wrap penalty in a Cond node that gates on the node range
        gated_expr = Cond(None, penalty_expr, 0.0, node_ranges=[node.nodes])
        return lowerer.lower(gated_expr)
    else:
        return lowerer.lower(penalty_expr)
