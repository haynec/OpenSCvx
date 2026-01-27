"""JAX visitors for logic expressions.

Visitors: All, Any, Cond
"""

import jax.numpy as jnp
from jax.lax import cond

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.logic import All, Any, Cond
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(All)
def _visit_all(lowerer, node: All):
    """Lower All expression to JAX function using jnp.all.

    Evaluates all predicates and combines them with AND semantics. Returns
    True only if ALL predicates are satisfied.

    Args:
        node: All expression node with predicates

    Returns:
        Function (x, u, node, params) -> scalar boolean
    """
    pred_fns = [lowerer.lower(p) for p in node.predicates]

    def all_fn(x, u, node_arg, params):
        # Evaluate each predicate and check if satisfied (residual <= 0)
        results = []
        for pred_fn in pred_fns:
            pred_val = pred_fn(x, u, node_arg, params)
            # For vector predicates, jnp.all reduces to scalar
            results.append(jnp.all(pred_val <= 0))
        # Combine all results with AND
        return jnp.all(jnp.array(results))

    return all_fn


@visitor(Any)
def _visit_any(lowerer, node: Any):
    """Lower Any expression to JAX function using jnp.any.

    Evaluates all predicates and combines them with OR semantics. Returns
    True if ANY predicate is satisfied.

    Args:
        node: Any expression node with predicates

    Returns:
        Function (x, u, node, params) -> scalar boolean
    """
    pred_fns = [lowerer.lower(p) for p in node.predicates]

    def any_fn(x, u, node_arg, params):
        # Evaluate each predicate and check if satisfied (residual <= 0)
        results = []
        for pred_fn in pred_fns:
            pred_val = pred_fn(x, u, node_arg, params)
            # For vector predicates, jnp.any checks if any element satisfies
            results.append(jnp.any(pred_val <= 0))
        # Combine all results with OR
        return jnp.any(jnp.array(results))

    return any_fn


@visitor(Cond)
def _visit_cond(lowerer, node: Cond):
    """Lower conditional expression to JAX function using jax.lax.cond.

    Implements JAX-traceable conditional logic by wrapping jax.lax.cond. The
    predicate is evaluated first, and then either the true or false branch
    is evaluated based on the predicate result.

    If node_ranges is specified, the conditional is only active within those
    ranges. Outside the ranges, the false branch is always evaluated.

    Args:
        node: Cond expression node with predicate, true_branch, false_branch,
            and optional node_ranges

    Returns:
        Function (x, u, node, params) -> result from selected branch

    Note:
        Uses jax.lax.cond for JAX-traceable conditional evaluation. The predicate
        can be an Inequality, All, or Any expression. Both branches are lowered
        and the appropriate one is selected at runtime based on the predicate value.

    Example:
        Conditional dynamics based on state::

            x = ox.State("x", shape=(3,))
            expr = ox.Cond(
                ox.Norm(x) >= 1.0,  # predicate
                x / ox.Norm(x),     # true branch
                x                   # false branch
            )

        Using All for explicit AND semantics::

            expr = ox.Cond(
                ox.All([x >= 0.0, x <= 10.0]),  # all must be satisfied
                1.0,                             # in range
                0.0                              # out of range
            )

        Using Any for OR semantics::

            expr = ox.Cond(
                ox.Any([in_region_a, in_region_b]),  # any must be satisfied
                region_value,
                default_value
            )
    """
    # Lower predicate (if any) and branches recursively
    pred_fn = lowerer.lower(node.predicate) if node.predicate is not None else None
    true_fn = lowerer.lower(node.true_branch)
    false_fn = lowerer.lower(node.false_branch)

    # Check if predicate is All/Any (returns boolean) or Inequality (returns residual)
    is_boolean_pred = isinstance(node.predicate, (All, Any))

    # Capture node_ranges for use in closure
    node_ranges = node.node_ranges

    def cond_fn(x, u, node_arg, params):
        # Start with predicate evaluation (or True if no predicate)
        if pred_fn is None:
            pred_bool = jnp.array(True)
        else:
            pred_val = pred_fn(x, u, node_arg, params)
            # Convert predicate result to boolean
            if is_boolean_pred:
                # All/Any already return boolean
                pred_bool = pred_val
            else:
                # Inequality returns residual: satisfied when residual <= 0
                pred_bool = jnp.squeeze(pred_val) <= 0

        # If node_ranges is specified, check if current node is in range
        if node_ranges is not None:
            # Extract scalar from node_arg (which may be array or scalar from vmap)
            node_scalar = jnp.atleast_1d(node_arg)[0]
            # Check if node_scalar is within any of the specified ranges [start, end)
            in_range = jnp.array(False)
            for start, end in node_ranges:
                in_range = in_range | ((node_scalar >= start) & (node_scalar < end))
            # Combined predicate: must be in range AND predicate satisfied
            pred_bool = in_range & pred_bool

        # Use jax.lax.cond for conditional evaluation
        return cond(
            pred_bool,
            lambda _: true_fn(x, u, node_arg, params),
            lambda _: false_fn(x, u, node_arg, params),
            operand=None,
        )

    return cond_fn
