"""JAX visitors for STL (Signal Temporal Logic) expressions.

Visitors: Or
"""

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.stl import Or
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Or)
def _visit_or(lowerer, node: Or):
    """Lower STL disjunction (Or) to JAX using STLJax library.

    Converts a symbolic Or constraint to an STLJax Or formula for handling
    disjunctive task specifications. Each predicate becomes an STLJax predicate.

    Args:
        node: Or expression node with predicates (Constraint or STLExpr)

    Returns:
        Function (x, u, node, params) -> STL robustness value

    Note:
        Uses STLJax library for signal temporal logic evaluation. The returned
        function computes the robustness metric for the disjunction, which is
        positive when at least one predicate is satisfied.

        Robustness extraction:
        - For Constraint (lhs <= rhs): robustness = rhs - lhs
        - For STLExpr: recursively lower the STL expression

    Example:
        Used for task specifications like "reach goal A OR goal B"::

            goal_A = ox.Norm(x - target_A) <= 1.0
            goal_B = ox.Norm(x - target_B) <= 1.0
            task = ox.stl.Or(goal_A, goal_B)

    See Also:
        - stljax.formula.Or: Underlying STLJax implementation
        - STL robustness: Quantitative measure of constraint satisfaction
    """
    from stljax.formula import Or as STLOr
    from stljax.formula import Predicate

    from openscvx.symbolic.expr.arithmetic import Sub
    from openscvx.symbolic.expr.constraint import Constraint
    from openscvx.symbolic.expr.stl import STLExpr

    # Extract robustness expressions from predicates and lower them
    robustness_fns = []
    for pred in node.predicates:
        if isinstance(pred, Constraint):
            # For Constraint (lhs <= rhs): robustness = rhs - lhs
            # Positive when satisfied (lhs <= rhs means rhs - lhs >= 0)
            robustness_expr = Sub(pred.rhs, pred.lhs)
            robustness_fns.append(lowerer.lower(robustness_expr))
        elif isinstance(pred, STLExpr):
            # For nested STL expressions, lower them directly
            # They already return robustness values
            robustness_fns.append(lowerer.lower(pred))
        else:
            raise TypeError(f"Unexpected predicate type: {type(pred)}")

    # Return a function that evaluates the STLJax Or
    def or_fn(x, u, node, params):
        # Create STLJax predicates for each robustness function
        predicates = []
        for i, robustness_fn in enumerate(robustness_fns):
            # Create a predicate function that captures the current params
            def make_pred_fn(fn):
                return lambda x: fn(x, None, None, params)

            pred_fn = make_pred_fn(robustness_fn)
            predicates.append(Predicate(f"pred_{i}", pred_fn))

        # Create and evaluate STLJax Or formula
        stl_or = STLOr(*predicates)
        return stl_or(x)

    return or_fn
