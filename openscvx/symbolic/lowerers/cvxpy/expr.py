"""CVXPy visitors for core expression types.

Visitors: Constant, Parameter, NodeReference
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.expr import Constant, NodeReference, Parameter
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Constant)
def _visit_constant(lowerer, node: Constant) -> cp.Expression:
    """Lower a constant value to a CVXPy constant.

    Wraps the constant's numpy array value in a CVXPy Constant expression.

    Args:
        node: Constant expression node

    Returns:
        CVXPy constant expression wrapping the value
    """
    return cp.Constant(node.value)


@visitor(Parameter)
def _visit_parameter(lowerer, node: Parameter) -> cp.Expression:
    """Lower a parameter to a CVXPy expression.

    Parameters are looked up by name in the variable_map. They can be mapped
    to CVXPy Parameter objects (for efficient parameter sweeps) or constants.

    Args:
        node: Parameter expression node

    Returns:
        CVXPy expression from variable_map (Parameter or constant)

    Raises:
        ValueError: If parameter name is not found in variable_map

    Note:
        For parameter sweeps without recompilation, map to cp.Parameter.
        For fixed values, map to cp.Constant or numpy arrays.
    """
    param_name = node.name
    if param_name in lowerer.variable_map:
        return lowerer.variable_map[param_name]
    else:
        raise ValueError(
            f"Parameter '{param_name}' not found in variable_map. "
            f"Add it during CVXPy lowering or use cp.Parameter for parameter sweeps."
        )


@visitor(NodeReference)
def _visit_node_reference(lowerer, node: "NodeReference") -> cp.Expression:
    """Lower NodeReference - extract value at a specific trajectory node.

    NodeReference enables cross-node constraints by referencing state/control
    values at specific discrete time points. This requires the variable_map to
    contain full trajectory arrays (N, n_x) or (N, n_u) rather than single-node
    vectors.

    Args:
        node: NodeReference expression with base and node_idx

    Returns:
        CVXPy expression representing the variable at the specified node:
        x[node_idx, slice] or u[node_idx, slice]

    Raises:
        ValueError: If the required trajectory variable is not in variable_map
        ValueError: If the base variable has no slice assigned
        NotImplementedError: If the base is a compound expression

    Example:
        For cross-node constraint: position.at(5) - position.at(4) <= 0.1

        variable_map = {
            "x": cp.vstack([x_nonscaled[k] for k in range(N)]),  # (N, n_x)
        }
        # position.at(5) lowers to x[5, position._slice]

    Note:
        The node_idx is already resolved to an absolute integer index during
        expression construction, so negative indices are already handled.
    """
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State

    idx = node.node_idx

    if isinstance(node.base, State):
        if "x" not in lowerer.variable_map:
            raise ValueError(
                "State vector 'x' not found in variable_map. "
                "For cross-node constraints, 'x' must be the full trajectory (N, n_x)."
            )

        cvx_var = lowerer.variable_map["x"]  # Should be (N, n_x) for cross-node constraints

        # Apply slice if state has one assigned
        if node.base._slice is not None:
            return cvx_var[idx, node.base._slice]
        else:
            # No slice means this is the entire unified state vector
            return cvx_var[idx, :]

    elif isinstance(node.base, Control):
        if "u" not in lowerer.variable_map:
            raise ValueError(
                "Control vector 'u' not found in variable_map. "
                "For cross-node constraints, 'u' must be the full trajectory (N, n_u)."
            )

        cvx_var = lowerer.variable_map["u"]  # Should be (N, n_u) for cross-node constraints

        # Apply slice if control has one assigned
        if node.base._slice is not None:
            return cvx_var[idx, node.base._slice]
        else:
            # No slice means this is the entire unified control vector
            return cvx_var[idx, :]

    else:
        # Compound expression (e.g., position[0].at(5))
        # This is more complex - would need to lower base in single-node context
        raise NotImplementedError(
            "Compound expressions in NodeReference are not yet supported for CVXPy lowering. "
            f"Base expression type: {type(node.base).__name__}. "
            "Only State and Control NodeReferences are currently supported."
        )
