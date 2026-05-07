"""Validation for bring-your-own-functions (byof).

This module provides validation for user-provided JAX functions in expert mode,
checking signatures, shapes, and differentiability before use.
"""

import inspect
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from openscvx.symbolic.expr.state import State

from openscvx.expert.byof import ByofSpec

__all__ = ["validate_byof"]


def validate_byof(
    byof: Union[ByofSpec, dict],
    states: List["State"],
    n_x: int,
    n_u: int,
    N: int = None,
    parameters: dict = None,
) -> None:
    """Validate byof function signatures and shapes.

    Checks that user-provided functions have the correct signatures and return
    appropriate shapes. Performs validation before functions are used to provide
    clear error messages.

    Accepts both a :class:`ByofSpec` and a raw dict (which is validated through
    ``ByofSpec.model_validate`` first, catching unknown keys and missing fields).

    Args:
        byof: ByofSpec or raw dict with user-provided functions
        states: List of State objects for determining expected shapes
        n_x: Total dimension of the unified state vector
        n_u: Total dimension of the unified control vector
        N: Number of nodes in the trajectory (optional). If provided, validates
            node indices in nodal constraints.

    Raises:
        ValueError: If any function has invalid signature or returns wrong shape
        TypeError: If functions are not callable

    Example:
        >>> validate_byof(byof, states, n_x=10, n_u=3, N=50)  # Raises if invalid
    """
    import jax
    import jax.numpy as jnp

    # Convert raw dict to ByofSpec (validates keys, required fields, types)
    if not isinstance(byof, ByofSpec):
        byof = ByofSpec.model_validate(byof)

    # Validate byof parameters
    from openscvx.symbolic.expr.parameter import Parameter

    for i, param in enumerate(byof.parameters):
        if not isinstance(param, Parameter):
            raise TypeError(f"byof parameters[{i}] must be a Parameter object, got {type(param)}")

    # Create dummy inputs for testing
    dummy_x = jnp.zeros(n_x)
    dummy_u = jnp.zeros(n_u)
    dummy_node = 0
    dummy_params = dict(parameters) if parameters else {}

    # Validate dynamics functions
    byof_dynamics = byof.dynamics
    if byof_dynamics:
        # Build mapping from state name to expected shape
        state_shapes = {state.name: state.shape for state in states}

        for state_name, fn in byof_dynamics.items():
            if state_name not in state_shapes:
                raise ValueError(
                    f"byof dynamics '{state_name}' does not match any state name. "
                    f"Available states: {list(state_shapes.keys())}"
                )

            if not callable(fn):
                raise TypeError(f"byof dynamics '{state_name}' must be callable, got {type(fn)}")

            # Check signature
            sig = inspect.signature(fn)
            if len(sig.parameters) != 4:
                raise ValueError(
                    f"byof dynamics '{state_name}' must have signature f(x, u, node, params), "
                    f"got {len(sig.parameters)} parameters: {list(sig.parameters.keys())}"
                )

            # Test call and check output shape
            try:
                result = fn(dummy_x, dummy_u, dummy_node, dummy_params)
            except Exception as e:
                raise ValueError(
                    f"byof dynamics '{state_name}' failed on test call with "
                    f"x.shape={dummy_x.shape}, u.shape={dummy_u.shape}: {e}"
                ) from e

            expected_shape = state_shapes[state_name]
            result_shape = jnp.asarray(result).shape
            if result_shape != expected_shape:
                raise ValueError(
                    f"byof dynamics '{state_name}' returned shape {result_shape}, "
                    f"expected {expected_shape} (state '{state_name}' shape)"
                )

            # Test that forward-mode AD works (the discretizer uses jax.jacfwd,
            try:
                tangent = jnp.ones_like(dummy_x)
                jax.jvp(
                    lambda x: fn(x, dummy_u, dummy_node, dummy_params),
                    (dummy_x,),
                    (tangent,),
                )
            except Exception as e:
                raise ValueError(
                    f"byof dynamics '{state_name}' is not differentiable with JAX. "
                    f"Ensure the function uses JAX operations (jax.numpy, not numpy): {e}"
                ) from e

    # Validate dynamics_discrete functions (same signature and shape as dynamics)
    byof_dynamics_discrete = byof.dynamics_discrete
    if byof_dynamics_discrete:
        state_shapes = {state.name: state.shape for state in states}
        for state_name, fn in byof_dynamics_discrete.items():
            if state_name not in state_shapes:
                raise ValueError(
                    f"byof dynamics_discrete '{state_name}' does not match any state name. "
                    f"Available states: {list(state_shapes.keys())}"
                )
            if not callable(fn):
                raise TypeError(
                    f"byof dynamics_discrete '{state_name}' must be callable, got {type(fn)}"
                )
            sig = inspect.signature(fn)
            if len(sig.parameters) != 4:
                raise ValueError(
                    f"byof dynamics_discrete '{state_name}' must have signature "
                    f"f(x, u, node, params), got {len(sig.parameters)} parameters: "
                    f"{list(sig.parameters.keys())}"
                )
            try:
                result = fn(dummy_x, dummy_u, dummy_node, dummy_params)
            except Exception as e:
                raise ValueError(
                    f"byof dynamics_discrete '{state_name}' failed on test call with "
                    f"x.shape={dummy_x.shape}, u.shape={dummy_u.shape}: {e}"
                ) from e
            expected_shape = state_shapes[state_name]
            result_shape = jnp.asarray(result).shape
            if result_shape != expected_shape:
                raise ValueError(
                    f"byof dynamics_discrete '{state_name}' returned shape {result_shape}, "
                    f"expected {expected_shape} (state '{state_name}' shape)"
                )
            try:
                tangent = jnp.ones_like(dummy_x)
                jax.jvp(
                    lambda x: jnp.sum(fn(x, dummy_u, dummy_node, dummy_params)),
                    (dummy_x,),
                    (tangent,),
                )
            except Exception as e:
                raise ValueError(
                    f"byof dynamics_discrete '{state_name}' is not differentiable with JAX. "
                    f"Ensure the function uses JAX operations (jax.numpy, not numpy): {e}"
                ) from e

    # Validate nodal constraints
    for i, constraint_spec in enumerate(byof.nodal_constraints):
        fn = constraint_spec.constraint_fn
        if not callable(fn):
            raise TypeError(
                f"byof nodal_constraints[{i}]['constraint_fn'] must be callable, got {type(fn)}"
            )

        # Check signature
        sig = inspect.signature(fn)
        if len(sig.parameters) != 4:
            raise ValueError(
                f"byof nodal_constraints[{i}]['constraint_fn'] must have signature "
                f"f(x, u, node, params), "
                f"got {len(sig.parameters)} parameters: {list(sig.parameters.keys())}"
            )

        # Test call
        try:
            result = fn(dummy_x, dummy_u, dummy_node, dummy_params)
        except Exception as e:
            raise ValueError(
                f"byof nodal_constraints[{i}]['constraint_fn'] failed on test call with "
                f"x.shape={dummy_x.shape}, u.shape={dummy_u.shape}: {e}"
            ) from e

        # Check that result is array-like (can be scalar or vector)
        try:
            result_array = jnp.asarray(result)
        except Exception as e:
            raise ValueError(
                f"byof nodal_constraints[{i}]['constraint_fn'] must return array-like value, "
                f"got {type(result)}: {e}"
            ) from e

        # Test forward-mode AD
        try:
            tangent = jnp.ones_like(dummy_x)
            jax.jvp(
                lambda x: jnp.sum(fn(x, dummy_u, dummy_node, dummy_params)),
                (dummy_x,),
                (tangent,),
            )
        except Exception as e:
            raise ValueError(
                f"byof nodal_constraints[{i}]['constraint_fn'] is not differentiable with JAX: {e}"
            ) from e

        # Validate nodes if provided
        if constraint_spec.nodes is not None:
            nodes = constraint_spec.nodes
            if len(nodes) == 0:
                raise ValueError(f"byof nodal_constraints[{i}]['nodes'] cannot be empty")

            # Validate node indices if N is provided
            if N is not None:
                for node in nodes:
                    # Handle negative indices (e.g., -1 for last node)
                    normalized_node = node if node >= 0 else N + node
                    # Validate range
                    if not (0 <= normalized_node < N):
                        raise ValueError(
                            f"byof nodal_constraints[{i}]['nodes'] contains invalid index {node} "
                            f"(normalized: {normalized_node}). Valid range is [0, {N}) or "
                            f"negative indices [-{N}, -1]."
                        )

    # Validate cross-nodal constraints
    dummy_X = jnp.zeros((10, n_x))  # Dummy trajectory with 10 nodes
    dummy_U = jnp.zeros((10, n_u))

    for i, fn in enumerate(byof.cross_nodal_constraints):
        if not callable(fn):
            raise TypeError(f"byof cross_nodal_constraints[{i}] must be callable, got {type(fn)}")

        # Check signature
        sig = inspect.signature(fn)
        if len(sig.parameters) != 3:
            raise ValueError(
                f"byof cross_nodal_constraints[{i}] must have signature f(X, U, params), "
                f"got {len(sig.parameters)} parameters: {list(sig.parameters.keys())}"
            )

        # Test call
        try:
            result = fn(dummy_X, dummy_U, dummy_params)
        except Exception as e:
            raise ValueError(
                f"byof cross_nodal_constraints[{i}] failed on test call with "
                f"X.shape={dummy_X.shape}, U.shape={dummy_U.shape}: {e}"
            ) from e

        # Check that result is array-like
        try:
            result_array = jnp.asarray(result)
        except Exception as e:
            raise ValueError(
                f"byof cross_nodal_constraints[{i}] must return array-like value, "
                f"got {type(result)}: {e}"
            ) from e

        # Test forward-mode AD
        try:
            tangent = jnp.ones_like(dummy_X)
            jax.jvp(
                lambda X: jnp.sum(fn(X, dummy_U, dummy_params)),
                (dummy_X,),
                (tangent,),
            )
        except Exception as e:
            raise ValueError(
                f"byof cross_nodal_constraints[{i}] is not differentiable with JAX: {e}"
            ) from e

    # Validate CTCS constraints
    for i, ctcs_spec in enumerate(byof.ctcs_constraints):
        fn = ctcs_spec.constraint_fn
        if not callable(fn):
            raise TypeError(
                f"byof ctcs_constraints[{i}]['constraint_fn'] must be callable, got {type(fn)}"
            )

        # Check signature
        sig = inspect.signature(fn)
        if len(sig.parameters) != 4:
            raise ValueError(
                f"byof ctcs_constraints[{i}]['constraint_fn'] must have signature "
                f"f(x, u, node, params), got {len(sig.parameters)} parameters: "
                f"{list(sig.parameters.keys())}"
            )

        # Test call
        try:
            result = fn(dummy_x, dummy_u, dummy_node, dummy_params)
        except Exception as e:
            raise ValueError(
                f"byof ctcs_constraints[{i}]['constraint_fn'] failed on test call: {e}"
            ) from e

        # Check that result is scalar
        result_array = jnp.asarray(result)
        if result_array.shape != ():
            raise ValueError(
                f"byof ctcs_constraints[{i}]['constraint_fn'] must return a scalar, "
                f"got shape {result_array.shape}"
            )

        # Test forward-mode AD
        try:
            tangent = jnp.ones_like(dummy_x)
            jax.jvp(
                lambda x: fn(x, dummy_u, dummy_node, dummy_params),
                (dummy_x,),
                (tangent,),
            )
        except Exception as e:
            raise ValueError(
                f"byof ctcs_constraints[{i}]['constraint_fn'] is not differentiable with JAX: {e}"
            ) from e

        # Validate penalty function if callable
        penalty_spec = ctcs_spec.penalty
        if callable(penalty_spec):
            try:
                test_residual = jnp.array(0.5)
                penalty_result = penalty_spec(test_residual)
                jnp.asarray(penalty_result)
            except Exception as e:
                raise ValueError(
                    f"byof ctcs_constraints[{i}]['penalty'] custom function failed: {e}"
                ) from e

        # Validate idx
        if ctcs_spec.idx < 0:
            raise ValueError(
                f"byof ctcs_constraints[{i}]['idx'] must be non-negative, got {ctcs_spec.idx}"
            )

        # Validate bounds
        bounds = ctcs_spec.bounds
        if bounds[0] > bounds[1]:
            raise ValueError(
                f"byof ctcs_constraints[{i}]['bounds'] min ({bounds[0]}) must be <= "
                f"max ({bounds[1]})"
            )

        # Validate initial value is within bounds
        if ctcs_spec.initial is not None:
            initial = ctcs_spec.initial
            if not (bounds[0] <= initial <= bounds[1]):
                raise ValueError(
                    f"byof ctcs_constraints[{i}]['initial'] ({initial}) must be within "
                    f"bounds [{bounds[0]}, {bounds[1]}]"
                )

        # Validate over (node interval) if provided
        if ctcs_spec.over is not None:
            start, end = ctcs_spec.over
            if start >= end:
                raise ValueError(
                    f"byof ctcs_constraints[{i}]['over'] start ({start}) must be < end ({end})"
                )
            if start < 0:
                raise ValueError(
                    f"byof ctcs_constraints[{i}]['over'] start ({start}) must be non-negative"
                )
            # Validate against trajectory length if N is provided
            if N is not None:
                if end > N:
                    raise ValueError(
                        f"byof ctcs_constraints[{i}]['over'] end ({end}) exceeds "
                        f"trajectory length ({N})"
                    )
