"""JAX visitors for user-supplied Jacobians.

Visitors: WithJacobian

Lowers a ``WithJacobian`` node to its wrapped subexpression wrapped in a
:func:`jax.custom_jvp` rule. The rule zeroes the tangent of every overridden
variable before differentiating the inner function, then adds ``J @ dvar`` for
each override — so overridden directions come from the user's Jacobian and all
others from autodiff. Because the override lives inside the lowered function,
every downstream differentiation (the discretizer's ``jacfwd``, constraint
linearization, sparse coloring) picks it up with no further plumbing.

Unlike the ``Vmap`` and ``Cond`` visitors, this one does not pause the value
memo: the nested ``jax.jvp`` trace receives its own ``(x, u)`` tracers rather
than reusing the captured outer ones, and the memo is keyed on all four
arguments, so a value cached in one trace can never be handed to a caller in
another.
"""

import jax
import jax.numpy as jnp

from openscvx.symbolic.expr.autodiff import WithJacobian
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(WithJacobian)
def _visit_with_jacobian(lowerer, node: WithJacobian):
    """Lower a user-Jacobian override to a ``jax.custom_jvp``-wrapped function.

    Args:
        node: WithJacobian expression node

    Returns:
        Function (x, u, node, params) -> value of the wrapped expression, whose
        derivative w.r.t. each overridden variable is the user's Jacobian.

    Raises:
        ValueError: If an overridden variable has no slice assigned (unification
            not run).
    """
    inner = lowerer.lower(node.expr)
    out_shape = node.expr.check_shape()
    overrides = []
    for var, jac in node.overrides:
        if var._slice is None:
            raise ValueError(f"{var.__class__.__name__} {var.name!r} has no slice assigned")
        # Squeezed constants come back as e.g. () for a (1, 1) Jacobian, so the
        # declared shape is restored before contracting with the tangent.
        jac_shape = (*out_shape, *var.shape)
        overrides.append((var._slice, isinstance(var, Control), lowerer.lower(jac), jac_shape))
    overrides = tuple(overrides)

    def with_jacobian_fn(x, u, node_idx, params):
        def inner_xu(x_, u_):
            return inner(x_, u_, node_idx, params)

        f = jax.custom_jvp(inner_xu)

        @f.defjvp
        def _jvp(primals, tangents):
            x_, u_ = primals
            dx, du = tangents
            # Autodiff supplies the non-overridden directions: blank the
            # overridden slices so their contribution comes only from below.
            dx_free, du_free = dx, du
            for sl, is_control, _, _ in overrides:
                if is_control:
                    du_free = du_free.at[sl].set(0.0)
                else:
                    dx_free = dx_free.at[sl].set(0.0)
            out, tangent = jax.jvp(inner_xu, (x_, u_), (dx_free, du_free))
            for sl, is_control, jac_fn, jac_shape in overrides:
                J = jnp.reshape(jac_fn(x_, u_, node_idx, params), jac_shape)
                dvar = du[sl] if is_control else dx[sl]
                tangent = tangent + jnp.tensordot(J, dvar, axes=1)
            return out, tangent

        return f(x, u)

    return with_jacobian_fn
