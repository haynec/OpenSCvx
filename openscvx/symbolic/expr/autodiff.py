"""User-supplied Jacobians for a subexpression.

Successive convexification linearizes the dynamics and constraints at every
iteration, so the *Jacobian* — not the value — is what shapes each convex
subproblem. Occasionally the exact Jacobian is the wrong one to hand the
solver: an aerodynamic drag term, a contact model, or a lookup table can be
smooth enough to integrate but stiff enough that its true derivative wrecks the
conditioning of the subproblem. The standard remedy is a deliberately
simplified (inexact) Jacobian — the trajectory still converges to a solution of
the *nonlinear* problem, because the value used for the defect is untouched;
only the search direction changes.

:class:`WithJacobian` is that remedy, written at sub-expression granularity::

    drag = -0.5 * rho * ox.Norm(vel) * vel
    accel = thrust / m + drag.with_jacobian({vel: J_drag})

The node wraps a subexpression and carries an override Jacobian per variable.
Lowering to JAX emits a :func:`jax.custom_jvp` rule around the wrapped
subexpression, so every downstream differentiation — the discretizer's
``jacfwd``, constraint linearization, sparse coloring — sees the user's
Jacobian in the overridden directions and autodiff everywhere else, with no
further special-casing. The primal value is always the wrapped expression
itself.
"""

import hashlib
from typing import Dict, List, Tuple

import numpy as np

from .expr import Expr, to_expr
from .variable import Variable


def _without_unit_axes(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Drop length-1 axes, the way :class:`Constant` squeezes a wrapped array."""
    return tuple(d for d in shape if d != 1)


class WithJacobian(Expr):
    """A subexpression whose derivative w.r.t. chosen variables is user-supplied.

    Construct this through :meth:`Expr.with_jacobian` rather than directly. The
    node evaluates exactly like the expression it wraps; only differentiation
    differs, and only in the overridden directions (see the module docstring for
    the motivation).

    Attributes:
        expr: The wrapped expression, and the node's value.
        overrides: ``(variable, jacobian)`` pairs in the order given, each
            Jacobian shaped ``(*expr.shape, *variable.shape)``.
    """

    def __init__(self, expr: Expr, jacobians: Dict[Variable, Expr]):
        """Wrap ``expr`` with an override Jacobian per variable.

        Args:
            expr: Expression whose derivative is being overridden.
            jacobians: Map from a ``State`` or ``Control`` to the Jacobian of
                ``expr`` with respect to it. Values are coerced with
                :func:`to_expr`, so constant matrices are accepted.

        Raises:
            TypeError: If a key is not a decision variable.
            ValueError: If no Jacobian is given.
        """
        if not jacobians:
            raise ValueError(
                "with_jacobian needs at least one {variable: jacobian} entry; an empty "
                "mapping would wrap the expression without changing anything."
            )
        self.expr = to_expr(expr)
        overrides = []
        for var, jac in jacobians.items():
            if not isinstance(var, Variable):
                raise TypeError(
                    f"with_jacobian keys must be the State or Control the derivative is "
                    f"taken with respect to, got {type(var).__name__}. Write "
                    f"`expr.with_jacobian({{vel: J_vel}})` with `vel` the variable itself."
                )
            overrides.append((var, to_expr(jac)))
        self.overrides: List[Tuple[Variable, Expr]] = overrides

    def children(self) -> List[Expr]:
        """Return the wrapped expression followed by each override Jacobian.

        The overridden variables are not children: they identify *directions*,
        not operands, and appear as leaves inside the Jacobian expressions only
        if the user put them there.
        """
        return [self.expr, *(jac for _, jac in self.overrides)]

    def canonicalize(self) -> Expr:
        """Canonicalize the wrapped expression and every override Jacobian."""
        return WithJacobian(
            self.expr.canonicalize(),
            {var: jac.canonicalize() for var, jac in self.overrides},
        )

    def check_shape(self) -> Tuple[int, ...]:
        """Return the wrapped expression's shape, validating each Jacobian.

        A Jacobian must be shaped ``(*expr.shape, *var.shape)``. Unit axes are
        ignored in the comparison: a constant is squeezed when it is wrapped, so
        an honest ``(3, 1)`` Jacobian arrives as ``(3,)``. The JAX lowering
        reshapes it back before contracting.

        Returns:
            The shape of the wrapped expression — the wrapper is value-transparent.

        Raises:
            ValueError: If a Jacobian's shape is not ``(*expr.shape, *var.shape)``.
        """
        shape = self.expr.check_shape()
        for var, jac in self.overrides:
            expected = (*shape, *var.shape)
            got = jac.check_shape()
            if _without_unit_axes(got) != _without_unit_axes(expected):
                raise ValueError(
                    f"Jacobian for {var.__class__.__name__} '{var.name}' has shape {got}, "
                    f"expected {expected}: the derivative of an expression of shape "
                    f"{shape} with respect to a variable of shape {var.shape}."
                )
        return shape

    def sparsity(self, n_x: int, n_u: int) -> Tuple[np.ndarray, np.ndarray]:
        """Conservative pattern: the children's, widened to the overridden columns.

        The default union over children is not enough here. A user Jacobian may
        be a constant (no dependence of its own) yet still couple outputs to
        every entry of the overridden variable, so those columns are marked
        dense on top of the children's pattern.
        """
        from .control import Control

        S_x, S_u = super().sparsity(n_x, n_u)
        for var, _ in self.overrides:
            if var._slice is None:
                continue
            S = S_u if isinstance(var, Control) else S_x
            S[:, var._slice] = True
        return S_x, S_u

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash the wrapper, the wrapped expression, and each override in order.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"WithJacobian")
        self.expr._hash_into(hasher)
        for var, jac in self.overrides:
            var._hash_into(hasher)
            jac._hash_into(hasher)

    def __repr__(self) -> str:
        vars_ = ", ".join(var.name for var, _ in self.overrides)
        return f"{self.expr!r}.with_jacobian({{{vars_}}})"
