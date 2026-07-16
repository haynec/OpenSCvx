"""Canonicalise user ``.convex()`` nodal constraints into cone constraints.

The function :func:`canonicalize_nodal_constraint` dispatches a single
:class:`~openscvx.symbolic.expr.constraint.NodalConstraint` (whose inner
constraint has ``is_convex == True``) to one or more
:class:`~openscvx.solvers.cones.ConeConstraint` objects that a JAX-native
solver can assemble directly.

Dispatch table
--------------

+--------------------------------------+-----------------------------------+
| Symbolic pattern                     | Cone type                         |
+======================================+===================================+
| ``Equality(affine, affine)``         | :class:`ZeroConeConstraint`       |
+--------------------------------------+-----------------------------------+
| ``Inequality(affine, affine)``       | :class:`NonnegConeConstraint`     |
+--------------------------------------+-----------------------------------+
| ``Inequality(Norm(e,2), t)``         | :class:`SOCConstraint`            |
| ``Inequality(Norm(e,"fro"), t)``     |                                   |
+--------------------------------------+-----------------------------------+
| ``Inequality(Norm(e,"inf"), t)``     | :class:`NonnegConeConstraint`     |
|                                      | (2·dim rows)                      |
+--------------------------------------+-----------------------------------+
| ``Inequality(Norm(e,1), t)``         | :exc:`NotImplementedError`        |
|                                      | (needs auxiliary variables)       |
+--------------------------------------+-----------------------------------+

where *affine* means :func:`~openscvx.symbolic.affine.is_affine_in_state_control`
returns ``True``, and *t* is :func:`~openscvx.symbolic.affine.is_constant`
(a :class:`~openscvx.symbolic.expr.Parameter` or literal scalar).

For :class:`SOCConstraint` the ``bound_fn`` may also be affine in state/control
(e.g. a state-dependent upper bound), as long as the bound itself is a scalar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from openscvx.solvers.cones import ConeConstraint
    from openscvx.symbolic.expr.constraint import NodalConstraint


def canonicalize_nodal_constraint(
    nc: "NodalConstraint",
) -> List["ConeConstraint"]:
    """Convert a single ``.convex()`` nodal constraint into cone constraints.

    Parameters
    ----------
    nc:
        A :class:`~openscvx.symbolic.expr.constraint.NodalConstraint` whose
        inner :class:`~openscvx.symbolic.expr.constraint.Constraint` has
        ``is_convex == True``.

    Returns
    -------
    list[ConeConstraint]
        One or more cone constraints that together express *nc*.  Each
        constraint is active at ``nc.nodes``.

    Raises
    ------
    ValueError
        If the constraint pattern is recognised as non-affine in
        state/control (and therefore cannot be assembled as a conic row).
    NotImplementedError
        For L1-norm constraints, which require auxiliary variables not
        yet supported in the JAX assemblers.
    TypeError
        If the inner constraint is neither
        :class:`~openscvx.symbolic.expr.constraint.Equality` nor
        :class:`~openscvx.symbolic.expr.constraint.Inequality`.
    """
    from openscvx.solvers.cones import NonnegConeConstraint, SOCConstraint, ZeroConeConstraint
    from openscvx.symbolic.affine import is_affine_in_state_control, is_constant
    from openscvx.symbolic.expr.constraint import Equality, Inequality, MatrixInequality
    from openscvx.symbolic.expr.linalg import Norm
    from openscvx.symbolic.lower import lower_to_jax

    constraint = nc.constraint
    nodes = tuple(nc.nodes)

    # ------------------------------------------------------------------
    # Helper: get the flat output dimension of an expression.
    # ------------------------------------------------------------------
    def _m(expr) -> int:
        shape = expr.check_shape()
        return int(np.prod(shape)) if shape else 1

    # ==================================================================
    # EQUALITY  →  ZeroConeConstraint
    # ==================================================================
    if isinstance(constraint, Equality):
        lhs, rhs = constraint.lhs, constraint.rhs
        if not (is_affine_in_state_control(lhs) and is_affine_in_state_control(rhs)):
            raise ValueError(
                f"Equality constraint is not affine in state/control: {constraint!r}.\n"
                "Only affine equality constraints are supported by the JAX "
                "backends.  Use CVXPyPTRSolver for non-affine convex constraints."
            )
        jax_fn = lower_to_jax(constraint)  # returns lhs − rhs
        m = _m(constraint)
        return [ZeroConeConstraint(nodes=nodes, jax_fn=jax_fn, m=m)]

    # ==================================================================
    # MATRIX INEQUALITY  →  PSDConeConstraint
    # ==================================================================
    if isinstance(constraint, MatrixInequality):
        raise NotImplementedError(
            "PSD/LMI constraints (>>, <<) require CVXPyPTRSolver — "
            "JAX-native backends (QPAX, Moreau) do not support SDP."
        )

    # ==================================================================
    # INEQUALITY  →  varies by pattern
    # ==================================================================
    if isinstance(constraint, Inequality):
        lhs, rhs = constraint.lhs, constraint.rhs

        # --------------------------------------------------------------
        # Detect canonical form produced by Constraint.canonicalize():
        #   (Norm(e) - bound) <= 0   →  extract Norm and bound.
        # builder.py canonicalises all constraints before lowering, so
        # this is the form we typically receive.
        # --------------------------------------------------------------
        from openscvx.symbolic.expr.arithmetic import Sub

        _norm_lhs: "Norm | None" = None
        _norm_bound_rhs = None
        if isinstance(lhs, Norm) and is_constant(rhs):
            # Direct form: Norm(e) <= constant
            _norm_lhs = lhs
            _norm_bound_rhs = rhs
        elif (
            isinstance(lhs, Sub)
            and isinstance(lhs.left, Norm)
            and is_constant(lhs.right)
            and is_constant(rhs)
        ):
            # Canonical form: (Norm(e) - bound) <= 0
            _norm_lhs = lhs.left
            _norm_bound_rhs = lhs.right

        # --------------------------------------------------------------
        # Special case: norm-cone  Norm(e) ≤ t
        # --------------------------------------------------------------
        if _norm_lhs is not None and _norm_bound_rhs is not None:
            lhs = _norm_lhs  # type: ignore[assignment]
            rhs = _norm_bound_rhs  # type: ignore[assignment]

        if isinstance(lhs, Norm):
            norm_arg = lhs.operand
            ord_ = lhs.ord

            # The argument of the norm must be affine in state/control.
            if not is_affine_in_state_control(norm_arg):
                raise ValueError(
                    f"Norm argument is not affine in state/control: {norm_arg!r}.\n"
                    "Only norms of affine expressions are supported."
                )

            # The bound must be affine in state/control (scalar).
            if not is_affine_in_state_control(rhs):
                raise ValueError(f"Norm upper bound is not affine in state/control: {rhs!r}.")

            # L1 norm: not yet supported (needs auxiliary variables).
            if ord_ == 1:
                raise NotImplementedError(
                    "L1-norm constraints require auxiliary epigraph variables "
                    "that are not yet supported in the JAX assemblers.  "
                    "Use CVXPyPTRSolver or reformulate manually."
                )

            # L2 / Frobenius  →  SOCConstraint
            if ord_ in (2, "fro"):
                arg_fn = lower_to_jax(norm_arg)
                bound_fn = lower_to_jax(rhs)
                norm_arg_shape = norm_arg.check_shape()
                m_arg = int(np.prod(norm_arg_shape)) if norm_arg_shape else 1

                # Wrap arg_fn to ensure it always returns a 1-D array.
                _raw_arg_fn = arg_fn

                def _flat_arg_fn(x, u, node, params, _fn=_raw_arg_fn, _m=m_arg):
                    import jax.numpy as _jnp

                    return _jnp.reshape(_fn(x, u, node, params), (_m,))

                # Ensure bound_fn returns a scalar.
                _raw_bound_fn = bound_fn

                def _scalar_bound_fn(x, u, node, params, _fn=_raw_bound_fn):
                    import jax.numpy as _jnp

                    return _jnp.reshape(_fn(x, u, node, params), ())

                if getattr(constraint, "slack_weight", None) is not None:
                    raise NotImplementedError(
                        "slack() is not yet supported for SOC (norm) constraints."
                        "Use it on affine inequality constraints only."
                    )
                return [
                    SOCConstraint(
                        nodes=nodes,
                        arg_fn=_flat_arg_fn,
                        bound_fn=_scalar_bound_fn,
                        m_arg=m_arg,
                    )
                ]

            # L-infinity  →  NonnegConeConstraint  (2·dim rows)
            if ord_ == "inf":
                # ‖v‖∞ ≤ t  ⟺  v_i ≤ t  AND  −v_i ≤ t  for all i
                # Residual: [v - t·1; −v - t·1] ≤ 0  →  shape (2·dim_v,)
                arg_fn = lower_to_jax(norm_arg)
                bound_fn = lower_to_jax(rhs)
                norm_arg_shape = norm_arg.check_shape()
                dim_v = int(np.prod(norm_arg_shape)) if norm_arg_shape else 1
                _raw_arg_fn = arg_fn
                _raw_bound_fn = bound_fn

                def _inf_fn(x, u, node, params, _afn=_raw_arg_fn, _bfn=_raw_bound_fn, _dv=dim_v):
                    import jax.numpy as _jnp

                    v = _jnp.reshape(_afn(x, u, node, params), (_dv,))
                    t = _jnp.reshape(_bfn(x, u, node, params), ())
                    return _jnp.concatenate([v - t, -v - t])

                slack_weight = getattr(constraint, "slack_weight", None)
                return [
                    NonnegConeConstraint(
                        nodes=nodes, jax_fn=_inf_fn, m=2 * dim_v, slack_weight=slack_weight
                    )
                ]

            # Unknown norm order.
            raise NotImplementedError(
                f"Norm order {ord_!r} is not supported.  Supported orders: 1, 2, 'fro', 'inf'."
            )

        # --------------------------------------------------------------
        # General affine inequality  →  NonnegConeConstraint
        # --------------------------------------------------------------
        if not (is_affine_in_state_control(lhs) and is_affine_in_state_control(rhs)):
            raise ValueError(
                f"Inequality constraint is not affine in state/control: {constraint!r}.\n"
                "Only affine inequalities and norm-cone inequalities are "
                "supported by the JAX backends.  Use CVXPyPTRSolver for "
                "non-affine convex constraints."
            )
        jax_fn = lower_to_jax(constraint)  # returns lhs − rhs  (≤ 0)
        m = _m(constraint)
        slack_weight = getattr(constraint, "slack_weight", None)
        return [NonnegConeConstraint(nodes=nodes, jax_fn=jax_fn, m=m, slack_weight=slack_weight)]

    # ==================================================================
    # Unsupported inner constraint type
    # ==================================================================
    raise TypeError(
        f"Unsupported constraint type for canonicalization: {type(constraint).__name__}. "
        "Expected Equality or Inequality."
    )
