"""Augmented Lagrangian autotuner for SCP weights.

The autotuner is a **pure functional update** on the :class:`AlgorithmState`
pytree — no mutation, no Python ``if`` on tracer values, no list appends. The
SCP loop records history outside this module.
"""

from typing import TYPE_CHECKING, List, Literal

import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_acceptance_ratio,
    color_adaptive_state,
)

from ..base import AdaptiveStateCode, AutotuningBase, HyperParams

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..base import AlgorithmState, CandidateIterate


class AugmentedLagrangianHyper(HyperParams):
    """Declared hyperparameters for :class:`AugmentedLagrangian`."""

    rho_init: float = 1.0
    rho_max: float = 1e2
    lam_cost_drop: int = -1


class AugmentedLagrangian(AutotuningBase):
    """Augmented Lagrangian autotuner.

    Uses an acceptance-ratio :math:`\\rho` between predicted and actual
    reduction in the nonlinear objective to drive a four-bucket update of
    the trust-region weight ``lam_prox`` and to update the virtual-control /
    virtual-buffer weights from constraint violations.

    The four buckets — ``REJECT`` / ``ACCEPT_HIGHER`` / ``ACCEPT_CONSTANT`` /
    ``ACCEPT_LOWER`` — are selected via a ``jnp.where`` cascade so the whole
    update traces under ``jax.jit``.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

    COLUMNS: List[Column] = [
        Column("J_nonlin", "J_nonlin", 8, "{: .1e}", None, Verbosity.STANDARD),
        Column("J_lin", "J_lin", 8, "{: .1e}", None, Verbosity.STANDARD),
        Column("pred_reduction", "pred_red", 9, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column("actual_reduction", "act_red", 9, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column(
            "acceptance_ratio",
            "acc_ratio",
            9,
            "{: .2e}",
            color_acceptance_ratio,
            Verbosity.STANDARD,
        ),
        Column("lam_prox", "lam_prox", 8, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column("adaptive_state", "Adaptive", 16, "{}", color_adaptive_state, Verbosity.FULL),
    ]

    def __init__(
        self,
        rho_init: float = 1.0,
        rho_max: float = 1e2,
        gamma_1: float = 2.0,
        gamma_2: float = 0.5,
        eta_0: float = 1e-2,
        eta_1: float = 1e-1,
        eta_2: float = 0.8,
        ep: float = 0.99,
        eta_lambda: float = 1e1,
        lam_vc_max: float = 1e5,
        lam_prox_min: float = 1e-3,
        lam_prox_max: float = 1e4,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        """Initialize Augmented Lagrangian autotuning parameters.

        All parameters have defaults and can be modified after instantiation
        via attribute access (e.g., ``autotuner.lam_prox_max = 1e6``); the
        declared hyperparameters (``rho_init`` / ``rho_max`` /
        ``lam_cost_drop``) live on the frozen ``hyper`` container instead
        (``autotuner.hyper = dataclasses.replace(autotuner.hyper,
        rho_max=1e7)``) and are also per-solve overrides — see
        :class:`AutotuningBase`.

        Args:
            rho_init: Initial penalty parameter for constraints. Defaults to 1.0.
            rho_max: Maximum penalty parameter. Defaults to 1e2.
            gamma_1: Factor to increase trust region weight when ratio is low.
                Defaults to 2.0.
            gamma_2: Factor to decrease trust region weight when ratio is high.
                Defaults to 0.5.
            eta_0: Acceptance ratio threshold below which solution is rejected.
                Defaults to 1e-2.
            eta_1: Threshold above which solution is accepted with constant weight.
                Defaults to 1e-1.
            eta_2: Threshold above which solution is accepted with lower weight.
                Defaults to 0.8.
            ep: Threshold for virtual control weight update (nu > ep vs nu <= ep).
                Must lie in (0, 1). Defaults to 0.99.
            eta_lambda: Step size for virtual control weight update. Defaults to 1e1.
            lam_vc_max: Maximum virtual control penalty weight. Defaults to 1e5.
            lam_prox_min: Minimum trust region (proximal) weight. Defaults to 1e-3.
            lam_prox_max: Maximum trust region (proximal) weight. Defaults to 1e4.
            lam_cost_drop: Iteration after which cost relaxation applies (-1 = never).
                Defaults to -1.
            lam_cost_relax: Factor applied to lam_cost after lam_cost_drop.
                Defaults to 1.0.
        """
        self.hyper = AugmentedLagrangianHyper(
            rho_init=rho_init,
            rho_max=rho_max,
            lam_cost_drop=lam_cost_drop,
        )
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.ep = ep
        self.eta_lambda = eta_lambda
        self.lam_vc_max = lam_vc_max
        self.lam_prox_min = lam_prox_min
        self.lam_prox_max = lam_prox_max
        self.lam_cost_relax = lam_cost_relax

    # -----------------------------------------------------------------------
    # Weight updates from constraint violation
    # -----------------------------------------------------------------------

    def _update_virtual_control_weights(
        self,
        candidate: "CandidateIterate",
        candidate_x_prop: jnp.ndarray,
        settings: Config,
        lam_vc: jnp.ndarray,
        lam_prox: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update virtual control penalty weights from state violation.

        Computes scaled violation ``nu = inv_S_x @ |x[1:] - x_prop|`` and
        applies a piecewise update: linear in ``nu`` when ``nu > ep``,
        quadratic otherwise. Result is clipped to ``lam_vc_max``.
        """
        nu = (settings.sim.inv_S_x @ jnp.abs(candidate.x[1:] - candidate_x_prop).T).T
        # TODO: (haynec) use per-variable lam_prox to scale VC updates proportionally
        lam_prox_scalar = jnp.max(lam_prox)
        scale = self.eta_lambda * (1.0 / (2.0 * lam_prox_scalar))
        case1 = lam_vc + nu * scale
        case2 = lam_vc + (nu**2) / self.ep * scale
        vc_new = jnp.where(nu > self.ep, case1, case2)
        return jnp.minimum(self.lam_vc_max, vc_new)

    def _update_virtual_buffer_nodal_weights(
        self,
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        lam_vb_nodal: jnp.ndarray,
        lam_prox: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update virtual-buffer weights for nodal constraints.

        Evaluates each nodal constraint, computes the same piecewise update
        as for virtual control, and writes the result back into the
        ``(N, n_nodal)`` weight array via ``.at[...].set(...)``.
        """
        lam_vb_new = lam_vb_nodal
        lam_prox_scalar = jnp.max(lam_prox)
        scale = self.eta_lambda * (1.0 / (2.0 * lam_prox_scalar))

        for idx, constraint in enumerate(nodal_constraints.nodal):
            g = constraint.func(candidate.x, candidate.u, 0, params)
            nu = jnp.maximum(0.0, g)

            if constraint.nodes is not None:
                nodes_array = jnp.asarray(constraint.nodes)
                nu_slice = nu[nodes_array]
                current = lam_vb_nodal[nodes_array, idx]
            else:
                nu_slice = nu
                current = lam_vb_nodal[:, idx]

            case1 = current + nu_slice * scale
            case2 = current + (nu_slice**2) / self.ep * scale
            updated = jnp.where(nu_slice > self.ep, case1, case2)

            if constraint.nodes is not None:
                lam_vb_new = lam_vb_new.at[nodes_array, idx].set(updated)
            else:
                lam_vb_new = lam_vb_new.at[:, idx].set(updated)

        return jnp.minimum(self.lam_vc_max, lam_vb_new)

    def _update_virtual_buffer_cross_weights(
        self,
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        lam_vb_cross: jnp.ndarray,
        lam_prox: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update virtual-buffer weights for cross-node constraints."""
        lam_vb_new = lam_vb_cross
        lam_prox_scalar = jnp.max(lam_prox)
        scale = self.eta_lambda * (1.0 / (2.0 * lam_prox_scalar))

        for idx, constraint in enumerate(nodal_constraints.cross_node):
            g = constraint.func(candidate.x, candidate.u, params)
            nu = jnp.sum(jnp.maximum(0.0, g))
            current = lam_vb_cross[idx]
            case1 = current + nu * scale
            case2 = current + (nu**2) / self.ep * scale
            updated = jnp.where(nu > self.ep, case1, case2)
            lam_vb_new = lam_vb_new.at[idx].set(updated)

        return jnp.minimum(self.lam_vc_max, lam_vb_new)

    # -----------------------------------------------------------------------
    # Main update
    # -----------------------------------------------------------------------

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
    ) -> "AlgorithmState":
        """Return the next-iterate state per the Augmented Lagrangian rules.

        Pure functional update — see class docstring.
        """
        candidate_x_prop = candidate.x_prop_plus[1:]
        nonlin_cost, nonlin_pen, nodal_pen = self.calculate_nonlinear_penalty(
            candidate_x_prop,
            candidate.x,
            candidate.u,
            state.lam_vc,
            state.lam_vb_nodal,
            state.lam_vb_cross,
            state.lam_cost,
            nodal_constraints,
            params,
            settings,
        )
        J_nonlin = nonlin_cost + nonlin_pen + nodal_pen

        # Cost relaxation: when state.k > hyper.lam_cost_drop, scale
        # state.lam_cost; otherwise reset to the algorithm's initial weight
        # (carried on the pytree as state.lam_cost_init, broadcast at
        # from_settings()). Both constants ride the pytree so per-solve
        # overrides and vmap sweeps reach the traced body. Scalar
        # lam_cost_relax preserves the user-specified per-state weight ratios.
        lam_cost_next = jnp.where(
            state.k > state.hyper.lam_cost_drop,
            state.lam_cost * self.lam_cost_relax,
            state.lam_cost_init,
        )

        def first_iter(state):
            # Iter 1: accept unconditionally, leave weights at their init values,
            # only refresh trajectory + propagation fields.
            return state.replace(
                x=candidate.x,
                u=candidate.u,
                x_prop=candidate.x_prop,
                x_prop_plus=candidate.x_prop_plus,
                lam_cost=lam_cost_next,
                J_nonlin=J_nonlin,
                adaptive_state_code=jnp.asarray(int(AdaptiveStateCode.INITIAL), dtype=jnp.int32),
            )

        def later_iter(state):
            # Recompute the previous iterate's J_nonlin from the pytree fields
            # (state.x/state.u were the *previous* accepted iterate).
            prev_x_prop = state.x_prop_plus[1:]
            prev_cost, prev_pen, prev_nodal_pen = self.calculate_nonlinear_penalty(
                prev_x_prop,
                state.x,
                state.u,
                state.lam_vc,
                state.lam_vb_nodal,
                state.lam_vb_cross,
                state.lam_cost,
                nodal_constraints,
                params,
                settings,
            )
            prev_J_nonlin = prev_cost + prev_pen + prev_nodal_pen

            actual = prev_J_nonlin - J_nonlin
            predicted = prev_J_nonlin - candidate.J_lin
            # If predicted reduction is exactly zero, force the reject bucket
            # (rho = -inf) deterministically instead of raising.
            safe_pred = jnp.where(predicted == 0.0, 1.0, predicted)
            rho = jnp.where(predicted == 0.0, -jnp.inf, actual / safe_pred)

            is_reject = rho < self.eta_0
            is_accept_higher = (rho >= self.eta_0) & (rho < self.eta_1)
            is_accept_constant = (rho >= self.eta_1) & (rho < self.eta_2)
            # is_accept_lower implicit (else)
            accepted = ~is_reject

            # Compute both lam_prox candidates and gate.
            lp_higher = jnp.minimum(self.lam_prox_max, self.gamma_1 * state.lam_prox)
            lp_lower = jnp.maximum(self.lam_prox_min, self.gamma_2 * state.lam_prox)
            new_lam_prox = jnp.where(
                is_reject | is_accept_higher,
                lp_higher,
                jnp.where(is_accept_constant, state.lam_prox, lp_lower),
            )

            # Virtual-control and virtual-buffer updates: compute against the
            # *candidate* trajectory and gate by `accepted`. Reject keeps the
            # previous values (so the next subproblem doesn't see a bumped
            # penalty that wasn't earned).
            lam_vc_upd = self._update_virtual_control_weights(
                candidate, candidate_x_prop, settings, state.lam_vc, new_lam_prox
            )
            lam_vb_nodal_upd = self._update_virtual_buffer_nodal_weights(
                candidate, nodal_constraints, params, state.lam_vb_nodal, new_lam_prox
            )
            lam_vb_cross_upd = self._update_virtual_buffer_cross_weights(
                candidate, nodal_constraints, params, state.lam_vb_cross, new_lam_prox
            )

            code = jnp.where(
                is_reject,
                jnp.int32(AdaptiveStateCode.REJECT),
                jnp.where(
                    is_accept_higher,
                    jnp.int32(AdaptiveStateCode.ACCEPT_HIGHER),
                    jnp.where(
                        is_accept_constant,
                        jnp.int32(AdaptiveStateCode.ACCEPT_CONSTANT),
                        jnp.int32(AdaptiveStateCode.ACCEPT_LOWER),
                    ),
                ),
            )

            return state.replace(
                x=jnp.where(accepted, candidate.x, state.x),
                u=jnp.where(accepted, candidate.u, state.u),
                x_prop=jnp.where(accepted, candidate.x_prop, state.x_prop),
                x_prop_plus=jnp.where(accepted, candidate.x_prop_plus, state.x_prop_plus),
                lam_prox=new_lam_prox,
                lam_vc=jnp.where(accepted, lam_vc_upd, state.lam_vc),
                lam_vb_nodal=jnp.where(accepted, lam_vb_nodal_upd, state.lam_vb_nodal),
                lam_vb_cross=jnp.where(accepted, lam_vb_cross_upd, state.lam_vb_cross),
                lam_cost=lam_cost_next,
                J_nonlin=jnp.where(accepted, J_nonlin, state.J_nonlin),
                predicted_reduction=predicted,
                actual_reduction=actual,
                acceptance_ratio=rho,
                adaptive_state_code=code,
            )

        return jax.lax.cond(state.k == 1, first_iter, later_iter, state)


# =============================================================================
# Pydantic spec for dict / YAML validation
# =============================================================================


class AugmentedLagrangianSpec(BaseModel):
    """Validates AugmentedLagrangian configuration from dict/YAML input."""

    type: Literal["AugmentedLagrangian"] = "AugmentedLagrangian"
    rho_init: float = 1.0
    rho_max: float = 1e2
    gamma_1: float = 2.0
    gamma_2: float = 0.5
    eta_0: float = 1e-2
    eta_1: float = 1e-1
    eta_2: float = 0.8
    ep: float = 0.99
    eta_lambda: float = 1e1
    lam_vc_max: float = 1e5
    lam_prox_min: float = 1e-3
    lam_prox_max: float = 1e4
    lam_cost_drop: int = -1
    lam_cost_relax: float = 1.0

    model_config = ConfigDict(extra="forbid")

    def build(self) -> AugmentedLagrangian:
        return AugmentedLagrangian(**self.model_dump(exclude={"type"}, exclude_unset=True))
