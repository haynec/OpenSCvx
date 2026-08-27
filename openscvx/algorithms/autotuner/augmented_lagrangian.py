"""Augmented Lagrangian autotuner for SCP weights.

The autotuner is a **pure functional update** on the :class:`AlgorithmState`
pytree — no mutation, no Python ``if`` on tracer values, no list appends. The
SCP loop records history outside this module.
"""

import warnings
from typing import TYPE_CHECKING, List, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config

from .acceptance_ratio import AcceptanceRatioAutotuner, AcceptanceRatioHyper

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..state import AlgorithmState, CandidateIterate


class AugmentedLagrangianHyper(AcceptanceRatioHyper):
    """Declared hyperparameters for :class:`AugmentedLagrangian`.

    Extends the shared :class:`AcceptanceRatioHyper` knobs with the
    constraint-violation multiplier-update parameters (``ep`` / ``eta_lambda``
    / ``lam_vc_max``). ``rho_init`` / ``rho_max`` are reserved for a future
    penalty-growth rule and are currently unused (see
    :meth:`AugmentedLagrangian.__init__`).
    """

    ep: float = 0.99
    eta_lambda: float = 1e1
    lam_vc_max: float = 1e5
    rho_init: float = 1.0
    rho_max: float = 1e2


class AugmentedLagrangian(AcceptanceRatioAutotuner):
    """Augmented Lagrangian autotuner.

    Uses an acceptance-ratio :math:`\\rho` between predicted and actual
    reduction in the nonlinear objective to drive a four-bucket update of
    the trust-region weight ``lam_prox`` (inherited from
    :class:`~openscvx.algorithms.autotuner.acceptance_ratio.AcceptanceRatioAutotuner`)
    and to update the virtual-control / virtual-buffer weights from constraint
    violations via the multiplier hook.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

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

        Every numeric knob is a declared hyperparameter: it lives on the frozen
        :class:`AugmentedLagrangianHyper` container at ``self.hyper`` and is a
        per-solve / batched override target (see :class:`AutotuningBase`).
        Knobs may still be read and written as plain attributes
        (``autotuner.lam_prox_max = 1e6``); the attribute proxy on
        :class:`AutotuningBase` routes the access into ``hyper``.

        Args:
            rho_init: Reserved for a future penalty-growth rule; currently
                unused. Passing a non-default value warns. Defaults to 1.0.
            rho_max: Reserved for a future penalty-growth rule; currently
                unused. Passing a non-default value warns. Defaults to 1e2.
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
            lam_cost_drop: Iteration after which cost relaxation applies
                (``state.k > lam_cost_drop``). ``-1`` relaxes from the first
                iteration; the default ``lam_cost_relax=1.0`` makes that a
                no-op. Defaults to -1.
            lam_cost_relax: Factor applied to lam_cost after lam_cost_drop.
                Defaults to 1.0.
        """
        if rho_init != 1.0 or rho_max != 1e2:
            warnings.warn(
                "AugmentedLagrangian rho_init/rho_max are reserved for a future "
                "penalty-growth rule and are currently unused — setting them has "
                "no effect on the solve.",
                UserWarning,
                stacklevel=2,
            )
        self.hyper = AugmentedLagrangianHyper(
            rho_init=rho_init,
            rho_max=rho_max,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            eta_0=eta_0,
            eta_1=eta_1,
            eta_2=eta_2,
            ep=ep,
            eta_lambda=eta_lambda,
            lam_vc_max=lam_vc_max,
            lam_prox_min=lam_prox_min,
            lam_prox_max=lam_prox_max,
            lam_cost_drop=lam_cost_drop,
            lam_cost_relax=lam_cost_relax,
        )

    # -----------------------------------------------------------------------
    # Multiplier (virtual-control / virtual-buffer) updates from violation
    # -----------------------------------------------------------------------

    def _update_multipliers(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        settings: Config,
        new_lam_prox: jnp.ndarray,
    ) -> dict:
        """Update the virtual-control and virtual-buffer weights from violation.

        Each weight follows the same piecewise (linear / quadratic) update keyed
        on the scaled constraint violation; the shared body gates the result by
        iterate acceptance.
        """
        candidate_x_prop = candidate.x_prop_plus[1:]
        return {
            "lam_vc": self._update_virtual_control_weights(
                state, candidate, candidate_x_prop, settings, new_lam_prox
            ),
            "lam_vb_nodal": self._update_virtual_buffer_nodal_weights(
                state, candidate, nodal_constraints, params, new_lam_prox
            ),
            "lam_vb_cross": self._update_virtual_buffer_cross_weights(
                state, candidate, nodal_constraints, params, new_lam_prox
            ),
        }

    def _update_virtual_control_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        candidate_x_prop: jnp.ndarray,
        settings: Config,
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
        scale = state.hyper.eta_lambda * (1.0 / (2.0 * lam_prox_scalar))
        case1 = state.lam_vc + nu * scale
        case2 = state.lam_vc + (nu**2) / state.hyper.ep * scale
        vc_new = jnp.where(nu > state.hyper.ep, case1, case2)
        return jnp.minimum(state.hyper.lam_vc_max, vc_new)

    def _update_virtual_buffer_nodal_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        lam_prox: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update virtual-buffer weights for nodal constraints.

        Evaluates each nodal constraint, computes the same piecewise update
        as for virtual control, and writes the result back into the
        ``(N, n_nodal)`` weight array via ``.at[...].set(...)``.
        """
        lam_vb_new = state.lam_vb_nodal
        lam_prox_scalar = jnp.max(lam_prox)
        scale = state.hyper.eta_lambda * (1.0 / (2.0 * lam_prox_scalar))

        for idx, constraint in enumerate(nodal_constraints.nodal):
            g = constraint.func(candidate.x, candidate.u, 0, params)
            nu = jnp.abs(g) if constraint.is_equality else jnp.maximum(0.0, g)

            if constraint.nodes is not None:
                nodes_array = jnp.asarray(constraint.nodes)
                nu_slice = nu[nodes_array]
                current = state.lam_vb_nodal[nodes_array, idx]
            else:
                nu_slice = nu
                current = state.lam_vb_nodal[:, idx]

            case1 = current + nu_slice * scale
            case2 = current + (nu_slice**2) / state.hyper.ep * scale
            updated = jnp.where(nu_slice > state.hyper.ep, case1, case2)

            if constraint.nodes is not None:
                lam_vb_new = lam_vb_new.at[nodes_array, idx].set(updated)
            else:
                lam_vb_new = lam_vb_new.at[:, idx].set(updated)

        # Intentional: the virtual-buffer weights share the virtual-control cap
        # ``lam_vc_max`` rather than a dedicated knob — one ceiling governs every
        # penalty weight in this autotuner.
        return jnp.minimum(state.hyper.lam_vc_max, lam_vb_new)

    def _update_virtual_buffer_cross_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        lam_prox: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update virtual-buffer weights for cross-node constraints."""
        lam_vb_new = state.lam_vb_cross
        lam_prox_scalar = jnp.max(lam_prox)
        scale = state.hyper.eta_lambda * (1.0 / (2.0 * lam_prox_scalar))

        for idx, constraint in enumerate(nodal_constraints.cross_node):
            g = constraint.func(candidate.x, candidate.u, params)
            nu = jnp.sum(jnp.abs(g) if constraint.is_equality else jnp.maximum(0.0, g))
            current = state.lam_vb_cross[idx]
            case1 = current + nu * scale
            case2 = current + (nu**2) / state.hyper.ep * scale
            updated = jnp.where(nu > state.hyper.ep, case1, case2)
            lam_vb_new = lam_vb_new.at[idx].set(updated)

        # Intentional: shares the virtual-control cap ``lam_vc_max`` — see
        # :py:meth:`_update_virtual_buffer_nodal_weights`.
        return jnp.minimum(state.hyper.lam_vc_max, lam_vb_new)

    def citation(self) -> List[str]:
        """Return BibTeX citations for the augmented-Lagrangian multiplier updates.

        Extends the inherited SCvx acceptance-ratio references with SCvx*,
        whose augmented-Lagrangian weight updates this autotuner implements.
        """
        return super().citation() + [
            r"""@inproceedings{oguri2023scvxstar,
  title={Successive Convexification with Feasibility Guarantee via Augmented
    Lagrangian for Non-Convex Optimal Control Problems},
  author={Oguri, Kenshiro},
  booktitle={2023 62nd IEEE Conference on Decision and Control (CDC)},
  pages={3296--3302},
  year={2023},
  publisher={IEEE}
}"""
        ]


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
