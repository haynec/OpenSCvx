"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

from openscvx.config import Config

from .base import AutotuningBase

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from .base import AlgorithmState, CandidateIterate
    from .weights import Weights


class ConstantProximalWeight(AutotuningBase):
    """Constant Proximal Weight method.

    This method keeps the trust region weight constant throughout the optimization,
    while still updating virtual control weights and handling cost relaxation.
    Useful when you want a fixed trust region size without adaptation.
    """

    def __init__(
        self,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.lam_cost_drop = lam_cost_drop
        self.lam_cost_relax = lam_cost_relax

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
        weights: "Weights",
    ) -> str:
        """Update SCP weights keeping trust region constant.

        Args:
            state: Solver state containing current weight values (mutated in place)
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters
            weights: Initial weights from the algorithm

        Returns:
            str: Adaptive state string (e.g., "Accept", "Reject")
        """
        # Update cost relaxation parameter after cost_drop iterations.
        # When lam_cost is a per-state array, scalar lam_cost_relax scales
        # uniformly, preserving the user-specified per-state weight ratios.
        if state.k > self.lam_cost_drop:
            candidate.lam_cost = state.lam_cost * self.lam_cost_relax
        else:
            candidate.lam_cost = weights.lam_cost

        candidate.lam_prox = state.lam_prox
        state.accept_solution(candidate)
        return "Accept Constant"
