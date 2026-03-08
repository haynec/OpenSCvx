"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

import numpy as np

from openscvx.config import Config

from .base import AutotuningBase

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from .base import AlgorithmState, CandidateIterate, Weights


class RampProximalWeight(AutotuningBase):
    """Ramp Proximal Weight method.

    This method ramps the proximal weight up linearly over the first few iterations,
    then keeps it constant.
    """

    def __init__(
        self,
        ramp_factor: float = 1.0,
        lam_prox_max: float = 1e3,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.ramp_factor = ramp_factor
        self.lam_prox_max = lam_prox_max
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
            weights: Normalized initial weights from the algorithm

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

        # Check if we're already at max before updating
        was_at_max = np.all(state.lam_prox >= self.lam_prox_max)

        # Calculate and append new value
        new_lam_prox = np.minimum(state.lam_prox * self.ramp_factor, self.lam_prox_max)
        state.lam_prox_history.append(new_lam_prox)

        # If we were already at max, or if we just reached it and it's staying constant
        if was_at_max:
            state.accept_solution(candidate)
            return "Accept Constant"
        else:
            state.accept_solution(candidate)
            return "Accept Higher"
