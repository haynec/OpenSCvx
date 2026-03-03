import copy
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from openscvx.algorithms import OptimizationResults
from openscvx.config import Config
from openscvx.discretization import Discretizer
from openscvx.utils import calculate_cost_from_boundaries

from .propagation import s_to_t, simulate_nonlinear_time, t_to_tau


def propagate_trajectory_results(
    params: dict,
    settings: Config,
    result: OptimizationResults,
    propagation_solver: callable,
    dynamics_discrete: Optional[Callable] = None,
    algebraic_prop: Optional[dict] = None,
    discretizer: Optional[Discretizer] = None,
) -> OptimizationResults:
    """Propagate the optimal trajectory and compute additional results.

    This function takes the optimal control solution and propagates it through the
    nonlinear dynamics to compute the actual state trajectory and other metrics.

    Args:
        params (dict): System parameters.
        settings (Config): Configuration settings.
        result (OptimizationResults): Optimization results object.
        propagation_solver (callable): Function for propagating the system state.
        dynamics_discrete (callable, optional): Discrete dynamics map used to apply
            node-wise impulsive/discrete updates before continuous propagation.
        algebraic_prop (dict, optional): Dictionary mapping output names to vmapped JAX functions.
        discretizer: Discretizer instance (used for ``dis_type``).
            Defaults to ``None`` which uses FOH.

    Returns:
        OptimizationResults: Updated results object containing:
            - t_full: Full time vector
            - x_full: Full state trajectory
            - u_full: Full control trajectory
            - cost: Computed cost
            - ctcs_violation: CTCS constraint violation
            - trajectory: Dict containing each variables values at full propagation fidelity
    """
    # Get arrays from result
    x = result.x
    u = result.u

    t = np.array(s_to_t(x, u, settings, discretizer)).squeeze()

    t_full = np.arange(t[0], t[-1], settings.prp.dt)

    tau_vals, u_full = t_to_tau(u, t_full, t, settings, discretizer)

    # Create a copy of x_prop for propagation to avoid mutating settings
    # Match free values from initial state to the initial value from the result
    x_prop_for_propagation = copy.copy(settings.sim.x_prop)

    # Only copy for states that exist in optimization (propagation may have extra states at the end)
    n_opt_states = x.shape[1]
    n_prop_states = settings.sim.x_prop.initial.shape[0]

    if n_opt_states == n_prop_states:
        # Same size - copy all
        # Use metadata from settings (immutable configuration)
        mask = jnp.array([t == "Free" for t in settings.sim.x.initial_type], dtype=bool)
        x_prop_for_propagation.initial = jnp.where(mask, x[0, :], settings.sim.x_prop.initial)
    else:
        # Propagation has extra states - only copy the overlapping portion
        # Use metadata from settings (immutable configuration)
        mask = jnp.array([t == "Free" for t in settings.sim.x.initial_type], dtype=bool)
        x_prop_initial_updated = settings.sim.x_prop.initial.copy()
        x_prop_initial_updated[:n_opt_states] = jnp.where(
            mask, x[0, :], settings.sim.x_prop.initial[:n_opt_states]
        )
        x_prop_for_propagation.initial = x_prop_initial_updated

    # Temporarily replace x_prop with our modified copy for propagation
    # Save original to restore after propagation
    original_x_prop = settings.sim.x_prop
    settings.sim.x_prop = x_prop_for_propagation

    try:
        x_full = simulate_nonlinear_time(
            params,
            x,
            u,
            tau_vals,
            t,
            settings,
            propagation_solver,
            dynamics_discrete=dynamics_discrete,
        )
    finally:
        # Always restore original x_prop, even if propagation fails
        settings.sim.x_prop = original_x_prop

    # Calculate cost using utility function and metadata from settings
    cost = calculate_cost_from_boundaries(x_full, settings.sim.x.initial_type, settings.sim.x.final_type)

    # Calculate CTCS constraint violation
    ctcs_violation = x_full[-1, settings.sim.ctcs_slice_prop]

    # Build trajectory dictionary with all states and controls
    trajectory_dict = {}

    # Add all states (user-defined and augmented)
    for state in result._states:
        trajectory_dict[state.name] = x_full[:, state._slice]

    # Add all controls (user-defined and augmented)
    for control in result._controls:
        trajectory_dict[control.name] = u_full[:, control._slice]

    # Compute algebraic outputs (vmapped over time)
    if algebraic_prop:
        for name, output_fn in algebraic_prop.items():
            # output_fn is vmapped: (T, n_x), (T, n_u), node, params -> (T, output_dim)
            # Pass node=0 since algebraic outputs shouldn't depend on node index
            output_values = output_fn(x_full, u_full, 0, params)
            trajectory_dict[name] = np.asarray(output_values)

    # Update the results object with post-processing data
    result.t_full = t_full
    result.x_full = x_full
    result.u_full = u_full
    result.cost = cost
    result.ctcs_violation = ctcs_violation
    result.trajectory = trajectory_dict

    return result
