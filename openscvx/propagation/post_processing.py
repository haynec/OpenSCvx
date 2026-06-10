import copy
from typing import Callable, Optional

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
    n_times: Optional[int] = None,
) -> OptimizationResults:
    """Propagate the optimal trajectory and compute additional results.

    This function takes the optimal control solution and propagates it through the
    nonlinear dynamics to compute the actual state trajectory and other metrics.

    When ``states_prop`` includes propagation-only states (e.g. via ``dynamics_prop`` /
    ``states_prop``), ``x_full`` has shape ``(n_times, n_prop_states)`` with
    ``n_prop_states > n_opt_states``. The discrete dynamics and cost use only the
    optimization-state portion; propagation-only states are preserved from the last
    propagated step and included in ``trajectory``.

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
        n_times (int, optional): When provided, build the dense output time grid as
            ``np.linspace(t[0], t[-1], n_times)`` instead of the default
            ``np.arange(t[0], t[-1], settings.prp.dt)``.  Useful when multiple
            trajectories must be stacked into a uniform array (e.g. the output of
            :meth:`~openscvx.problem.Problem.post_process_batched`).

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

    # Build dense output times.
    if n_times is not None:
        t_full = np.linspace(t[0], t[-1], n_times)
    else:
        # Default: step at prp.dt and always include the exact terminal time so
        # that trajectory[..., -1] corresponds to the true final state.
        t_full = np.arange(t[0], t[-1], settings.prp.dt)
        if t_full.size == 0 or not np.isclose(t_full[-1], t[-1]):
            t_full = np.concatenate([t_full, np.array([t[-1]])])

    tau_vals, u_full = t_to_tau(u, t_full, t, settings, discretizer)

    # Create a copy of x_prop for propagation to avoid mutating settings.
    x_prop_for_propagation = copy.copy(settings.sim.x_prop)

    n_opt_states = x.shape[1]
    n_prop_states = settings.sim.x_prop.initial.shape[0]

    # Seed from x[0, :] directly so param-fixed "Fix" components use this solve's
    # pinned value rather than the lowering-time x_prop.initial default.
    x0_opt = np.array(x[0, :], dtype=float)

    # Seed from the pre-impulse state to avoid impulse duplication
    if dynamics_discrete is not None and np.any(settings.sim.u._impulsive_mask()):
        init_fixed = np.asarray(settings.sim.x.initial_type) == "Fix"
        x_initial = np.asarray(settings.sim.x.initial, dtype=float)
        x0_opt = np.where(init_fixed, x_initial, x0_opt)

    if n_opt_states == n_prop_states:
        x_prop_for_propagation.initial = x0_opt
    else:
        # Propagation has extra states appended beyond the optimisation states;
        # copy only the overlapping prefix and leave prop-only states untouched.
        x_prop_initial_updated = np.array(settings.sim.x_prop.initial, dtype=float)
        x_prop_initial_updated[:n_opt_states] = x0_opt
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
    # dynamics_discrete operates on optimization states only; when propagation has
    # extra states, pass only the opt-state portion and then reattach the prop-only tail
    x_minus = np.asarray(x_full[-1, :n_opt_states])
    x_plus = np.asarray(
        dynamics_discrete(
            x_minus,
            np.asarray(u[-1]),
            int(settings.sim.n - 1),
            params,
        )
    ).reshape(-1)
    if n_prop_states > n_opt_states:
        # Preserve propagation-only states (not updated by discrete dynamics)
        full_final = np.concatenate([x_plus, np.asarray(x_full[-1, n_opt_states:])], axis=0)
    else:
        full_final = x_plus
    x_for_cost = np.concatenate([x_full[:-1], full_final[None, :]], axis=0)

    cost = calculate_cost_from_boundaries(
        x_for_cost[:, :n_opt_states],
        settings.sim.x.initial_type,
        settings.sim.x.final_type,
    )

    # Calculate CTCS constraint violation (use state after final impulse when applicable)
    if dynamics_discrete is not None and np.any(settings.sim.u._impulsive_mask()):
        ctcs_violation = full_final[settings.sim.ctcs_slice_prop]
    else:
        ctcs_violation = x_full[-1, settings.sim.ctcs_slice_prop]

    # Build trajectory dictionary with all states and controls.
    # result._states is states_prop (opt + propagation-only); each state._slice
    # indexes into the full propagation state, so propagation-only states are included.
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
