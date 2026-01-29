"""Trajectory visualization and plotting utilities.

This module provides reusable building blocks for visualizing trajectory
optimization results. It is intentionally minimal - we provide common utilities
that can be composed together, not a complete solution that tries to do
everything for you.

**2D Plots** (plotly-based):
    Two-layer API for time series visualization::

        from openscvx.plotting import plot_states, plot_controls, plot_vector_norm

        # High-level: subplot grid with individual scaling per component
        plot_states(results, ["position", "velocity"]).show()
        plot_controls(results, ["thrust"]).show()

        # Low-level: single component
        plot_state_component(results, "position", component=2).show()  # z only

        # Specialized plots
        plot_vector_norm(results, "thrust", bounds=(rho_min, rho_max)).show()
        plot_projections_2d(results, velocity_var_name="velocity").show()

**3D Visualization** (viser-based):
    The ``viser`` submodule provides composable primitives for building
    interactive 3D visualizations. See ``openscvx.plotting.viser`` for details::

        from openscvx.plotting import viser
        server = viser.create_server(positions)
        viser.add_gates(server, gate_vertices)
        server.sleep_forever()

For problem-specific visualization examples (drones, rockets, etc.), see
``examples/plotting_viser.py``.
"""

from . import viser
from .plotting import (
    plot_control_component,
    plot_controls,
    plot_projections_2d,
    plot_state_component,
    plot_states,
    plot_trust_region_heatmap,
    plot_vector_norm,
    plot_virtual_control_heatmap,
)
from .scp_iteration import plot_scp_convergence_histories, plot_scp_iterations

__all__ = [
    # 2D plotting functions (plotly)
    "plot_state_component",
    "plot_states",
    "plot_control_component",
    "plot_controls",
    "plot_projections_2d",
    "plot_vector_norm",
    "plot_trust_region_heatmap",
    "plot_virtual_control_heatmap",
    "plot_scp_iterations",
    "plot_scp_convergence_histories",
    # 3D visualization submodule (viser)
    "viser",
]
