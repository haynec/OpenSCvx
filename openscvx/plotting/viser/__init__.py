"""Composable 3D visualization primitives using viser.

This module provides building blocks for creating interactive 3D trajectory
visualizations. The design philosophy is to give you useful primitives that
you can mix and match - not a monolithic plotting function that tries to
handle every case.

**Basic Pattern**:
    1. Create a ``viser.ViserServer`` (use our helper, or make your own!)
    2. Add static scene elements (obstacles, gates, ground planes, etc.)
    3. Add animated elements - each returns ``(handle, update_callback)``
    4. Wire up animation controls with your list of update callbacks
    5. Call ``server.sleep_forever()`` to keep the visualization running

**Example - Building Your Own Visualization**::

    from openscvx.plotting import viser

    # Step 1: Create server (or just use viser.ViserServer() directly!)
    server = viser.create_server(positions)

    # Step 2: Add static elements
    viser.add_gates(server, gate_vertices)
    viser.add_ellipsoid_obstacles(server, centers, radii)
    viser.add_ghost_trajectory(server, positions, colors)

    # Step 3: Add animated elements (collect the update callbacks)
    _, update_trail = viser.add_animated_trail(server, positions, colors)
    _, update_marker = viser.add_position_marker(server, positions)
    _, update_thrust = viser.add_thrust_vector(server, positions, thrust)

    # Step 4: Wire up animation controls
    viser.add_animation_controls(
        server, time_array,
        [update_trail, update_marker, update_thrust]
    )

    # Step 5: Keep server running
    server.sleep_forever()

**Available Primitives**:
    - Server: ``create_server``, ``compute_velocity_colors``, ``compute_grid_size``
    - Static: ``add_gates``, ``add_ellipsoid_obstacles``, ``add_glideslope_cone``,
      ``add_ghost_trajectory``
    - Animated: ``add_animated_trail``, ``add_position_marker``, ``add_thrust_vector``,
      ``add_attitude_frame``, ``add_viewcone``, ``add_target_marker(s)``
    - Plotly: ``add_animated_plotly_marker``, ``add_animated_vector_norm_plot``
    - SCP iteration: ``add_scp_animation_controls``, ``add_scp_iteration_nodes``, etc.

For problem-specific examples (drones with viewcones, rockets with glideslope
constraints, etc.), see ``examples/plotting_viser.py``.
"""

# Server setup
# Animated components
from .animated import (
    UpdateCallback,
    add_animated_trail,
    add_animation_controls,
    add_attitude_frame,
    add_position_marker,
    add_target_marker,
    add_target_markers,
    add_thrust_vector,
    add_viewcone,
)

# Plotly integration
from .plotly_integration import (
    add_animated_plotly_marker,
    add_animated_plotly_vline,
    add_animated_vector_norm_plot,
)

# Static primitives
from .primitives import (
    add_ellipsoid_obstacles,
    add_gates,
    add_ghost_trajectory,
    add_glideslope_cone,
)
from .orbits import add_circular_orbit, create_hohmann_transfer_server

# SCP iteration visualization
from .scp import (
    add_scp_animation_controls,
    add_scp_ghost_iterations,
    add_scp_iteration_attitudes,
    add_scp_iteration_nodes,
    add_scp_propagation_lines,
    extract_propagation_positions,
)
from .server import compute_grid_size, compute_velocity_colors, create_server

__all__ = [
    # Server
    "create_server",
    "compute_velocity_colors",
    "compute_grid_size",
    # Static primitives
    "add_gates",
    "add_ellipsoid_obstacles",
    "add_glideslope_cone",
    "add_ghost_trajectory",
    # Orbit helpers
    "add_circular_orbit",
    "create_hohmann_transfer_server",
    # Animated components
    "UpdateCallback",
    "add_animated_trail",
    "add_position_marker",
    "add_target_marker",
    "add_target_markers",
    "add_thrust_vector",
    "add_attitude_frame",
    "add_viewcone",
    # Animation controls
    "add_animation_controls",
    # Plotly integration
    "add_animated_plotly_marker",
    "add_animated_plotly_vline",
    "add_animated_vector_norm_plot",
    # SCP visualization
    "add_scp_iteration_nodes",
    "add_scp_iteration_attitudes",
    "add_scp_ghost_iterations",
    "extract_propagation_positions",
    "add_scp_propagation_lines",
    "add_scp_animation_controls",
]
