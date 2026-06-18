"""SCP iteration visualization components for viser.

This module contains functions for visualizing the successive convex programming
(SCP) optimization process, showing how the solution evolves across iterations.
"""

import threading
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import viser

# Type alias for update callbacks
UpdateCallback = Callable[[int], None]


def add_scp_iteration_nodes(
    server: viser.ViserServer,
    positions: list[np.ndarray],
    colors: list[tuple[int, int, int]] | None = None,
    point_size: float = 0.3,
    cmap_name: str = "viridis",
) -> tuple[list[viser.PointCloudHandle], UpdateCallback]:
    """Add animated optimization nodes that update per SCP iteration.

    Pre-buffers point clouds for all iterations and toggles visibility for performance.
    This avoids transmitting point data on every frame update.

    Args:
        server: ViserServer instance
        positions: List of position arrays per iteration, each shape (N, 3)
        colors: Optional list of RGB colors per iteration. If None, uses viridis colormap.
        point_size: Size of node markers
        cmap_name: Matplotlib colormap name (default: "viridis")

    Returns:
        Tuple of (list of point_handles, update_callback)
    """
    n_iterations = len(positions)

    # Default: use viridis colormap
    if colors is None:
        cmap = plt.get_cmap(cmap_name)
        colors = []
        for i in range(n_iterations):
            t = i / max(n_iterations - 1, 1)
            rgb = cmap(t)[:3]
            colors.append(tuple(int(c * 255) for c in rgb))

    # Convert colors to numpy arrays for viser compatibility
    colors_np = [np.array([c[0], c[1], c[2]], dtype=np.uint8) for c in colors]

    # Pre-create point clouds for all iterations (only first visible initially)
    handles = []
    for i in range(n_iterations):
        pos = np.asarray(positions[i], dtype=np.float32)
        handle = server.scene.add_point_cloud(
            f"/scp/nodes/iter_{i}",
            points=pos,
            colors=colors_np[i],
            point_size=point_size,
            visible=(i == 0),
        )
        handles.append(handle)

    # Track current visible iteration to minimize visibility toggles
    state = {"current_idx": 0}

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)
        if idx != state["current_idx"]:
            handles[state["current_idx"]].visible = False
            handles[idx].visible = True
            state["current_idx"] = idx

    return handles, update


def add_scp_iteration_attitudes(
    server: viser.ViserServer,
    positions: list[np.ndarray],
    attitudes: list[np.ndarray] | None,
    axes_length: float = 1.5,
    axes_radius: float = 0.03,
    stride: int = 1,
) -> tuple[list[viser.FrameHandle], UpdateCallback | None]:
    """Add animated attitude frames at each node that update per SCP iteration.

    Args:
        server: ViserServer instance
        positions: List of position arrays per iteration, each shape (N, 3)
        attitudes: List of quaternion arrays per iteration, each shape (N, 4) in wxyz format.
            If None, returns empty list and None callback.
        axes_length: Length of coordinate frame axes
        axes_radius: Radius of axes cylinders
        stride: Show attitude frame every `stride` nodes (1 = all nodes)

    Returns:
        Tuple of (list of frame handles, update_callback)
    """
    if attitudes is None:
        return [], None

    n_iterations = len(positions)
    n_nodes = len(positions[0])

    # Create frame handles for nodes at stride intervals
    node_indices = list(range(0, n_nodes, stride))
    handles = []

    for i, node_idx in enumerate(node_indices):
        handle = server.scene.add_frame(
            f"/scp/attitudes/frame_{i}",
            wxyz=attitudes[0][node_idx],
            position=positions[0][node_idx],
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        handles.append(handle)

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)
        pos = positions[idx]
        att = attitudes[idx]

        for i, node_idx in enumerate(node_indices):
            # Handle case where number of nodes changes between iterations
            if node_idx < len(pos) and node_idx < len(att):
                handles[i].position = pos[node_idx]
                handles[i].wxyz = att[node_idx]

    return handles, update


def add_scp_ghost_iterations(
    server: viser.ViserServer,
    positions: list[np.ndarray],
    point_size: float = 0.15,
    cmap_name: str = "viridis",
) -> tuple[list[viser.PointCloudHandle], UpdateCallback]:
    """Add ghost trails showing all previous SCP iterations.

    Pre-buffers point clouds for all iterations and toggles visibility for performance.
    Shows all previous iterations with viridis coloring to visualize convergence.

    Args:
        server: ViserServer instance
        positions: List of position arrays per iteration, each shape (N, 3)
        point_size: Size of ghost points
        cmap_name: Matplotlib colormap name for ghost colors

    Returns:
        Tuple of (list of handles, update_callback)
    """
    n_iterations = len(positions)
    cmap = plt.get_cmap(cmap_name)

    # Pre-create point clouds for all iterations with their colors
    # (all initially hidden, shown progressively as ghosts)
    handles = []
    for i in range(n_iterations):
        t = i / max(n_iterations - 1, 1)
        rgb = cmap(t)[:3]
        color = np.array([int(c * 255) for c in rgb], dtype=np.uint8)
        pos = np.asarray(positions[i], dtype=np.float32)

        handle = server.scene.add_point_cloud(
            f"/scp/ghosts/iter_{i}",
            points=pos,
            colors=color,
            point_size=point_size,
            visible=False,  # All start hidden
        )
        handles.append(handle)

    # Track which iterations are currently visible as ghosts
    state = {"visible_up_to": -1}

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)
        # Ghosts are iterations 0 through idx-1 (everything before current)
        new_visible_up_to = idx - 1

        if new_visible_up_to != state["visible_up_to"]:
            # Show/hide only the iterations that changed
            if new_visible_up_to > state["visible_up_to"]:
                # Show newly visible ghosts
                for i in range(state["visible_up_to"] + 1, new_visible_up_to + 1):
                    handles[i].visible = True
            else:
                # Hide ghosts that should no longer be visible
                for i in range(new_visible_up_to + 1, state["visible_up_to"] + 1):
                    handles[i].visible = False
            state["visible_up_to"] = new_visible_up_to

    return handles, update


def extract_propagation_positions(
    discretization_history: list[np.ndarray],
    n_x: int,
    n_u: int,
    position_slice: slice,
    scene_scale: float = 1.0,
) -> list[list[np.ndarray]]:
    """Extract 3D position trajectories from discretization history.

    The discretization history contains the multi-shot integration results.
    Each V matrix has shape (flattened_size, n_timesteps) where:
    - flattened_size = (N-1) * segment_size
    - segment_size = n_x + n_x*n_x + 2*n_x*n_u (state + STM + control influence matrices)
    - n_timesteps = number of integration substeps

    Args:
        discretization_history: List of V matrices from each SCP iteration
        n_x: Number of states
        n_u: Number of controls
        position_slice: Slice for extracting position from state vector
        scene_scale: Divide positions by this factor for visualization

    Returns:
        List of propagation trajectories per iteration.
        Each iteration contains a list of (n_substeps, 3) arrays, one per segment.
    """
    from openscvx.algorithms.multishot import unpack_multishot_V

    if not discretization_history:
        return []

    propagations = []
    for V in discretization_history:
        n_segments = V.shape[0] // (n_x + n_x * n_x + 2 * n_x * n_u)
        placeholder_t = np.linspace(0.0, 1.0, n_segments + 1)
        prop = unpack_multishot_V(
            V, n_x=n_x, n_u=n_u, t_nodes=placeholder_t, states=()
        )
        iteration_segments = [
            seg[:, position_slice] / scene_scale for seg in prop.segments()
        ]
        propagations.append(
            [np.asarray(seg, dtype=np.float32) for seg in iteration_segments]
        )

    return propagations


def add_scp_propagation_lines(
    server: viser.ViserServer,
    propagations: list[list[np.ndarray]],
    line_width: float = 2.0,
    cmap_name: str = "viridis",
) -> tuple[list, UpdateCallback]:
    """Add animated nonlinear propagation lines that update per SCP iteration.

    Shows the actual integrated trajectory between optimization nodes,
    revealing defects (gaps) in early iterations that close as SCP converges.
    All iterations up to the current one are shown with viridis coloring,
    similar to ghost iterations for nodes.

    Args:
        server: ViserServer instance
        propagations: List of propagation trajectories per iteration from
            extract_propagation_positions(). Each iteration contains a list
            of (n_substeps, 3) position arrays, one per segment.
        line_width: Width of propagation lines
        cmap_name: Matplotlib colormap name (default: "viridis")

    Returns:
        Tuple of (list of line handles, update_callback)
    """
    if not propagations:
        return [], lambda _: None

    n_iterations = len(propagations)
    n_segments = len(propagations[0])
    cmap = plt.get_cmap(cmap_name)

    # Pre-compute colors for each iteration
    iteration_colors = []
    for i in range(n_iterations):
        t = i / max(n_iterations - 1, 1)
        rgb = cmap(t)[:3]
        iteration_colors.append(np.array([int(c * 255) for c in rgb], dtype=np.uint8))

    # Create line handles for each (iteration, segment) pair
    # Structure: handles[iter_idx][seg_idx]
    all_handles = []

    for iter_idx in range(n_iterations):
        iter_handles = []
        color = iteration_colors[iter_idx]

        for seg_idx in range(n_segments):
            seg_pos = propagations[iter_idx][seg_idx]  # Shape (n_substeps, 3)

            if len(seg_pos) < 2:
                iter_handles.append(None)
                continue

            # Create line segments connecting consecutive substeps
            segments = np.array(
                [[seg_pos[i], seg_pos[i + 1]] for i in range(len(seg_pos) - 1)],
                dtype=np.float32,
            )

            handle = server.scene.add_line_segments(
                f"/scp/propagation/iter_{iter_idx}/segment_{seg_idx}",
                points=segments,
                colors=color,
                line_width=line_width,
                visible=(iter_idx == 0),  # Only first iteration visible initially
            )
            iter_handles.append(handle)

        all_handles.append(iter_handles)

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)

        # Show all iterations up to and including current, hide the rest
        for i in range(n_iterations):
            should_show = i <= idx
            for handle in all_handles[i]:
                if handle is not None:
                    handle.visible = should_show

    return all_handles, update


def add_scp_animation_controls(
    server: viser.ViserServer,
    n_iterations: int,
    update_callbacks: list[UpdateCallback],
    autoplay: bool = False,
    frame_duration_ms: int = 500,
    folder_name: str = "SCP Animation",
) -> None:
    """Add GUI controls for stepping through SCP iterations.

    Creates play/pause button, step buttons, iteration slider, and speed control.

    Args:
        server: ViserServer instance
        n_iterations: Total number of SCP iterations
        update_callbacks: List of update functions to call each iteration
        autoplay: Whether to start playing automatically
        frame_duration_ms: Default milliseconds per iteration frame
        folder_name: Name for the GUI folder
    """
    # Filter out None callbacks
    callbacks = [cb for cb in update_callbacks if cb is not None]

    def update_all(iter_idx: int) -> None:
        """Update all visualization components."""
        for callback in callbacks:
            callback(iter_idx)

    # --- GUI Controls ---
    with server.gui.add_folder(folder_name):
        play_button = server.gui.add_button("Play")
        with server.gui.add_folder("Step Controls", expand_by_default=False):
            prev_button = server.gui.add_button("< Previous")
            next_button = server.gui.add_button("Next >")
        iter_slider = server.gui.add_slider(
            "Iteration",
            min=0,
            max=n_iterations - 1,
            step=1,
            initial_value=0,
        )
        speed_slider = server.gui.add_slider(
            "Speed (ms/iter)",
            min=50,
            max=2000,
            step=50,
            initial_value=frame_duration_ms,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    # Animation state
    state = {"playing": autoplay, "iteration": 0, "needs_update": True}

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]
        state["needs_update"] = True  # Trigger immediate update on play
        play_button.name = "Pause" if state["playing"] else "Play"

    @prev_button.on_click
    def _(_) -> None:
        if state["iteration"] > 0:
            state["iteration"] -= 1
            iter_slider.value = state["iteration"]
            update_all(state["iteration"])

    @next_button.on_click
    def _(_) -> None:
        if state["iteration"] < n_iterations - 1:
            state["iteration"] += 1
            iter_slider.value = state["iteration"]
            update_all(state["iteration"])

    @iter_slider.on_update
    def _(_) -> None:
        if not state["playing"]:
            state["iteration"] = int(iter_slider.value)
            update_all(state["iteration"])

    def animation_loop() -> None:
        """Background thread for SCP iteration playback."""
        last_update = time.time()
        while True:
            time.sleep(0.016)  # ~60 fps check rate

            # Handle immediate update requests (e.g., on play button click)
            if state["needs_update"]:
                state["needs_update"] = False
                last_update = time.time()
                update_all(state["iteration"])
                continue

            if state["playing"]:
                current_time = time.time()
                elapsed_ms = (current_time - last_update) * 1000

                if elapsed_ms >= speed_slider.value:
                    last_update = current_time
                    state["iteration"] += 1

                    if state["iteration"] >= n_iterations:
                        if loop_checkbox.value:
                            state["iteration"] = 0
                        else:
                            state["iteration"] = n_iterations - 1
                            state["playing"] = False
                            play_button.name = "Play"

                    iter_slider.value = state["iteration"]
                    update_all(state["iteration"])

    # Start animation thread
    thread = threading.Thread(target=animation_loop, daemon=True)
    thread.start()

    # Initial update to ensure first frame is fully rendered
    update_all(0)
