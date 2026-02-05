"""Animated scene elements for viser visualization.

Each function in this module adds an animated element to a viser scene and
returns a tuple of ``(handle, update_callback)``. The update callback has
signature ``update_callback(frame_idx: int) -> None`` and updates the visual
to reflect the state at that frame index.

Collect these callbacks and pass them to ``add_animation_controls()`` to
wire up playback with GUI controls (play/pause, scrubber, speed, etc.).

Example::

    _, update_trail = add_animated_trail(server, positions, colors)
    _, update_marker = add_position_marker(server, positions)
    _, update_thrust = add_thrust_vector(server, positions, thrust, attitude)

    add_animation_controls(server, time_array, [update_trail, update_marker, update_thrust])
"""

import threading
import time
from typing import Callable

import numpy as np
import viser

# Type alias for update callbacks: fn(frame_idx: int) -> None
UpdateCallback = Callable[[int], None]


def add_animated_trail(
    server: viser.ViserServer,
    pos: np.ndarray,
    colors: np.ndarray,
    point_size: float = 0.15,
) -> tuple[viser.PointCloudHandle, UpdateCallback]:
    """Add an animated trail that grows with the animation.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        colors: RGB color array of shape (N, 3)
        point_size: Size of trail points

    Returns:
        Tuple of (handle, update_callback)
    """
    handle = server.scene.add_point_cloud(
        "/trail",
        points=pos[:1],
        colors=colors[:1],
        point_size=point_size,
    )

    def update(frame_idx: int) -> None:
        idx = frame_idx + 1  # Include current frame
        handle.points = pos[:idx]
        handle.colors = colors[:idx]

    return handle, update


def add_position_marker(
    server: viser.ViserServer,
    pos: np.ndarray,
    radius: float = 0.5,
    color: tuple[int, int, int] = (100, 200, 255),
) -> tuple[viser.IcosphereHandle, UpdateCallback]:
    """Add an animated position marker (sphere at current position).

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        radius: Marker radius
        color: RGB color tuple

    Returns:
        Tuple of (handle, update_callback)
    """
    handle = server.scene.add_icosphere(
        "/current_pos",
        radius=radius,
        color=color,
        position=pos[0],
    )

    def update(frame_idx: int) -> None:
        handle.position = pos[frame_idx]

    return handle, update


def add_target_marker(
    server: viser.ViserServer,
    target_pos: np.ndarray,
    name: str = "target",
    radius: float = 0.8,
    color: tuple[int, int, int] = (255, 50, 50),
    show_trail: bool = True,
    trail_color: tuple[int, int, int] | None = None,
) -> tuple[viser.IcosphereHandle, UpdateCallback | None]:
    """Add a viewplanning target marker (static or moving).

    Args:
        server: ViserServer instance
        target_pos: Target position - either shape (3,) for static or (N, 3) for moving
        name: Unique name for this target (used in scene path)
        radius: Marker radius
        color: RGB color tuple for marker
        show_trail: If True and target is moving, show trajectory trail
        trail_color: RGB color for trail (defaults to dimmed marker color)

    Returns:
        Tuple of (handle, update_callback). update_callback is None for static targets.
    """
    target_pos = np.asarray(target_pos)

    # Check if static (single position) or moving (trajectory)
    is_moving = target_pos.ndim == 2 and target_pos.shape[0] > 1

    initial_pos = target_pos[0] if is_moving else target_pos

    # Add marker
    handle = server.scene.add_icosphere(
        f"/targets/{name}/marker",
        radius=radius,
        color=color,
        position=initial_pos,
    )

    # For moving targets, optionally show trail
    if is_moving and show_trail:
        if trail_color is None:
            trail_color = tuple(int(c * 0.5) for c in color)
        server.scene.add_point_cloud(
            f"/targets/{name}/trail",
            points=target_pos,
            colors=trail_color,
            point_size=0.1,
        )

    if not is_moving:
        # Static target - no update needed
        return handle, None

    def update(frame_idx: int) -> None:
        # Clamp to valid range for target trajectory
        idx = min(frame_idx, len(target_pos) - 1)
        handle.position = target_pos[idx]

    return handle, update


def add_target_markers(
    server: viser.ViserServer,
    target_positions: list[np.ndarray],
    colors: list[tuple[int, int, int]] | None = None,
    radius: float = 0.8,
    show_trails: bool = True,
) -> list[tuple[viser.IcosphereHandle, UpdateCallback | None]]:
    """Add multiple viewplanning target markers.

    Args:
        server: ViserServer instance
        target_positions: List of target positions, each either (3,) or (N, 3)
        colors: List of RGB colors, one per target. Defaults to distinct colors.
        radius: Marker radius
        show_trails: If True, show trails for moving targets

    Returns:
        List of (handle, update_callback) tuples
    """
    # Default colors if not provided
    if colors is None:
        default_colors = [
            (255, 50, 50),  # Red
            (50, 255, 50),  # Green
            (50, 50, 255),  # Blue
            (255, 255, 50),  # Yellow
            (255, 50, 255),  # Magenta
            (50, 255, 255),  # Cyan
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(target_positions))]

    results = []
    for i, (pos, color) in enumerate(zip(target_positions, colors)):
        handle, update = add_target_marker(
            server,
            pos,
            name=f"target_{i}",
            radius=radius,
            color=color,
            show_trail=show_trails,
        )
        results.append((handle, update))

    return results


def _rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz format).

    Args:
        v: Vector of shape (3,)
        q: Quaternion of shape (4,) in [w, x, y, z] format

    Returns:
        Rotated vector of shape (3,)
    """
    w, x, y, z = q
    # Quaternion rotation: v' = q * v * q_conj
    # Using the formula for rotating a vector by a quaternion
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def add_thrust_vector(
    server: viser.ViserServer,
    pos: np.ndarray,
    thrust: np.ndarray | None,
    attitude: np.ndarray | None = None,
    scale: float = 0.3,
    color: tuple[int, int, int] = (255, 100, 100),
    line_width: float = 4.0,
) -> tuple[viser.LineSegmentsHandle | None, UpdateCallback | None]:
    """Add an animated thrust/force vector visualization.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        thrust: Thrust/force array of shape (N, 3), or None to skip
        attitude: Quaternion array of shape (N, 4) in [w, x, y, z] format.
            If provided, thrust is assumed to be in body frame and will be
            rotated to world frame using the attitude.
        scale: Scale factor for thrust vector length
        color: RGB color tuple
        line_width: Line width

    Returns:
        Tuple of (handle, update_callback), or (None, None) if thrust is None
    """
    if thrust is None:
        return None, None

    def get_thrust_world(frame_idx: int) -> np.ndarray:
        """Get thrust vector in world frame."""
        thrust_body = thrust[frame_idx]
        if attitude is not None:
            return _rotate_vector_by_quaternion(thrust_body, attitude[frame_idx])
        return thrust_body

    thrust_world = get_thrust_world(0)
    thrust_end = pos[0] + thrust_world * scale
    handle = server.scene.add_line_segments(
        "/thrust_vector",
        points=np.array([[pos[0], thrust_end]]),  # Shape (1, 2, 3)
        colors=color,
        line_width=line_width,
    )

    def update(frame_idx: int) -> None:
        thrust_world = get_thrust_world(frame_idx)
        thrust_end = pos[frame_idx] + thrust_world * scale
        handle.points = np.array([[pos[frame_idx], thrust_end]])

    return handle, update


def add_attitude_frame(
    server: viser.ViserServer,
    pos: np.ndarray,
    attitude: np.ndarray | None,
    axes_length: float = 2.0,
    axes_radius: float = 0.05,
) -> tuple[viser.FrameHandle | None, UpdateCallback | None]:
    """Add an animated body coordinate frame showing attitude.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        attitude: Quaternion array of shape (N, 4) in [w, x, y, z] format, or None to skip
        axes_length: Length of the coordinate axes
        axes_radius: Radius of the axes cylinders

    Returns:
        Tuple of (handle, update_callback), or (None, None) if attitude is None
    """
    if attitude is None:
        return None, None

    # Viser uses wxyz quaternion format
    handle = server.scene.add_frame(
        "/body_frame",
        wxyz=attitude[0],
        position=pos[0],
        axes_length=axes_length,
        axes_radius=axes_radius,
    )

    def update(frame_idx: int) -> None:
        handle.wxyz = attitude[frame_idx]
        handle.position = pos[frame_idx]

    return handle, update


def _generate_viewcone_vertices(
    half_angle_x: float,
    half_angle_y: float | None,
    depth: float,
    norm_type: float | str,
    n_segments: int = 32,
) -> np.ndarray:
    """Generate viewcone vertices in sensor frame (apex at origin, pointing along +Z).

    The base cross-section follows the p-norm unit ball boundary (superellipse):
        ||[x/a, y/b]||_p = 1

    Args:
        half_angle_x: Half-angle in x direction (radians)
        half_angle_y: Half-angle in y direction (radians). If None, uses half_angle_x.
        depth: Depth/length of the cone
        norm_type: p-norm value (1, 2, 3, ..., or "inf"/float("inf") for infinity norm)
        n_segments: Number of segments around the boundary

    Returns:
        Vertices array of shape (N, 3) where first vertex is apex at origin
    """
    if half_angle_y is None:
        half_angle_y = half_angle_x

    # Compute base dimensions at the given depth
    base_half_x = depth * np.tan(half_angle_x)
    base_half_y = depth * np.tan(half_angle_y)

    vertices = [[0.0, 0.0, 0.0]]  # Apex at origin

    # Handle inf norm
    if norm_type == "inf" or norm_type == float("inf"):
        p = 100.0  # Large p approximates inf-norm
    else:
        p = float(norm_type)

    # Generate superellipse boundary points
    # Parameterization: x = sign(cos(t)) * |cos(t)|^(2/p), y = sign(sin(t)) * |sin(t)|^(2/p)
    for i in range(n_segments):
        theta = 2 * np.pi * i / n_segments
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Superellipse parameterization
        x = np.sign(cos_t) * (np.abs(cos_t) ** (2.0 / p)) * base_half_x
        y = np.sign(sin_t) * (np.abs(sin_t) ** (2.0 / p)) * base_half_y
        vertices.append([x, y, depth])

    return np.array(vertices, dtype=np.float32)


def _generate_viewcone_faces(n_base_vertices: int) -> np.ndarray:
    """Generate faces for a cone/pyramid mesh.

    Args:
        n_base_vertices: Number of vertices on the base (excluding apex)

    Returns:
        Faces array of shape (F, 3) with vertex indices
    """
    faces = []

    # Side faces: triangles from apex (index 0) to each edge of base
    # Winding: apex -> current -> next gives outward-facing normals (visible from outside)
    for i in range(n_base_vertices):
        current_i = i + 1
        next_i = (i + 1) % n_base_vertices + 1
        faces.append([0, current_i, next_i])

    # Base cap: triangulate as a fan from first base vertex
    # Winding for outward-facing normal (visible from outside/below the cone)
    for i in range(2, n_base_vertices):
        faces.append([1, i + 1, i])

    return np.array(faces, dtype=np.int32)


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (wxyz) to rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def add_viewcone(
    server: viser.ViserServer,
    pos: np.ndarray,
    attitude: np.ndarray | None,
    half_angle_x: float,
    half_angle_y: float | None = None,
    scale: float = 10.0,
    norm_type: float | str = 2,
    R_sb: np.ndarray | None = None,
    color: tuple[int, int, int] = (35, 138, 141),  # Viridis at t~0.33 (teal)
    opacity: float = 0.4,
    wireframe: bool = False,
    n_segments: int = 32,
) -> tuple[viser.MeshHandle | None, UpdateCallback | None]:
    """Add an animated viewcone mesh that matches p-norm constraints.

    The sensor is assumed to look along +Z in its own frame (boresight = [0,0,1]).
    The viewcone represents the constraint ||[x,y]||_p <= tan(alpha) * z.

    Cross-section shapes by norm:
        - p=1: diamond
        - p=2: circle/ellipse
        - p>2: rounded square (superellipse)
        - p=inf: square/rectangle

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        attitude: Quaternion array of shape (N, 4) in [w, x, y, z] format, or None to skip
        half_angle_x: Half-angle of the cone in x direction (radians).
            For symmetric cones, this is pi/alpha_x where alpha_x is the constraint parameter.
        half_angle_y: Half-angle in y direction (radians). If None, uses half_angle_x.
            For asymmetric constraints, this is pi/alpha_y.
        scale: Depth/length of the cone visualization
        norm_type: p-norm value (1, 2, 3, ..., or "inf" for infinity norm)
        R_sb: Body-to-sensor rotation matrix (3x3). If None, sensor is aligned with body z-axis.
        color: RGB color tuple
        opacity: Mesh opacity (0-1), ignored if wireframe=True
        wireframe: If True, render as wireframe instead of solid
        n_segments: Number of segments for cone smoothness

    Returns:
        Tuple of (handle, update_callback), or (None, None) if attitude is None
    """
    if attitude is None:
        return None, None

    # Convert inputs to numpy arrays (handles JAX arrays)
    pos = np.asarray(pos, dtype=np.float64)
    attitude = np.asarray(attitude, dtype=np.float64)
    if R_sb is not None:
        R_sb = np.asarray(R_sb, dtype=np.float64)

    # Generate base geometry in sensor frame
    base_vertices = _generate_viewcone_vertices(
        half_angle_x, half_angle_y, scale, norm_type, n_segments
    )
    n_base_verts = len(base_vertices) - 1  # Exclude apex
    faces = _generate_viewcone_faces(n_base_verts)

    # Sensor-to-body rotation (transpose of body-to-sensor)
    R_sensor_to_body = R_sb.T if R_sb is not None else np.eye(3)

    def transform_vertices(frame_idx: int) -> np.ndarray:
        """Transform cone vertices from sensor frame to world frame."""
        # Get body-to-world rotation from attitude quaternion
        q_body = attitude[frame_idx]
        R_body_to_world = _quaternion_to_rotation_matrix(q_body)

        # Full transform: sensor -> body -> world
        R_sensor_to_world = R_body_to_world @ R_sensor_to_body

        # Transform vertices and translate to position
        world_vertices = (R_sensor_to_world @ base_vertices.T).T + pos[frame_idx]
        return world_vertices.astype(np.float32)

    # Create initial mesh
    initial_vertices = transform_vertices(0)
    handle = server.scene.add_mesh_simple(
        "/viewcone_mesh",
        vertices=initial_vertices,
        faces=faces,
        color=color,
        wireframe=wireframe,
        opacity=opacity if not wireframe else 1.0,
    )

    def update(frame_idx: int) -> None:
        handle.vertices = transform_vertices(frame_idx)

    return handle, update


# =============================================================================
# Animation Controls
# =============================================================================


def add_animation_controls(
    server: viser.ViserServer,
    traj_time: np.ndarray,
    update_callbacks: list[UpdateCallback],
    loop: bool = True,
    folder_name: str = "Animation",
) -> None:
    """Add animation GUI controls and start the animation loop.

    Creates play/pause button, reset button, time slider, speed slider, and loop checkbox.
    Runs animation in a background daemon thread.

    Args:
        server: ViserServer instance
        traj_time: Time array of shape (N,) with timestamps for each frame
        update_callbacks: List of update functions to call each frame
        loop: Whether to loop animation by default
        folder_name: Name for the GUI folder
    """
    traj_time = traj_time.flatten()
    n_frames = len(traj_time)
    t_start, t_end = float(traj_time[0]), float(traj_time[-1])
    duration = t_end - t_start

    # Filter out None callbacks
    callbacks = [cb for cb in update_callbacks if cb is not None]

    def time_to_frame(t: float) -> int:
        """Convert simulation time to frame index."""
        return int(np.clip(np.searchsorted(traj_time, t, side="right") - 1, 0, n_frames - 1))

    def update_all(sim_t: float) -> None:
        """Update all visualization components."""
        idx = time_to_frame(sim_t)
        for callback in callbacks:
            callback(idx)

    # --- GUI Controls ---
    with server.gui.add_folder(folder_name):
        play_button = server.gui.add_button("Play")
        reset_button = server.gui.add_button("Reset")
        time_slider = server.gui.add_slider(
            "Time (s)",
            min=t_start,
            max=t_end,
            step=duration / 100,
            initial_value=t_start,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop)

    # Animation state
    state = {"playing": False, "sim_time": t_start}

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]
        play_button.name = "Pause" if state["playing"] else "Play"

    @reset_button.on_click
    def _(_) -> None:
        state["sim_time"] = t_start
        time_slider.value = t_start
        update_all(t_start)

    @time_slider.on_update
    def _(_) -> None:
        if not state["playing"]:
            state["sim_time"] = float(time_slider.value)
            update_all(state["sim_time"])

    def animation_loop() -> None:
        """Background thread for realtime animation playback."""
        last_time = time.time()
        while True:
            time.sleep(0.016)  # ~60 fps
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if state["playing"]:
                # Advance simulation time (speed=1.0 is realtime)
                state["sim_time"] += dt * speed_slider.value

                if state["sim_time"] >= t_end:
                    if loop_checkbox.value:
                        state["sim_time"] = t_start
                    else:
                        state["sim_time"] = t_end
                        state["playing"] = False
                        play_button.name = "Play"

                time_slider.value = state["sim_time"]
                update_all(state["sim_time"])

    # Start animation thread
    thread = threading.Thread(target=animation_loop, daemon=True)
    thread.start()
