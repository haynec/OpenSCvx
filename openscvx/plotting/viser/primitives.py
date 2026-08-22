"""Static scene primitives for viser visualization.

Functions for adding non-animated elements: obstacles, gates, constraint cones,
ghost trajectories, etc. Called once during scene setup.
"""

import numpy as np
import viser


def _generate_ellipsoid_mesh(
    center: np.ndarray,
    radii: np.ndarray,
    axes: np.ndarray | None = None,
    subdivisions: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ellipsoid mesh vertices and faces via icosphere subdivision.

    Args:
        center: Center position (3,)
        radii: Radii along each principal axis (3,)
        axes: Rotation matrix (3, 3) defining principal axes. If None, uses identity.
        subdivisions: Number of icosphere subdivisions (higher = smoother)

    Returns:
        Tuple of (vertices, faces) where vertices is (V, 3) and faces is (F, 3)
    """
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    icosahedron_verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    # Normalize to unit sphere
    icosahedron_verts /= np.linalg.norm(icosahedron_verts[0])

    icosahedron_faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    vertices = icosahedron_verts
    faces = icosahedron_faces

    # Subdivide faces
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache = {}

        def get_midpoint(i1: int, i2: int) -> int:
            """Get or create midpoint vertex between two vertices."""
            key = (min(i1, i2), max(i1, i2))
            if key in midpoint_cache:
                return midpoint_cache[key]

            nonlocal vertices
            p1, p2 = vertices[i1], vertices[i2]
            mid = (p1 + p2) / 2.0
            mid = mid / np.linalg.norm(mid)  # Project onto unit sphere

            idx = len(vertices)
            vertices = np.vstack([vertices, mid])
            midpoint_cache[key] = idx
            return idx

        for tri in faces:
            v0, v1, v2 = tri
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            new_faces.extend([[v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]])

        faces = np.array(new_faces, dtype=np.int32)

    # Scale by radii to create ellipsoid
    vertices = vertices / radii

    # Rotate by principal axes if provided
    if axes is not None:
        vertices = vertices @ axes.T

    # Translate to center
    vertices = vertices + center

    return vertices.astype(np.float32), faces


def add_ellipsoid_obstacles(
    server: viser.ViserServer,
    centers: list[np.ndarray],
    radii: list[np.ndarray],
    axes: list[np.ndarray] | None = None,
    color: tuple[int, int, int] = (255, 100, 100),
    opacity: float = 0.6,
    wireframe: bool = False,
    subdivisions: int = 2,
) -> list:
    """Add ellipsoidal obstacles to the scene.

    Args:
        server: ViserServer instance
        centers: List of center positions, each shape (3,)
        radii: List of radii along principal axes, each shape (3,)
        axes: List of rotation matrices (3, 3) defining principal axes.
            If None, ellipsoids are axis-aligned.
        color: RGB color tuple
        opacity: Opacity (0-1), only used when wireframe=False
        wireframe: If True, render as wireframe instead of solid
        subdivisions: Icosphere subdivisions (higher = smoother, 2 is usually good)

    Returns:
        List of mesh handles
    """
    handles = []

    if axes is None:
        axes = [None] * len(centers)

    for i, (center, rad, ax) in enumerate(zip(centers, radii, axes)):
        # Convert JAX arrays to numpy if needed
        center = np.asarray(center, dtype=np.float64)
        rad = np.asarray(rad, dtype=np.float64)
        if ax is not None:
            ax = np.asarray(ax, dtype=np.float64)

        vertices, faces = _generate_ellipsoid_mesh(center, rad, ax, subdivisions)

        handle = server.scene.add_mesh_simple(
            f"/obstacles/ellipsoid_{i}",
            vertices=vertices,
            faces=faces,
            color=color,
            wireframe=wireframe,
            opacity=opacity if not wireframe else 1.0,
        )
        handles.append(handle)

    return handles


def add_gates(
    server: viser.ViserServer,
    vertices: list,
    color: tuple[int, int, int] = (255, 165, 0),
    line_width: float = 3.0,
) -> None:
    """Add gate/obstacle wireframes to the scene.

    Args:
        server: ViserServer instance
        vertices: List of vertex arrays (4 vertices for planar gate, 8 for box)
        color: RGB color tuple
        line_width: Line width for wireframe
    """
    for i, verts in enumerate(vertices):
        verts = np.array(verts)
        n_verts = len(verts)

        if n_verts == 4:
            # Planar gate: 4 vertices forming a closed loop
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        elif n_verts == 8:
            # 3D box: 8 vertices
            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # front face
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # back face
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # connecting edges
            ]
        else:
            # Unknown format, skip
            continue

        # Shape (N, 2, 3) for N line segments
        points = np.array([[verts[e[0]], verts[e[1]]] for e in edges])
        server.scene.add_line_segments(
            f"/gates/gate_{i}",
            points=points,
            colors=color,
            line_width=line_width,
        )


def _generate_cone_mesh(
    apex: np.ndarray,
    height: float,
    half_angle_deg: float,
    n_segments: int = 32,
    axis: np.ndarray | tuple = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cone mesh with apex at given position, opening along specified axis.

    Args:
        apex: Apex position (3,) - the tip of the cone
        height: Height of the cone (extends along axis direction from apex)
        half_angle_deg: Half-angle of the cone from the axis in degrees
        n_segments: Number of segments around the circumference
        axis: Unit vector direction the cone opens toward (default +Z)

    Returns:
        Tuple of (vertices, faces) where vertices is (V, 3) and faces is (F, 3)
    """
    half_angle_rad = np.radians(half_angle_deg)
    base_radius = height * np.tan(half_angle_rad)

    # Normalize axis
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    # Build orthonormal basis: find two vectors perpendicular to axis
    # Pick a reference vector not parallel to axis
    if abs(axis[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Gram-Schmidt to get perpendicular vectors
    u = ref - np.dot(ref, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    # Vertices: apex + base circle points
    vertices = [apex.copy()]  # Apex at index 0

    # Base center position
    base_center = apex + height * axis

    # Base circle vertices
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        offset = base_radius * (np.cos(angle) * u + np.sin(angle) * v)
        vertices.append(base_center + offset)

    # Center of base for closing the bottom
    vertices.append(base_center.copy())  # Index n_segments + 1

    vertices = np.array(vertices, dtype=np.float32)

    # Faces: triangles from apex to base edge pairs, plus base cap
    faces = []

    # Side faces (apex to each edge of base)
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Triangle: apex, base[i], base[next_i]
        faces.append([0, i + 1, next_i + 1])

    # Base cap faces (center to each edge)
    base_center_idx = n_segments + 1
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Triangle: center, base[next_i], base[i] (reverse winding for outward normal)
        faces.append([base_center_idx, next_i + 1, i + 1])

    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


def add_glideslope_cone(
    server: viser.ViserServer,
    apex: np.ndarray | tuple = (0.0, 0.0, 0.0),
    height: float = 2000.0,
    glideslope_angle_deg: float = 86.0,
    axis: np.ndarray | tuple = (0.0, 0.0, 1.0),
    color: tuple[int, int, int] = (100, 200, 100),
    opacity: float = 0.2,
    wireframe: bool = False,
    n_segments: int = 32,
    *,
    name: str = "/constraints/glideslope_cone",
) -> viser.MeshHandle:
    """Add a glideslope/approach constraint cone to the scene.

    The glideslope constraint typically has the form:
        ||position_perp|| <= tan(angle) * position_along_axis

    This creates a cone with apex at the target, opening along the specified axis.

    Args:
        server: ViserServer instance
        apex: Apex position (docking/landing site), default is origin
        height: Height of the cone visualization
        glideslope_angle_deg: Glideslope angle in degrees (measured from axis).
            For constraint ||r_perp|| <= tan(theta) * r_axis, pass theta here.
            Common values: 86 deg (very wide), 70 deg (moderate), 45 deg (steep)
        axis: Unit vector direction the cone opens toward. Default (0,0,1) for +Z.
            Use (-1,0,0) for R-bar approach (from below in radial direction).
        color: RGB color tuple
        opacity: Opacity (0-1)
        wireframe: If True, render as wireframe
        n_segments: Number of segments for cone smoothness
        name: Scene path for this cone. Pass a distinct name per cone when a
            scene shows more than one (e.g. a wide approach cone and a tight
            terminal cone).

    Returns:
        Mesh handle for the cone
    """
    apex = np.asarray(apex, dtype=np.float32)

    vertices, faces = _generate_cone_mesh(apex, height, glideslope_angle_deg, n_segments, axis=axis)

    handle = server.scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=faces,
        color=color,
        wireframe=wireframe,
        opacity=opacity if not wireframe else 1.0,
    )

    return handle


def add_ghost_trajectory(
    server: viser.ViserServer,
    pos: np.ndarray,
    colors: np.ndarray,
    opacity: float = 0.3,
    line_width: float = 2.0,
    *,
    name: str = "/ghost_traj",
) -> viser.LineSegmentsHandle:
    """Add a faint ghost trajectory showing the full path.

    The path is drawn as a polyline: consecutive positions are joined by line
    segments, each shaded by the dimmed colors of its endpoints. Lines read as a
    continuous path at any camera distance, where a point cloud of the same
    samples breaks up (and does not render at all in some browsers).

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        colors: RGB color array of shape (N, 3)
        opacity: Opacity factor (0-1) applied to colors
        line_width: Width of the path in pixels
        name: Scene path for this trajectory. Pass a distinct name per vehicle
            when more than one shares a server.

    Returns:
        Line segments handle for the path
    """
    pos = np.asarray(pos)
    ghost_colors = (np.asarray(colors) * opacity).astype(np.uint8)
    return server.scene.add_line_segments(
        name,
        points=np.stack([pos[:-1], pos[1:]], axis=1),
        colors=np.stack([ghost_colors[:-1], ghost_colors[1:]], axis=1),
        line_width=line_width,
    )
