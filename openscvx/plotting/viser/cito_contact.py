"""Viser primitives for CITO contact forces, friction cones, and impulses."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import viser

from openscvx.plotting.viser.animated import UpdateCallback
from openscvx.plotting.viser.primitives import _generate_cone_mesh

# Type alias matching animated.py
UpdateCallbackList = List[UpdateCallback]


def _effective_dfoh_channels(
    trajectory: dict[str, np.ndarray],
    zoh_key: str,
    foh_key: str,
    n_frames: int,
) -> np.ndarray | None:
    """Sum ZOH + FOH dFOH channels along the trajectory (missing key → zero)."""
    zoh = trajectory.get(zoh_key)
    foh = trajectory.get(foh_key)
    if zoh is None and foh is None:
        return None
    out = np.zeros((n_frames, 0), dtype=float)
    if zoh is not None:
        z = np.asarray(zoh, dtype=float).reshape(n_frames, -1)
        out = z if out.size == 0 else out + z
    if foh is not None:
        f = np.asarray(foh, dtype=float).reshape(n_frames, -1)
        out = f if out.size == 0 else out + f
    return out


def sample_contact_wrenches_from_trajectory(
    robot,
    trajectory: dict[str, np.ndarray],
    *,
    z_ground: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame contact positions and forces along a propagated trajectory.

    Returns:
        ``p_c`` (N, 3), ``f_world`` (N, 3), ``n_c`` (N, 3) unit normals (outward from ground).
    """
    from openscvx.integrations.frax_cito import contact_kinematics, contact_wrench_world

    q = np.asarray(trajectory["q"], dtype=float)
    qd = np.asarray(trajectory["qd"], dtype=float)
    n_frames = q.shape[0]

    phi_t = _effective_dfoh_channels(trajectory, "phi_t_zoh", "phi_t_foh", n_frames)
    phi_n = _effective_dfoh_channels(trajectory, "phi_n_zoh", "phi_n_foh", n_frames)
    if phi_t is None:
        phi_t = np.zeros((n_frames, 2), dtype=float)
    if phi_n is None:
        phi_n = np.zeros((n_frames, 1), dtype=float)

    p_c = np.zeros((n_frames, 3), dtype=float)
    f_world = np.zeros((n_frames, 3), dtype=float)
    n_c = np.zeros((n_frames, 3), dtype=float)

    for k in range(n_frames):
        kin = contact_kinematics(robot, q[k], qd[k], z_ground=z_ground)
        p_c[k] = np.asarray(kin["p_c"], dtype=float)
        n_c[k] = np.asarray(kin["n_c"], dtype=float)
        R_c = np.asarray(kin["R_c"], dtype=float)
        f_world[k] = np.asarray(
            contact_wrench_world(phi_t[k], phi_n[k, :1], R_c),
            dtype=float,
        ).reshape(3)

    return p_c, f_world, n_c


def sample_impulses_from_nodes(
    robot,
    nodes: dict[str, np.ndarray],
    node_times: np.ndarray,
    *,
    z_ground: float = 0.0,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Impulse events ``(t, p_c, f_imp_world)`` at SCvx nodes with nonzero ``Phi``."""
    from openscvx.integrations.frax_cito import contact_kinematics, contact_wrench_world

    q = np.asarray(nodes["q"], dtype=float)
    qd = np.asarray(nodes["qd"], dtype=float)
    n_nodes = q.shape[0]
    t_nodes = np.asarray(node_times, dtype=float).reshape(-1)
    if t_nodes.shape[0] != n_nodes:
        t_nodes = np.linspace(t_nodes[0], t_nodes[-1], n_nodes)

    phi_t_arr = nodes.get("Phi_t")
    phi_n_arr = nodes.get("Phi_n")
    phi_t = (
        np.asarray(phi_t_arr, dtype=float).reshape(n_nodes, -1)
        if phi_t_arr is not None
        else np.zeros((n_nodes, 2), dtype=float)
    )
    phi_n = (
        np.asarray(phi_n_arr, dtype=float).reshape(n_nodes, -1)
        if phi_n_arr is not None
        else np.zeros((n_nodes, 1), dtype=float)
    )

    events: list[tuple[float, np.ndarray, np.ndarray]] = []
    for k in range(n_nodes):
        kin = contact_kinematics(robot, q[k], qd[k], z_ground=z_ground)
        R_c = np.asarray(kin["R_c"], dtype=float)
        f_imp = np.asarray(
            contact_wrench_world(phi_t[k], phi_n[k, :1], R_c),
            dtype=float,
        ).reshape(3)
        if np.linalg.norm(f_imp) < 1e-9:
            continue
        events.append((float(t_nodes[k]), np.asarray(kin["p_c"], dtype=float), f_imp))

    return events


def _force_arrow_segments(
    origin: np.ndarray,
    force: np.ndarray,
    *,
    force_scale: float,
    min_length: float = 0.02,
) -> np.ndarray:
    """Segment with arrowhead at ``origin`` (contact); tail extends along ``+force``."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    force = np.asarray(force, dtype=float).reshape(3)
    mag = float(np.linalg.norm(force))
    if mag < 1e-12:
        length = min_length
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = force / mag
        length = max(min_length, mag * force_scale)
    tail = origin - direction * length
    return np.array([tail, origin], dtype=np.float32)


def _rgb_scale(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    a = float(np.clip(alpha, 0.0, 1.0))
    return tuple(int(c * a) for c in color)


def add_friction_cone_at_contact(
    server: viser.ViserServer,
    apex_traj: np.ndarray,
    normal_traj: np.ndarray,
    *,
    mu: float,
    cone_height: float = 0.12,
    name: str = "contact_0",
    color: tuple[int, int, int] = (255, 40, 40),
    opacity: float = 0.5,
    n_segments: int = 32,
) -> tuple[viser.MeshHandle, UpdateCallback]:
    """Red transparent friction cone: apex at contact, opens along ``-n_c``."""
    half_angle_deg = math.degrees(math.atan(float(mu)))
    apex_traj = np.asarray(apex_traj, dtype=float)
    normal_traj = np.asarray(normal_traj, dtype=float)

    def _mesh_at(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        apex = apex_traj[frame_idx]
        n = normal_traj[frame_idx]
        n_norm = np.linalg.norm(n)
        axis = -n / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, -1.0], dtype=float)
        return _generate_cone_mesh(
            apex.astype(np.float32),
            float(cone_height),
            half_angle_deg,
            n_segments=n_segments,
            axis=axis,
        )

    v0, f0 = _mesh_at(0)
    handle = server.scene.add_mesh_simple(
        f"/contact/{name}/friction_cone",
        vertices=v0,
        faces=f0,
        color=color,
        opacity=opacity,
    )

    def update(frame_idx: int) -> None:
        verts, _ = _mesh_at(frame_idx)
        handle.vertices = verts

    return handle, update


def add_contact_force_arrows(
    server: viser.ViserServer,
    apex_traj: np.ndarray,
    force_traj: np.ndarray,
    *,
    name: str = "contact_0",
    force_scale: float = 0.002,
    color: tuple[int, int, int] = (255, 30, 30),
) -> tuple[viser.ArrowsHandle, UpdateCallback]:
    """Red arrows for world-frame contact force (tip at contact, inside friction cone)."""
    apex_traj = np.asarray(apex_traj, dtype=float)
    force_traj = np.asarray(force_traj, dtype=float)
    n_frames = apex_traj.shape[0]

    def _points_at(frame_idx: int) -> np.ndarray:
        seg = _force_arrow_segments(
            apex_traj[frame_idx],
            force_traj[frame_idx],
            force_scale=force_scale,
        )
        return seg.reshape(1, 2, 3)

    handle = server.scene.add_arrows(
        f"/contact/{name}/force",
        _points_at(0),
        color,
        shaft_radius=0.006,
        head_radius=0.014,
        head_length=0.025,
    )

    def update(frame_idx: int) -> None:
        idx = min(frame_idx, n_frames - 1)
        handle.points = _points_at(idx)

    return handle, update


def add_impulse_arrows_with_fade(
    server: viser.ViserServer,
    time_vec: np.ndarray,
    impulse_events: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    fade_duration: float = 1.0,
    force_scale: float = 0.004,
    base_color: tuple[int, int, int] = (60, 120, 255),
    name: str = "contact_0",
) -> UpdateCallback:
    """Blue impulse arrows at contact; opacity fades linearly over ``fade_duration`` (s)."""
    time_vec = np.asarray(time_vec, dtype=float).reshape(-1)
    events = list(impulse_events)

    handle = server.scene.add_arrows(
        f"/contact/{name}/impulse",
        np.zeros((0, 2, 3), dtype=np.float32),
        base_color,
        shaft_radius=0.008,
        head_radius=0.018,
        head_length=0.03,
    )

    def update(frame_idx: int) -> None:
        t = time_vec[min(frame_idx, len(time_vec) - 1)]
        segments = []
        colors = []
        for t_imp, p_c, f_imp in events:
            if t < t_imp or t > t_imp + fade_duration:
                continue
            alpha = 1.0 - (t - t_imp) / fade_duration
            seg = _force_arrow_segments(p_c, f_imp, force_scale=force_scale)
            segments.append(seg)
            colors.append(_rgb_scale(base_color, alpha))

        if segments:
            handle.points = np.stack(segments, axis=0).astype(np.float32)
            handle.colors = np.asarray(colors, dtype=np.uint8)
            handle.visible = True
        else:
            handle.points = np.zeros((0, 2, 3), dtype=np.float32)
            handle.visible = False

    return update


def iterate_nodes_from_results(results, iter_idx: int) -> dict[str, np.ndarray]:
    """Nodal states and controls for one SCP iterate (including the initial guess)."""
    x = np.asarray(results.X[iter_idx], dtype=float)
    u = np.asarray(results.U[iter_idx], dtype=float)
    nodes: dict[str, np.ndarray] = {}
    for st in results._states:
        if st._slice is not None:
            nodes[st.name] = x[:, st._slice]
    for c in results._controls:
        if c._slice is not None:
            nodes[c.name] = u[:, c._slice]
    return nodes


def cito_trajectory_on_multishot_times(
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    t_ms: np.ndarray,
    nodes: dict[str, np.ndarray],
    t_nodes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Dense CITO trajectory for contact sampling along a multi-shoot time grid."""
    t_ms = np.asarray(t_ms, dtype=float).reshape(-1)
    t_nodes = np.asarray(t_nodes, dtype=float).reshape(-1)
    traj: dict[str, np.ndarray] = {
        "q": np.asarray(q_traj, dtype=float),
        "qd": np.asarray(qd_traj, dtype=float),
        "time": t_ms,
    }
    for key in ("phi_t_zoh", "phi_t_foh", "phi_n_zoh", "phi_n_foh"):
        if key not in nodes:
            continue
        u = np.asarray(nodes[key], dtype=float)
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        traj[key] = np.column_stack(
            [
                np.interp(t_ms, t_nodes, u[:, j], left=float(u[0, j]), right=float(u[-1, j]))
                for j in range(u.shape[1])
            ]
        )
    return traj


def refresh_cito_contact_view_state(
    view_state: dict,
    robot,
    *,
    z_ground: float = 0.0,
    enable_impulses: bool = False,
) -> None:
    """Re-sample contact wrenches (and impulses) after the active iterate changes."""
    view_state["p_c"], view_state["f_world"], view_state["n_c"] = sample_contact_wrenches_from_trajectory(
        robot, view_state["trajectory"], z_ground=z_ground
    )
    if enable_impulses and view_state.get("nodes") is not None:
        view_state["impulse_events"] = sample_impulses_from_nodes(
            robot,
            view_state["nodes"],
            view_state["node_times"],
            z_ground=z_ground,
        )
    else:
        view_state["impulse_events"] = []


def add_cito_contact_visualization_view_state(
    server: viser.ViserServer,
    robot,
    view_state: dict,
    *,
    mu: float,
    z_ground: float = 0.0,
    enable_impulses: bool = False,
    cone_height: float = 0.12,
    force_scale: float = 0.002,
    impulse_force_scale: float = 0.004,
    impulse_fade_s: float = 1.0,
) -> UpdateCallbackList:
    """Contact visuals driven by ``view_state`` (refreshed when the SCP iterate changes)."""
    refresh_cito_contact_view_state(
        view_state, robot, z_ground=z_ground, enable_impulses=enable_impulses
    )

    half_angle_deg = math.degrees(math.atan(float(mu)))

    def _cone_mesh(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = min(frame_idx, len(view_state["p_c"]) - 1)
        apex = view_state["p_c"][idx]
        n = view_state["n_c"][idx]
        n_norm = np.linalg.norm(n)
        axis = -n / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, -1.0], dtype=float)
        return _generate_cone_mesh(
            apex.astype(np.float32),
            float(cone_height),
            half_angle_deg,
            axis=axis,
        )

    v0, f0 = _cone_mesh(0)
    cone_handle = server.scene.add_mesh_simple(
        "/contact/foot/friction_cone",
        vertices=v0,
        faces=f0,
        color=(255, 40, 40),
        opacity=0.5,
    )

    def update_cone(frame_idx: int) -> None:
        verts, _ = _cone_mesh(frame_idx)
        cone_handle.vertices = verts

    force_handle = server.scene.add_arrows(
        "/contact/foot/force",
        _force_arrow_segments(
            view_state["p_c"][0],
            view_state["f_world"][0],
            force_scale=force_scale,
        ).reshape(1, 2, 3),
        (255, 30, 30),
        shaft_radius=0.006,
        head_radius=0.014,
        head_length=0.025,
    )

    def update_force(frame_idx: int) -> None:
        idx = min(frame_idx, len(view_state["p_c"]) - 1)
        force_handle.points = _force_arrow_segments(
            view_state["p_c"][idx],
            view_state["f_world"][idx],
            force_scale=force_scale,
        ).reshape(1, 2, 3)

    impulse_handle = server.scene.add_arrows(
        "/contact/foot/impulse",
        np.zeros((0, 2, 3), dtype=np.float32),
        (60, 120, 255),
        shaft_radius=0.008,
        head_radius=0.018,
        head_length=0.03,
    )

    def update_impulse(frame_idx: int) -> None:
        t_vec = np.asarray(view_state["trajectory"]["time"], dtype=float).reshape(-1)
        t = t_vec[min(frame_idx, len(t_vec) - 1)]
        segments = []
        colors = []
        for t_imp, p_c, f_imp in view_state.get("impulse_events", ()):
            if t < t_imp or t > t_imp + impulse_fade_s:
                continue
            alpha = 1.0 - (t - t_imp) / impulse_fade_s
            segments.append(_force_arrow_segments(p_c, f_imp, force_scale=impulse_force_scale))
            colors.append(_rgb_scale((60, 120, 255), alpha))
        if segments:
            impulse_handle.points = np.stack(segments, axis=0).astype(np.float32)
            impulse_handle.colors = np.asarray(colors, dtype=np.uint8)
            impulse_handle.visible = True
        else:
            impulse_handle.points = np.zeros((0, 2, 3), dtype=np.float32)
            impulse_handle.visible = False

    callbacks: UpdateCallbackList = [update_cone, update_force]
    if enable_impulses:
        callbacks.append(update_impulse)
    return callbacks


def add_cito_contact_visualization(
    server: viser.ViserServer,
    robot,
    trajectory: dict[str, np.ndarray],
    *,
    mu: float,
    z_ground: float = 0.0,
    enable_impulses: bool = False,
    nodes: dict[str, np.ndarray] | None = None,
    node_times: np.ndarray | None = None,
    cone_height: float = 0.12,
    force_scale: float = 0.002,
    impulse_force_scale: float = 0.004,
    impulse_fade_s: float = 1.0,
) -> UpdateCallbackList:
    """Add friction cone, contact-force arrows, and optional fading impulse arrows.

    Returns update callbacks to register with ``add_animation_controls``.
    """
    p_c, f_world, n_c = sample_contact_wrenches_from_trajectory(
        robot, trajectory, z_ground=z_ground
    )
    t_vec = np.asarray(trajectory["time"], dtype=float).reshape(-1)

    callbacks: UpdateCallbackList = []
    _, upd_cone = add_friction_cone_at_contact(
        server, p_c, n_c, mu=mu, cone_height=cone_height, name="foot"
    )
    callbacks.append(upd_cone)

    _, upd_force = add_contact_force_arrows(
        server, p_c, f_world, name="foot", force_scale=force_scale
    )
    callbacks.append(upd_force)

    if enable_impulses and nodes is not None and node_times is not None:
        events = sample_impulses_from_nodes(
            robot, nodes, node_times, z_ground=z_ground
        )
        if events:
            callbacks.append(
                add_impulse_arrows_with_fade(
                    server,
                    t_vec,
                    events,
                    fade_duration=impulse_fade_s,
                    force_scale=impulse_force_scale,
                    name="foot",
                )
            )

    return callbacks
