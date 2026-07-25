"""Shared visualization and export helpers for the compound-STC rocket examples.

Seven examples in this tree solve the same 6-DoF powered-descent problem under
compound state-triggered constraints — ``6DoF_pdg_stc.py``,
``6DoF_pdg_stc_ifthen.py``, and the five ``senss/6DoF_pdg_stc_senss*.py``
variants.  They differ in their *problem definitions*, which stay fully inline
in each file, but they all want the same picture afterwards: a nine-panel
Plotly figure whose bounds step down when a trigger fires, a speed-coloured
3-D path, an animated viser scene with a gimballed line-of-sight cone, and a
motion-platform CSV.

That picture lives here.  Everything is parameterised by two frozen records —
:class:`CstcLimits` (the constraint bounds and trigger thresholds, mirroring the
constant block near the top of each example) and :class:`CstcScales` (the
solver's length/mass normalization) — so the shared functions take arguments
rather than reaching back into a module's globals.

Import side effects: none.  ``tests/test_examples.py`` imports every file under
``examples/``, so nothing here reads a file or allocates at module scope.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import viser
import viser.transforms as vtf

from examples.plotting import qdcm
from openscvx.plotting.viser import (
    UpdateCallback,
    add_animation_controls,
    add_viewcone,
    create_server,
)

#: RGB colour of the line-of-sight cone and its altitude trigger disc.
LOS_COLOR = (80, 160, 255)

#: A scene marker: viser path, world position in metres, RGB colour, radius (viser units).
SiteMarker = tuple[str, np.ndarray, tuple[int, int, int], float]

#: An altitude trigger disc: height above the pad in metres, RGB colour, label.
AltitudeTrigger = tuple[float, tuple[int, int, int], str]


@dataclass(frozen=True)
class CstcScales:
    """The solver's non-dimensionalization: divide metres by ``r_scale``, kg by ``m_scale``."""

    r_scale: float
    m_scale: float


@dataclass(frozen=True, eq=False)
class CstcLimits:
    """Everything the cSTC plots need about a problem beyond its trajectory.

    The nine-panel figure draws each bound as two segments — the loose value up
    to the trigger crossing, the tight cSTC value after it — so it needs both
    halves of every bound plus the thresholds that decide where to switch.  The
    three geometry fields fix the cones the glideslope and line-of-sight angles
    are measured against.  Fields mirror the constant block in each example, in
    physical units (metres, m/s, degrees, radians, newtons, kilograms).
    """

    # Reference geometry
    gs_apex_m: np.ndarray  # glideslope cone vertex
    los_apex_m: np.ndarray  # line-of-sight cone target
    pad_z_m: float  # z-level "height above pad" is measured from

    # Altitude and phase triggers
    alt_trigger_h1_m: float
    alt_trigger_h2_m: float
    spd_stc_trig: float
    theta_stc_trig_deg: float

    # Thrust band, three-engine then single-engine
    t_max: float
    t_min: float
    t_max_aft: float
    t_min_aft: float

    # Loose bound / tight cSTC bound pairs
    v_stc_cons: float
    theta_max_deg: float
    theta_stc_deg: float
    delta_engine_max_deg: float
    delta_stc_deg: float
    w_b_max_rad_s: float
    omega_stc_rad_s: float
    gs_max_deg: float
    gs_stc_deg: float

    # Single-sided bounds
    delta_boresight_max_deg: float
    los_stc_deg: float
    m_dry: float


# ── Frames and angles ─────────────────────────────────────────────────────────


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Reorder quaternions from the model's ``[x, y, z, w]`` to viser's ``[w, x, y, z]``."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def los_body_to_sensor(elev: float, azimuth: float) -> np.ndarray:
    """Body-to-sensor DCM for the ``los_elev``/``los_az`` spherical gimbal.

    Rows are the sensor's x, y and boresight axes in body coordinates, so a
    neutral gimbal (both angles zero) gives the identity.
    """
    boresight = np.array(
        [np.sin(elev) * np.cos(azimuth), np.sin(elev) * np.sin(azimuth), np.cos(elev)],
        dtype=np.float64,
    )
    boresight /= np.linalg.norm(boresight) + 1e-12
    x_raw = np.cross([0.0, 0.0, 1.0], boresight)
    x_norm = np.linalg.norm(x_raw)
    x = np.array([1.0, 0.0, 0.0]) if x_norm < 1e-9 else x_raw / x_norm
    return np.stack([x, np.cross(boresight, x), boresight], axis=0)


def los_body_to_sensor_series(elev: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` gimbal rotations, one per trajectory sample."""
    return np.stack([los_body_to_sensor(e, a) for e, a in zip(np.ravel(elev), np.ravel(azimuth))])


def los_angle_deg(
    pos: np.ndarray,
    q_xyzw: np.ndarray,
    elev: np.ndarray,
    azimuth: np.ndarray,
    *,
    apex: np.ndarray,
) -> np.ndarray:
    """Angle between the apex-relative position and the sensor boresight, in degrees.

    This is the quantity the line-of-sight cSTC bounds: the constraint holds
    while the vehicle stays inside a cone of half-angle ``los_stc_deg`` about
    the boresight, seen from ``apex``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    apex = np.asarray(apex, dtype=np.float64)
    elev = np.ravel(elev)
    azimuth = np.ravel(azimuth)

    angles = np.zeros(pos.shape[0])
    for k in range(pos.shape[0]):
        los_b = np.array(
            [
                np.sin(elev[k]) * np.cos(azimuth[k]),
                np.sin(elev[k]) * np.sin(azimuth[k]),
                np.cos(elev[k]),
            ]
        )
        los_i = qdcm(xyzw_to_wxyz(q_xyzw[k])) @ los_b
        r_k = pos[k] - apex
        r_norm = np.linalg.norm(r_k)
        if r_norm > 1e-8:
            cos_val = np.dot(r_k, los_i) / (r_norm * (np.linalg.norm(los_i) + 1e-12))
            angles[k] = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
    return angles


def quat_xyzw_to_rpy(quat_xyzw: np.ndarray, *, degrees: bool = True) -> np.ndarray:
    """Convert ``[qx, qy, qz, qw]`` quaternions to roll, pitch, yaw.

    Extrinsic XYZ Euler angles (roll about x, pitch about y, yaw about z),
    matching ``CT-cSTC/CT-cSTC.ipynb``'s ``rotation_matrix``/``euler_to_quat``.
    """
    from scipy.spatial.transform import Rotation

    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if quat_xyzw.ndim == 1:
        quat_xyzw = quat_xyzw.reshape(1, 4)
    rpy = Rotation.from_quat(quat_xyzw).as_euler("XYZ")
    return np.degrees(rpy) if degrees else rpy


def tilt_deg(q_xyzw: np.ndarray) -> np.ndarray:
    """Tilt from vertical, in degrees, from the quaternion's ``[qx, qy]`` pair."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1.0, 1.0)))


# ── Trajectory preparation ────────────────────────────────────────────────────


def prepare_for_viser(result, *, scales: CstcScales, scene_scale: float) -> None:
    """Rewrite ``result`` in place into viser display units.

    The cSTC model is already ENU with z up, which is viser's convention, so
    positions only need unscaling to metres and dividing by ``scene_scale``;
    no axis permutation is involved.  The thrust direction is reconstructed
    from the polar control parameterisation so the plume can be drawn, and the
    attitude quaternion is reordered to ``[w, x, y, z]``.
    """
    traj = result.trajectory
    position_slice = slice(1, 4)  # mass occupies column 0 of the state vector

    traj["position"] = np.asarray(traj["position"], dtype=np.float64) * scales.r_scale / scene_scale
    traj["velocity"] = np.asarray(traj["velocity"], dtype=np.float64) * scales.r_scale
    traj["attitude"] = xyzw_to_wxyz(np.asarray(traj["attitude"], dtype=np.float64))

    thrust = np.asarray(traj["thrust_mag"], dtype=np.float64).flatten()
    elev = np.asarray(traj["gimbal_elev"], dtype=np.float64).flatten()
    azimuth = np.asarray(traj["gimbal_az"], dtype=np.float64).flatten()
    traj["thrust_body"] = np.stack(
        [
            thrust * np.sin(elev) * np.cos(azimuth),
            thrust * np.sin(elev) * np.sin(azimuth),
            thrust * np.cos(elev),
        ],
        axis=-1,
    )

    traj["los_elev"] = np.asarray(traj["los_elev"], dtype=np.float64).flatten()
    traj["los_az"] = np.asarray(traj["los_az"], dtype=np.float64).flatten()

    for i, X in enumerate(result.X):
        X = np.asarray(X, dtype=np.float64, copy=True)
        X[:, position_slice] *= scales.r_scale / scene_scale
        result.X[i] = X


def node_trigger_indices(
    nodes, limits: CstcLimits, *, scales: CstcScales
) -> tuple[int | None, int | None, int | None]:
    """First node index at which each altitude/thrust trigger activates.

    Returns ``(k_h1, k_h2, k_aft)``; an entry is ``None`` when the trigger
    never fires at a node, which the callers render as "not crossed at nodes".
    """
    pos_m = np.asarray(nodes["position"]) * scales.r_scale
    vel_ms = np.asarray(nodes["velocity"]) * scales.r_scale
    h_pad = pos_m[:, 2] - limits.pad_z_m
    speed = np.linalg.norm(vel_ms, axis=1)
    tilt = tilt_deg(np.asarray(nodes["attitude"]))

    def first(mask: np.ndarray) -> int | None:
        return int(np.argmax(mask)) if mask.any() else None

    return (
        first(h_pad < limits.alt_trigger_h1_m),
        first(h_pad < limits.alt_trigger_h2_m),
        first((speed < limits.spd_stc_trig) & (tilt < limits.theta_stc_trig_deg)),
    )


# ── Viser scene pieces ────────────────────────────────────────────────────────


def _horizontal_disc_mesh(
    center: np.ndarray, radius: float, *, n_segments: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle fan for a flat disc in the z = ``center[2]`` plane."""
    center = np.asarray(center, dtype=np.float32)
    angles = 2.0 * np.pi * np.arange(n_segments) / n_segments
    rim = center + radius * np.stack(
        [np.cos(angles), np.sin(angles), np.zeros(n_segments)], axis=-1
    )
    vertices = np.concatenate([center[None, :], rim], axis=0).astype(np.float32)
    faces = np.array(
        [[0, i + 1, (i + 1) % n_segments + 1] for i in range(n_segments)], dtype=np.int32
    )
    return vertices, faces


def add_altitude_trigger_discs(
    server,
    pos: np.ndarray,
    *,
    center_xy: tuple[float, float],
    base_z_m: float,
    scene_scale: float,
    triggers: Sequence[AltitudeTrigger],
) -> None:
    """Draw the horizontal surfaces where altitude-based cSTC phases activate.

    Triggers measure height above the pad, not above the world origin, so each
    disc sits at ``base_z_m + alt_m`` and is centred on ``center_xy`` (both in
    metres).  ``pos`` is the trajectory in viser units and only sets the radius.
    """
    cx, cy = center_xy[0] / scene_scale, center_xy[1] / scene_scale
    xy_extent = float(np.max(np.linalg.norm(pos[:, :2] - np.array([cx, cy]), axis=1)))
    radius = max(xy_extent * 1.25, 2.0)

    for alt_m, color, description in triggers:
        z = (base_z_m + alt_m) / scene_scale
        verts, faces = _horizontal_disc_mesh(np.array([cx, cy, z], dtype=np.float32), radius)
        server.scene.add_mesh_simple(
            f"/cstc_triggers/alt_{int(alt_m)}",
            vertices=verts,
            faces=faces,
            color=color,
            opacity=0.16,
        )

        rim = verts[1:]
        server.scene.add_line_segments(
            f"/cstc_triggers/alt_{int(alt_m)}_ring",
            points=np.stack(
                [[rim[i], rim[(i + 1) % len(rim)]] for i in range(len(rim))], axis=0
            ).astype(np.float32),
            colors=color,
            line_width=2.5,
        )
        server.scene.add_label(
            f"/cstc_triggers/alt_{int(alt_m)}_label",
            text=description,
            position=(cx + radius * 0.85, cy, z + 0.05),
        )
        server.scene.add_line_segments(
            f"/cstc_triggers/alt_{int(alt_m)}_stem",
            points=np.array(
                [[[cx - radius * 0.95, cy, base_z_m / scene_scale], [cx - radius * 0.95, cy, z]]],
                dtype=np.float32,
            ),
            colors=tuple(int(c * 0.7) for c in color),
            line_width=1.5,
        )


def add_los_viewcone(
    server,
    result,
    *,
    half_angle_deg: float,
    scale: float,
    color: tuple[int, int, int] = LOS_COLOR,
    name: str = "/los_viewcone",
) -> tuple[viser.FrameHandle | None, UpdateCallback | None]:
    """Animated line-of-sight cone for the gimballed boresight sensor.

    The cSTC constrains the vehicle to stay inside a cone about the boresight
    as seen *from the pad*, so the cone must open along sensor **-Z** while
    :func:`los_body_to_sensor` puts the boresight on **+Z**.  A 180° roll about
    the sensor's x-axis reconciles the two; because it is a proper rotation the
    triangle winding stays correct and the library primitive does the rest.
    """
    traj = result.trajectory
    boresight_flip = np.diag([1.0, -1.0, -1.0])
    R_sb = boresight_flip @ los_body_to_sensor_series(traj["los_elev"], traj["los_az"])
    half_angle = np.radians(half_angle_deg)
    return add_viewcone(
        server,
        np.asarray(traj["position"], dtype=np.float64),
        np.asarray(traj["attitude"], dtype=np.float64),
        half_angle,
        half_angle,
        scale,
        R_sb=R_sb,
        color=color,
        opacity=0.4,
        name=name,
    )


def add_site_markers(server, markers: Sequence[SiteMarker], *, scene_scale: float) -> None:
    """Drop labelled spheres on the landing site, LoS target, waypoints, and friends."""
    for name, position_m, color, radius in markers:
        server.scene.add_icosphere(
            name,
            radius=radius,
            color=color,
            position=tuple(float(v) for v in np.asarray(position_m) / scene_scale),
        )


def add_cstc_phase_markers(
    server,
    pos: np.ndarray,
    node_indices: tuple[int | None, int | None, int | None],
    *,
    n_nodes: int,
    limits: CstcLimits,
) -> None:
    """Mark the nodes where each cSTC phase starts, and explain them in the GUI.

    ``node_indices`` is what :func:`node_trigger_indices` returns and ``pos`` is
    the trajectory in viser units.  The altitude phases also get discs from
    :func:`add_altitude_trigger_discs`; the speed-and-tilt phase has no surface
    to draw, so the node marker is the only cue it ever fires.
    """
    k_h1, k_h2, k_aft = node_indices
    for k, color in [(k_aft, (255, 210, 50)), (k_h2, LOS_COLOR), (k_h1, (255, 80, 80))]:
        if k is None:
            continue
        k = int(np.clip(k, 0, len(pos) - 1))
        server.scene.add_icosphere(
            f"/phase_markers/k{k}",
            radius=0.14,
            color=color,
            position=tuple(float(v) for v in pos[k]),
        )

    def label(k: int | None) -> str:
        return f"k={k}" if k is not None else "not crossed at nodes"

    with server.gui.add_folder("cSTC Phase Boundaries"):
        server.gui.add_markdown(
            f"**Phase structure (N={n_nodes} nodes)**\n\n"
            f"**Altitude triggers** (horizontal discs):\n"
            f"- 🔵 h < {int(limits.alt_trigger_h2_m)} m → {label(k_h2)}: LOS boresight viewcone  \n"
            f"- 🔴 h < {int(limits.alt_trigger_h1_m)} m → {label(k_h1)}: tight terminal  \n\n"
            f"**Speed ∧ tilt trigger** (node marker only):\n"
            f"- 🟡 {label(k_aft)}: single-engine thrust "
            f"(||v||<{int(limits.spd_stc_trig)} ∧ θ<{int(limits.theta_stc_trig_deg)}°)  \n"
        )


def add_camera_fov_slider(
    server: viser.ViserServer, *, initial_deg: float, folder: str = "Camera"
) -> viser.GuiInputHandle:
    """Add a vertical field-of-view slider that drives every connected client's camera."""
    with server.gui.add_folder(folder):
        slider = server.gui.add_slider(
            "FOV (°)", min=5.0, max=120.0, step=1.0, initial_value=initial_deg
        )

    def apply(client: viser.ClientHandle) -> None:
        client.camera.fov = float(np.radians(slider.value))

    slider.on_update(lambda _e=None: [apply(c) for c in server.get_clients().values()])
    server.on_client_connect(apply)
    return slider


def sensor_fpv_pose(
    position: np.ndarray,
    attitude_wxyz: np.ndarray,
    R_sb: np.ndarray,
    *,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera pose that looks down the flipped LOS boresight (sensor -Z).

    Roll is locked to the sensor frame rather than to world up, so the image
    does not twist as the vehicle yaws; ``roll_deg`` rotates the image plane
    about the boresight on top of that.  Returns ``(position, wxyz)`` in
    viser's OpenCV camera convention (+X right, +Y down, +Z forward).
    """
    R_bw = qdcm(np.asarray(attitude_wxyz, dtype=np.float64))
    R_sensor_to_world = R_bw @ np.asarray(R_sb, dtype=np.float64).T

    forward = -R_sensor_to_world[:, 2]
    forward /= np.linalg.norm(forward) + 1e-12
    right = -R_sensor_to_world[:, 0]
    right -= forward * np.dot(right, forward)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-12

    phi = np.radians(float(roll_deg))
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    R_world_cam = np.stack(
        [cos_p * right + sin_p * down, -sin_p * right + cos_p * down, forward], axis=1
    )
    return np.asarray(position, dtype=np.float64), vtf.SO3.from_matrix(R_world_cam).wxyz


def create_sensor_fpv_server(
    result,
    traj_time: np.ndarray,
    *,
    markers: Sequence[SiteMarker],
    scene_scale: float,
    decorate_scene: Callable[[viser.ViserServer], None] | None = None,
    fov_deg: float = 76.0,
) -> viser.ViserServer:
    """Second viser window: the onboard LOS sensor's first-person view.

    The camera rides :func:`sensor_fpv_pose`, so what you see is exactly what
    the boresight sees and the blue cone in the main window frames.  Sliders
    offset the viewpoint in world ENU and roll the image about the boresight —
    both display-only, for lining the view up against reference footage.
    ``decorate_scene`` is called once on the fresh server to add whatever
    scenery the example wants (terrain, in the SENSS variants).
    """
    traj = result.trajectory
    pos = np.asarray(traj["position"], dtype=np.float64)
    attitude = np.asarray(traj["attitude"], dtype=np.float64)
    R_sb = los_body_to_sensor_series(traj["los_elev"], traj["los_az"])

    server = create_server(pos, dark_mode=True, show_grid=False)
    if decorate_scene is not None:
        decorate_scene(server)
    fov_slider = add_camera_fov_slider(server, initial_deg=fov_deg, folder="Sensor")
    add_site_markers(server, markers, scene_scale=scene_scale)

    state = {"idx": 0, "roll_deg": 0.0, "offset_m": (0.0, 0.0, 0.0)}

    def apply_camera(client: viser.ClientHandle, frame_idx: int) -> None:
        cam_pos, cam_wxyz = sensor_fpv_pose(
            pos[frame_idx], attitude[frame_idx], R_sb[frame_idx], roll_deg=state["roll_deg"]
        )
        cam_pos = cam_pos + np.asarray(state["offset_m"], dtype=np.float64) / scene_scale
        client.camera.position = tuple(float(v) for v in cam_pos)
        client.camera.wxyz = tuple(float(v) for v in cam_wxyz)
        client.camera.fov = float(np.radians(fov_slider.value))

    with server.gui.add_folder("Sensor FPV"):
        offset_sliders = [
            server.gui.add_slider(
                f"Position {axis} (m)", min=-500.0, max=500.0, step=1.0, initial_value=0.0
            )
            for axis in "XYZ"
        ]
        roll_slider = server.gui.add_slider(
            "Roll about boresight (°)", min=-180.0, max=180.0, step=1.0, initial_value=0.0
        )
        server.gui.add_markdown(
            "_Camera forward matches the blue LOS viewcone (sensor -Z). "
            "Position sliders offset the viewpoint in world ENU (m); roll rotates "
            "the image about the boresight. **Sensor → FOV** sets field of view._"
        )

        def sync_camera(_e=None) -> None:
            state["roll_deg"] = float(roll_slider.value)
            state["offset_m"] = tuple(float(s.value) for s in offset_sliders)
            for client in server.get_clients().values():
                apply_camera(client, state["idx"])

        for ctrl in (*offset_sliders, roll_slider):
            ctrl.on_update(sync_camera)

    def update_camera(frame_idx: int) -> None:
        state["idx"] = int(frame_idx)
        for client in server.get_clients().values():
            apply_camera(client, state["idx"])

    server.on_client_connect(lambda client: apply_camera(client, state["idx"]))
    add_animation_controls(server, traj_time, [update_camera], loop=True)
    return server


# ── Plotly figures and file export ────────────────────────────────────────────


def save_plotly_figure(fig, basename: str) -> None:
    """Write a Plotly figure as HTML and, when kaleido is installed, PNG and PDF."""
    fig.write_html(f"{basename}.html")
    print(f"  Saved {basename}.html")
    try:
        fig.write_image(f"{basename}.png", scale=2)
        fig.write_image(f"{basename}.pdf")
        print(f"  Saved {basename}.{{png,pdf}}")
    except Exception as exc:
        print(f"  Skipped PNG/PDF for {basename} ({exc}); install kaleido for static export.")


def apply_panel_dark_theme(fig) -> None:
    """Restyle a states/controls panel for dark backgrounds, in place.

    The panel's palette is chosen for print (``blue``/``green``/``purple`` on
    white); on a dark slide those read as muddy, so each is swapped for a
    lighter twin and the chrome is dimmed to match.
    """
    light_twin = {
        "blue": "#6eb5ff",
        "green": "#3ddc84",
        "purple": "#c084fc",
        "burlywood": "#e0b878",
    }

    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is not None and getattr(marker, "color", None) == "black":
            trace.marker.color = "#e8e8e8"
        line = getattr(trace, "line", None)
        if line is not None and getattr(line, "color", None) in light_twin:
            trace.line.color = light_twin[line.color]

    shapes = []
    for shape in fig.layout.shapes or ():
        shape = dict(shape.to_plotly_json() if hasattr(shape, "to_plotly_json") else shape)
        line = dict(shape.get("line") or {})
        line["color"] = light_twin.get(line.get("color"), line.get("color"))
        shape["line"] = line
        shapes.append(shape)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#1a1a1a",
        font={"color": "#e8e8e8"},
        shapes=shapes,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10, "color": "#e8e8e8"},
        },
    )
    axis_style = {
        "gridcolor": "#333333",
        "zerolinecolor": "#444444",
        "linecolor": "#666666",
        "tickfont": {"color": "#e8e8e8"},
        "title_font": {"color": "#e8e8e8"},
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)


def round_sigfigs(x: np.ndarray, sig: int = 5) -> np.ndarray:
    """Round to a fixed number of significant figures (zeros pass through)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nonzero = x != 0.0
    if np.any(nonzero):
        power = np.floor(np.log10(np.abs(x[nonzero])))
        scale = 10.0 ** (sig - 1 - power)
        out[nonzero] = np.round(x[nonzero] * scale) / scale
    return out


def write_trajectory_rpy_csv(
    path: str,
    t: np.ndarray,
    pos: np.ndarray,
    rpy: np.ndarray,
    *,
    header: str = "t,rx,ry,rz,phi,theta,psi",
) -> str:
    """Write a time/position/attitude table for the SENSS motion platform.

    The caller owns the axis and angle convention — each variant targets a
    different physical rig, and that remap is the one genuinely per-example
    thing in these files, so it stays visible in the example rather than
    hiding behind a flag here.  This function only rounds and writes.
    """
    data = round_sigfigs(np.column_stack([np.ravel(t), pos, rpy]))
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"  Saved {path}")
    return path


def resample_uniform(t: np.ndarray, columns: Sequence[np.ndarray], factor: int):
    """Stretch time by ``factor`` and interpolate to keep the original sample rate.

    Used by the CSV exporters that drive the platform slower than real time.
    Returns ``(t_dense, [column_dense, ...])``; ``factor == 1`` is a no-op.
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError(f"factor must be a positive integer >= 1, got {factor}")
    t = np.ravel(t) * float(factor)
    if factor == 1:
        return t, list(columns)
    t_dense = np.concatenate(
        [np.linspace(t[i], t[i + 1], factor, endpoint=False) for i in range(t.size - 1)] + [t[-1:]]
    )
    return t_dense, [
        np.column_stack([np.interp(t_dense, t, c[:, j]) for j in range(c.shape[1])])
        for c in columns
    ]


def plot_cstc_panel(result, limits: CstcLimits, *, scales: CstcScales):
    """Nine-panel Plotly figure of the states and controls against their cSTC bounds.

    Each bound is drawn as two segments joined at the moment the trajectory
    *actually* crosses the trigger threshold — not at a fixed node index — so
    what you see is the state-triggered behaviour the solver produced, and a
    bound that steps down before the state does is a real violation.

    Panels: thrust, speed, tilt, engine gimbal, angular velocity, glideslope,
    boresight deflection, mass, LOS angle.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Dense propagated trajectory, unscaled to physical units ───────────────
    traj = result.trajectory
    pos = np.asarray(traj["position"]) * scales.r_scale
    vel = np.asarray(traj["velocity"]) * scales.r_scale
    mass = np.asarray(traj["mass"]).flatten() * scales.m_scale
    q = np.asarray(traj["attitude"])
    omega = np.asarray(traj["angular_velocity"])
    thrust_N = np.asarray(traj["thrust_mag"]).flatten() * scales.m_scale * scales.r_scale
    gimbal = np.asarray(traj["gimbal_elev"]).flatten()
    boresight = np.asarray(traj["los_elev"]).flatten()
    boresight_az = np.asarray(traj["los_az"]).flatten()
    t_full = np.asarray(traj["time"]).flatten()

    # ── Node values, drawn as scatter over the propagated curves ──────────────
    nodes = result.nodes
    pos_n = np.asarray(nodes["position"]) * scales.r_scale
    vel_n = np.asarray(nodes["velocity"]) * scales.r_scale
    mass_n = np.asarray(nodes["mass"]).flatten() * scales.m_scale
    q_n = np.asarray(nodes["attitude"])
    omega_n = np.asarray(nodes["angular_velocity"])
    thrust_n_N = np.asarray(nodes["thrust_mag"]).flatten() * scales.m_scale * scales.r_scale
    gimbal_n = np.asarray(nodes["gimbal_elev"]).flatten()
    boresight_n = np.asarray(nodes["los_elev"]).flatten()
    boresight_az_n = np.asarray(nodes["los_az"]).flatten()
    t_nodes = np.asarray(nodes["time"]).flatten()

    # ── Derived quantities ────────────────────────────────────────────────────
    speed, speed_n = np.linalg.norm(vel, axis=1), np.linalg.norm(vel_n, axis=1)
    tilt, tilt_n = tilt_deg(q), tilt_deg(q_n)
    omega_dps = np.degrees(np.linalg.norm(omega, axis=1))
    omega_dps_n = np.degrees(np.linalg.norm(omega_n, axis=1))

    pos_rel, pos_rel_n = pos - limits.gs_apex_m, pos_n - limits.gs_apex_m
    r_xy = np.linalg.norm(pos_rel[:, :2], axis=1)
    r_xy_n = np.linalg.norm(pos_rel_n[:, :2], axis=1)
    gs_deg = 90.0 - np.degrees(np.arctan2(pos_rel[:, 2], r_xy + 1e-8))
    gs_deg_n = 90.0 - np.degrees(np.arctan2(pos_rel_n[:, 2], r_xy_n + 1e-8))

    los_ang = los_angle_deg(pos, q, boresight, boresight_az, apex=limits.los_apex_m)
    los_ang_n = los_angle_deg(pos_n, q_n, boresight_n, boresight_az_n, apex=limits.los_apex_m)

    # ── Trigger times from actual trajectory crossings ────────────────────────
    h_pad = pos[:, 2] - limits.pad_z_m

    def first_time(mask: np.ndarray) -> float:
        return float(t_full[int(np.argmax(mask))]) if mask.any() else float(t_full[-1])

    t_h1 = first_time(h_pad < limits.alt_trigger_h1_m)
    t_h2 = first_time(h_pad < limits.alt_trigger_h2_m)
    t_aft = first_time((speed < limits.spd_stc_trig) & (tilt < limits.theta_stc_trig_deg))
    t_end = float(t_nodes[-1])

    print(
        f"  Trigger times:  h<{int(limits.alt_trigger_h1_m)}m @ {t_h1:.2f}s   "
        f"h<{int(limits.alt_trigger_h2_m)}m @ {t_h2:.2f}s   "
        f"(v<{int(limits.spd_stc_trig)} & θ<{int(limits.theta_stc_trig_deg)}) @ {t_aft:.2f}s"
    )

    # ── Colour palette (matches the CT-cSTC notebook) ─────────────────────────
    c_node, c_state, c_h1, c_h2 = "black", "blue", "red", "orange"
    c_aft, c_trig, c_up, c_low = "lightseagreen", "burlywood", "green", "purple"

    fig = make_subplots(rows=3, cols=3, vertical_spacing=0.10, horizontal_spacing=0.08)
    legend_seen: set[str] = set()

    def once(name: str) -> bool:
        """True the first time a legend entry is requested (traces repeat per panel)."""
        if name in legend_seen:
            return False
        legend_seen.add(name)
        return True

    def seg(x0, x1, y, *, row, col, color, name=None):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y, y],
                mode="lines",
                line={"color": color, "dash": "dash", "width": 1.5},
                name=name,
                showlegend=once(name) if name else False,
                legendgroup=name,
            ),
            row=row,
            col=col,
        )

    def line(x, y, *, row, col, color, name=None):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": color, "width": 2},
                name=name,
                showlegend=once(name) if name else False,
                legendgroup=name,
            ),
            row=row,
            col=col,
        )

    def scatter(x, y, *, row, col, name="Node point"):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"color": c_node, "size": 7},
                name=name,
                showlegend=once(name),
                legendgroup=name,
            ),
            row=row,
            col=col,
        )

    def vline(x, *, row, col, color):
        fig.add_vline(x=x, line={"color": color, "dash": "dash", "width": 1.5}, row=row, col=col)

    # ── Thrust (row 1, col 1) — band narrows to one engine after the aft trigger
    seg(0, t_aft, limits.t_max * 1e-3, row=1, col=1, color=c_up, name="Upper bound")
    seg(0, t_aft, limits.t_min * 1e-3, row=1, col=1, color=c_low, name="Lower bound")
    seg(t_aft, t_end, limits.t_max_aft * 1e-3, row=1, col=1, color=c_up)
    seg(t_aft, t_end, limits.t_min_aft * 1e-3, row=1, col=1, color=c_low)
    vline(t_aft, row=1, col=1, color=c_aft)
    line(t_full, thrust_N * 1e-3, row=1, col=1, color="red", name="Control input")
    scatter(t_nodes, thrust_n_N * 1e-3, row=1, col=1)
    fig.update_yaxes(title_text="Thrust, T [kN]", row=1, col=1)

    # ── Speed (row 1, col 2) ──────────────────────────────────────────────────
    line(t_full, speed, row=1, col=2, color=c_state, name="State")
    scatter(t_nodes, speed_n, row=1, col=2)
    seg(0, t_end, limits.spd_stc_trig, row=1, col=2, color=c_trig, name="$v^{\\mathrm{trig}}$")
    vline(t_h1, row=1, col=2, color=c_h1)
    vline(t_aft, row=1, col=2, color=c_aft)
    seg(t_h1, t_end, limits.v_stc_cons, row=1, col=2, color=c_up, name="STC bound")
    fig.update_yaxes(title_text="Speed, ||v||₂ [m s⁻¹]", range=[0, speed.max() + 5], row=1, col=2)

    # ── Tilt (row 1, col 3) ───────────────────────────────────────────────────
    line(t_full, tilt, row=1, col=3, color=c_state)
    scatter(t_nodes, tilt_n, row=1, col=3)
    seg(0, t_h1, limits.theta_max_deg, row=1, col=3, color=c_up)
    seg(t_h1, t_end, limits.theta_stc_deg, row=1, col=3, color=c_up)
    seg(
        0,
        t_end,
        limits.theta_stc_trig_deg,
        row=1,
        col=3,
        color=c_trig,
        name="$\\theta^{\\mathrm{trig}}$",
    )
    vline(t_h1, row=1, col=3, color=c_h1)
    fig.update_yaxes(title_text="Tilt angle, θ [deg]", row=1, col=3)

    # ── Engine gimbal deflection (row 2, col 1) ───────────────────────────────
    line(t_full, np.degrees(gimbal), row=2, col=1, color="red")
    scatter(t_nodes, np.degrees(gimbal_n), row=2, col=1)
    seg(0, t_h1, limits.delta_engine_max_deg, row=2, col=1, color=c_up)
    seg(0, t_h1, -limits.delta_engine_max_deg, row=2, col=1, color=c_low)
    seg(t_h1, t_end, limits.delta_stc_deg, row=2, col=1, color=c_up)
    seg(t_h1, t_end, -limits.delta_stc_deg, row=2, col=1, color=c_low)
    vline(t_h1, row=2, col=1, color=c_h1)
    fig.update_yaxes(title_text="Engine gimbal, δᵉ [deg]", row=2, col=1)

    # ── Angular velocity (row 2, col 2) ───────────────────────────────────────
    line(t_full, omega_dps, row=2, col=2, color=c_state)
    scatter(t_nodes, omega_dps_n, row=2, col=2)
    seg(0, t_h1, np.degrees(limits.w_b_max_rad_s), row=2, col=2, color=c_up)
    seg(t_h1, t_end, np.degrees(limits.omega_stc_rad_s), row=2, col=2, color=c_up)
    vline(t_h1, row=2, col=2, color=c_h1)
    fig.update_yaxes(title_text="Angular velocity, ω_B [deg s⁻¹]", row=2, col=2)

    # ── Glideslope (row 2, col 3) — last sample sits at the apex, where γ is ill-defined
    gs_bound_pre = 90.0 - limits.gs_max_deg
    gs_bound_stc = 90.0 - limits.gs_stc_deg
    line(t_full[:-1], gs_deg[:-1], row=2, col=3, color=c_state)
    scatter(t_nodes[:-1], gs_deg_n[:-1], row=2, col=3)
    seg(0, t_h1, gs_bound_pre, row=2, col=3, color=c_up)
    seg(t_h1, t_end, gs_bound_stc, row=2, col=3, color=c_up)
    vline(t_h1, row=2, col=3, color=c_h1)
    fig.update_yaxes(
        title_text="Glideslope, γ [deg]",
        range=[0, max(gs_bound_pre, gs_deg[:-1].max()) + 5],
        row=2,
        col=3,
    )

    # ── LOS boresight elevation angle (row 3, col 1) ──────────────────────────
    line(t_full, np.degrees(boresight), row=3, col=1, color="red")
    scatter(t_nodes, np.degrees(boresight_n), row=3, col=1)
    seg(t_h2, t_end, limits.delta_boresight_max_deg, row=3, col=1, color=c_up)
    vline(t_h2, row=3, col=1, color=c_h2)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_yaxes(title_text="Boresight deflection, δᵇ [deg]", row=3, col=1)

    # ── Mass (row 3, col 2) ───────────────────────────────────────────────────
    line(t_full, mass / 1e3, row=3, col=2, color=c_state)
    scatter(t_nodes, mass_n / 1e3, row=3, col=2)
    fig.add_hline(
        y=limits.m_dry / 1e3, line={"color": c_low, "dash": "dash", "width": 1.5}, row=3, col=2
    )
    fig.update_xaxes(title_text="Time [s]", row=3, col=2)
    fig.update_yaxes(title_text="Mass, m [10³ kg]", row=3, col=2)

    # ── LOS view angle (row 3, col 3) ─────────────────────────────────────────
    line(t_full, los_ang, row=3, col=3, color=c_state)
    scatter(t_nodes, los_ang_n, row=3, col=3)
    seg(t_h2, t_end, limits.los_stc_deg, row=3, col=3, color=c_up)
    vline(t_h2, row=3, col=3, color=c_h2)
    fig.update_xaxes(title_text="Time [s]", row=3, col=3)
    fig.update_yaxes(title_text="LOS angle, ψ [deg]", row=3, col=3)

    for row in range(1, 4):
        for col in range(1, 4):
            fig.update_xaxes(range=[0, t_end], row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=650,
        margin={"t": 80, "b": 40, "l": 50, "r": 30},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
    )
    return fig


def plot_cstc_trajectory_3d(
    result,
    *,
    scales: CstcScales,
    markers: Sequence[tuple[str, np.ndarray, str]],
    title: str,
):
    """3-D path coloured by speed, with the propagated curve and the SCP nodes.

    ``markers`` are ``(label, position_m, colour)`` diamonds for the sites the
    boundary conditions pin — one landing pad for a descent, three for a hop.
    """
    import plotly.graph_objects as go

    pos = np.asarray(result.trajectory["position"]) * scales.r_scale
    vel = np.asarray(result.trajectory["velocity"]) * scales.r_scale
    pos_n = np.asarray(result.nodes["position"]) * scales.r_scale
    speed = np.linalg.norm(vel, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            customdata=speed,
            line={
                "color": speed,
                "colorscale": "Rainbow",
                "cmin": float(speed.min()),
                "cmax": float(speed.max()),
                "width": 4,
                "colorbar": {"title": "Speed [m/s]"},
            },
            showlegend=False,
            hovertemplate=(
                "Crossrange: %{x:.1f} m<br>Downrange: %{y:.1f} m<br>"
                "Altitude: %{z:.1f} m<br>Speed: %{customdata:.1f} m/s<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pos_n[:, 0],
            y=pos_n[:, 1],
            z=pos_n[:, 2],
            mode="markers",
            marker={"color": "black", "size": 4},
            name="Node",
        )
    )
    for label, site_m, color in markers:
        site_m = np.asarray(site_m, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=[site_m[0]],
                y=[site_m[1]],
                z=[site_m[2]],
                mode="markers",
                marker={"color": color, "size": 10, "symbol": "diamond"},
                name=label,
            )
        )

    all_xyz = np.vstack([pos, pos_n])
    pad = 30.0
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.5},
        width=900,
        height=800,
        scene={
            "xaxis_title": "Crossrange [m]",
            "yaxis_title": "Downrange [m]",
            "zaxis_title": "Altitude [m]",
            "xaxis": {"range": [all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad]},
            "yaxis": {"range": [all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad]},
            "zaxis": {"range": [0, all_xyz[:, 2].max() + 30]},
            "aspectmode": "data",
        },
        legend={"x": 0.02, "y": 0.98},
    )
    return fig
