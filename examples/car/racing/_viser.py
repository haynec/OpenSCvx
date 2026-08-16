"""3D Viser scenes shared by the race-car examples.

The LMS track, the low-poly car, and the playback wiring are the same whether
the trajectory came from one minimum-time solve, a closed-loop MPC log, or a
whole field racing wheel to wheel, so every example in this directory builds
its scene here rather than re-rolling one. Three entry points differ only in
the camera and the number of cars:
:func:`create_race_car_viser_server` (one lap, overview camera),
:func:`create_race_car_chase_viser_server` (one lap, chase camera), and
:func:`create_race_car_comparison_viser_server` (several cars on one clock,
optionally with the live telemetry panels from ``_plotting``).

Everything here composes ``openscvx.plotting.viser`` primitives; the 2D
Plotly side of the same examples lives in ``_plotting.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser

from examples.car.racing._time2spatial import transformProj2Orig
from examples.car.racing._tracks.readDataFcn import getTrack

# Body sits slightly above the asphalt strip (m).
CAR_RIDE_HEIGHT = 0.012


def _track_strip_mesh(
    x_left: np.ndarray,
    y_left: np.ndarray,
    x_right: np.ndarray,
    y_right: np.ndarray,
    z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the lane surface between left and right boundary polylines."""
    n = len(x_left)
    if n < 2:
        raise ValueError("track boundary must have at least two points")
    verts = np.empty((2 * n, 3), dtype=np.float32)
    verts[0::2, 0] = x_left
    verts[0::2, 1] = y_left
    verts[0::2, 2] = z
    verts[1::2, 0] = x_right
    verts[1::2, 1] = y_right
    verts[1::2, 2] = z

    faces: list[list[int]] = []
    for i in range(n - 1):
        a = 2 * i
        b = a + 1
        c = a + 2
        d = a + 3
        faces.append([a, b, d])
        faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int32)


def _add_lms_track_scene(
    server: "viser.ViserServer",
    *,
    track_file: str = "LMS_Track.txt",
    lane_width: float = 0.12,
    distance_marker_step: float | str | None = "auto",
) -> None:
    """Static LMS track: asphalt strip, kerbs, centreline, start/finish.

    ``distance_marker_step`` spaces the "x m" labels along the centreline:
    a number is the spacing in metres, ``"auto"`` picks roughly nine per lap,
    and ``None`` hides the markers entirely.
    """
    sref, xref, yref, psiref, _ = getTrack(track_file)
    dist = float(lane_width)
    x_left = xref - dist * np.sin(psiref)
    y_left = yref + dist * np.cos(psiref)
    x_right = xref + dist * np.sin(psiref)
    y_right = yref - dist * np.cos(psiref)

    xmin = min(x_left.min(), x_right.min(), xref.min())
    xmax = max(x_left.max(), x_right.max(), xref.max())
    ymin = min(y_left.min(), y_right.min(), yref.min())
    ymax = max(y_left.max(), y_right.max(), yref.max())
    pad = 0.35
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    gw = (xmax - xmin) + 2.0 * pad
    gh = (ymax - ymin) + 2.0 * pad

    server.scene.add_box(
        "/ground",
        dimensions=(gw, gh, 0.004),
        position=(cx, cy, -0.004),
        color=(28, 32, 38),
    )

    asphalt_verts, asphalt_faces = _track_strip_mesh(x_left, y_left, x_right, y_right, z=0.0)
    server.scene.add_mesh_simple(
        "/track/asphalt",
        vertices=asphalt_verts,
        faces=asphalt_faces,
        color=(55, 58, 62),
        flat_shading=False,
    )

    if len(xref) >= 2:
        centre_segments = np.stack(
            [
                np.column_stack([xref[:-1], yref[:-1], np.zeros(len(xref) - 1)]),
                np.column_stack([xref[1:], yref[1:], np.zeros(len(xref) - 1)]),
            ],
            axis=1,
        ).astype(np.float32)
        server.scene.add_line_segments(
            "/track/centreline",
            points=centre_segments,
            colors=np.array([120, 125, 135], dtype=np.uint8),
            line_width=1.5,
        )

    def _boundary_segments(xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.column_stack([xb[:-1], yb[:-1], np.full(len(xb) - 1, 0.004)]),
                np.column_stack([xb[1:], yb[1:], np.full(len(xb) - 1, 0.004)]),
            ],
            axis=1,
        ).astype(np.float32)

    server.scene.add_line_segments(
        "/track/kerb_left",
        points=_boundary_segments(x_left, y_left),
        colors=np.array([220, 45, 45], dtype=np.uint8),
        line_width=3.5,
    )
    server.scene.add_line_segments(
        "/track/kerb_right",
        points=_boundary_segments(x_right, y_right),
        colors=np.array([240, 240, 240], dtype=np.uint8),
        line_width=3.5,
    )

    # Start / finish line at s ≈ 0
    k0 = int(np.argmin(np.abs(sref)))
    psi0 = float(psiref[k0])
    half_w = dist
    p0 = np.array([xref[k0], yref[k0], 0.006], dtype=np.float64)
    server.scene.add_box(
        "/track/start_finish",
        dimensions=(2.0 * half_w, 0.018, 0.002),
        position=tuple(p0),
        wxyz=tuple(float(x) for x in _yaw_wxyz(psi0)),
        color=(255, 220, 40),
    )
    server.scene.add_label(
        "/track/start_finish/label",
        text="START / FINISH",
        position=tuple(p0 + np.array([0.0, 0.0, 0.035])),
    )

    if distance_marker_step is not None:
        if distance_marker_step == "auto":
            # Roughly every ninth of a lap, whatever the track length.
            step = max(1.0, round(sref[-1] / 9.0))
        else:
            step = float(distance_marker_step)
        for si in np.arange(0.0, sref[-1], step):
            k = int(np.argmin(np.abs(sref - si)))
            server.scene.add_label(
                f"/track/distance/{k}",
                text=f"{si:g} m",
                position=(float(xref[k]), float(yref[k]), 0.05),
            )


def _yaw_wxyz(yaw: float) -> np.ndarray:
    import viser.transforms as vtf

    return np.asarray(vtf.SO3.from_z_radians(float(yaw)).wxyz, dtype=np.float64)


def _add_race_car(
    server: "viser.ViserServer",
    base_path: str = "/car",
    body_color: tuple[int, int, int] = (220, 35, 45),
) -> dict:
    """Low-poly 1:43-scale race car built from boxes (body frame: +x forward, +z up)."""
    frame = server.scene.add_frame(
        base_path,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, CAR_RIDE_HEIGHT),
        show_axes=False,
    )
    body = server.scene.add_box(
        f"{base_path}/body",
        dimensions=(0.068, 0.034, 0.016),
        position=(0.0, 0.0, 0.008),
        color=body_color,
    )
    server.scene.add_box(
        f"{base_path}/cockpit",
        dimensions=(0.028, 0.026, 0.012),
        position=(-0.008, 0.0, 0.018),
        color=(30, 32, 38),
    )
    server.scene.add_box(
        f"{base_path}/nose",
        dimensions=(0.022, 0.022, 0.010),
        position=(0.042, 0.0, 0.006),
        color=(240, 240, 245),
    )
    server.scene.add_box(
        f"{base_path}/wing",
        dimensions=(0.014, 0.048, 0.004),
        position=(-0.034, 0.0, 0.014),
        color=(18, 18, 22),
    )
    wheel_color = (22, 22, 26)
    wheel_specs = [
        ("fl", 0.026, 0.016, 0.0),
        ("fr", 0.026, -0.016, 0.0),
        ("rl", -0.024, 0.016, 0.0),
        ("rr", -0.024, -0.016, 0.0),
    ]
    wheels: dict[str, object] = {}
    for name, wx, wy, steer in wheel_specs:
        wheels[name] = server.scene.add_box(
            f"{base_path}/wheel_{name}",
            dimensions=(0.012, 0.008, 0.012),
            position=(wx, wy, 0.0),
            wxyz=tuple(float(x) for x in _yaw_wxyz(steer)),
            color=wheel_color,
        )
    return {"frame": frame, "body": body, "wheels": wheels}


def _states_to_scene_arrays(
    s: np.ndarray,
    n: np.ndarray,
    alpha: np.ndarray,
    v: np.ndarray,
    throttle: np.ndarray,
    delta: np.ndarray,
    t: np.ndarray,
    *,
    track_file: str = "LMS_Track.txt",
    trim_warmup: bool = True,
) -> dict[str, np.ndarray]:
    """Convert path-parametric state histories to Cartesian Viser arrays."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    n = np.asarray(n, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    throttle = np.asarray(throttle, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).flatten()

    if trim_warmup:
        lap_start = int(np.searchsorted(s, 0.0))
        s, n, alpha, v = s[lap_start:], n[lap_start:], alpha[lap_start:], v[lap_start:]
        throttle, delta = throttle[lap_start:], delta[lap_start:]
        t = t[lap_start:]

    x, y, psi, _ = transformProj2Orig(s, n, alpha, v, track_file)
    z = np.full_like(x, CAR_RIDE_HEIGHT, dtype=np.float64)
    pos = np.column_stack([x, y, z]).astype(np.float32)

    vx = v * np.cos(psi)
    vy = v * np.sin(psi)
    vel = np.column_stack([vx, vy, np.zeros_like(vx)]).astype(np.float64)

    return {
        "t": t,
        "pos": pos,
        "vel": vel,
        "psi": psi.astype(np.float64),
        "speed": v.astype(np.float64),
        "throttle": throttle.astype(np.float64),
        "delta": delta.astype(np.float64),
    }


def extract_race_trajectory(
    results,
    *,
    track_file: str = "LMS_Track.txt",
    trim_warmup: bool = True,
) -> dict[str, np.ndarray]:
    """Convert post-processed race-car results to Cartesian scene arrays."""
    from openscvx.algorithms import OptimizationResults

    if not isinstance(results, OptimizationResults):
        raise TypeError("results must be OptimizationResults (call post_process first)")
    if not results.trajectory:
        raise ValueError("results.trajectory missing; call post_process() first")

    traj = results.trajectory
    return _states_to_scene_arrays(
        traj["s"][:, 0],
        traj["n"][:, 0],
        traj["alpha"][:, 0],
        traj["v"][:, 0],
        traj["D"][:, 0],
        traj["delta"][:, 0],
        results.t_full,
        track_file=track_file,
        trim_warmup=trim_warmup,
    )


def extract_race_trajectory_from_sim(
    simX: np.ndarray,
    t_sim: np.ndarray,
    *,
    track_file: str = "LMS_Track.txt",
    trim_warmup: bool = True,
) -> dict[str, np.ndarray]:
    """Convert closed-loop MPC logs ``simX`` to Cartesian scene arrays.

    ``simX`` columns: ``[s, n, α, v, D, δ]`` (same layout as ``race_car_mpc.py``).
    """
    simX = np.asarray(simX, dtype=np.float64)
    if simX.ndim != 2 or simX.shape[1] < 6:
        raise ValueError("simX must have shape (N, 6) with columns [s, n, α, v, D, δ]")
    return _states_to_scene_arrays(
        simX[:, 0],
        simX[:, 1],
        simX[:, 2],
        simX[:, 3],
        simX[:, 4],
        simX[:, 5],
        t_sim,
        track_file=track_file,
        trim_warmup=trim_warmup,
    )


def _chase_camera_pose(
    car_pos: np.ndarray,
    yaw: float,
    *,
    look_ahead: float = 0.10,
    chase_distance: float = 0.16,
    vertical_offset: float = 0.09,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cam_pos, cam_wxyz, look_at)`` for a chase cam behind the car."""
    from examples.animations._camera import chase_pose

    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    focus = np.asarray(car_pos, dtype=np.float64) + look_ahead * forward
    return chase_pose(
        np.asarray(car_pos, dtype=np.float64),
        focus,
        chase_distance=chase_distance,
        vertical_offset=vertical_offset,
    )


def _attach_chase_camera(
    server: "viser.ViserServer",
    pos: np.ndarray,
    psi: np.ndarray,
    *,
    look_ahead: float = 0.10,
    chase_distance: float = 0.16,
    vertical_offset: float = 0.09,
):
    """Wire per-client chase camera updates; returns an animation callback."""

    def update_chase_camera(frame_idx: int) -> None:
        cam_pos, cam_wxyz, look_at = _chase_camera_pose(
            pos[frame_idx],
            float(psi[frame_idx]),
            look_ahead=look_ahead,
            chase_distance=chase_distance,
            vertical_offset=vertical_offset,
        )
        for client in server.get_clients().values():
            client.camera.position = tuple(float(x) for x in cam_pos)
            client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
            client.camera.look_at = tuple(float(x) for x in look_at)

    cam_pos, cam_wxyz, look_at = _chase_camera_pose(
        pos[0],
        float(psi[0]),
        look_ahead=look_ahead,
        chase_distance=chase_distance,
        vertical_offset=vertical_offset,
    )
    server.initial_camera.position = tuple(float(x) for x in cam_pos)
    server.initial_camera.wxyz = tuple(float(x) for x in cam_wxyz)
    server.initial_camera.look_at = tuple(float(x) for x in look_at)
    server.initial_camera.up = (0.0, 0.0, 1.0)

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        cam_pos, cam_wxyz, look_at = _chase_camera_pose(
            pos[0],
            float(psi[0]),
            look_ahead=look_ahead,
            chase_distance=chase_distance,
            vertical_offset=vertical_offset,
        )
        client.camera.position = tuple(float(x) for x in cam_pos)
        client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
        client.camera.look_at = tuple(float(x) for x in look_at)

    return update_chase_camera


def _create_track_server() -> "viser.ViserServer":
    """Dark, gridless viser server for the track scenes.

    The origin triad is switched off deliberately: its 0.5 m axes would dwarf a
    kart track only 0.24 m wide and sit right on the start/finish line. The
    track outline itself is the orienting feature here.
    """
    from openscvx.plotting.viser import create_server

    return create_server(None, dark_mode=True, show_grid=False, show_origin=False)


def _frame_overview_camera(server: "viser.ViserServer", pos: np.ndarray) -> None:
    """Point the initial camera at the bounding box of ``pos``.

    One fixed three-quarter view scaled by the trajectory's own extent, so the
    2 m kart track and the 4x-scaled circuit both fill the frame.
    """
    centre = np.mean(pos, axis=0)
    span_xy = float(np.ptp(pos[:, :2], axis=0).max()) + 1e-6
    server.initial_camera.position = tuple(
        centre + np.array([-0.55 * span_xy, -0.75 * span_xy, 0.95 * span_xy])
    )
    server.initial_camera.look_at = tuple(float(x) for x in centre)
    server.initial_camera.up = (0.0, 0.0, 1.0)


def _add_ghost_and_trace(
    server: "viser.ViserServer",
    prefix: str,
    pos: np.ndarray,
    *,
    ghost_color: tuple[int, int, int],
    trace_color: tuple[int, int, int],
) -> "viser.LineSegmentsHandle":
    """Full-lap ghost line plus the empty trace line the animation fills in.

    The ghost shows the whole lap from the first frame; the returned trace
    handle starts empty and is extended to the current frame by the caller's
    animation callback, so together they read as "where the car goes" and
    "where it has been".
    """
    from openscvx.plotting.viser import add_ghost_trajectory

    if len(pos) >= 2:
        add_ghost_trajectory(
            server,
            pos,
            np.broadcast_to(np.asarray(ghost_color, dtype=np.uint8), (len(pos), 3)),
            opacity=1.0,
            name=f"{prefix}/ghost",
        )
    return server.scene.add_line_segments(
        f"{prefix}/trace",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.array(trace_color, dtype=np.uint8),
        line_width=4.0,
    )


def _create_race_car_viser_server(
    results=None,
    *,
    simX: np.ndarray | None = None,
    t_sim: np.ndarray | None = None,
    data: dict[str, np.ndarray] | None = None,
    track_file: str = "LMS_Track.txt",
    lane_width: float = 0.12,
    loop_animation: bool = True,
    trim_warmup: bool = True,
    distance_marker_step: float | str | None = "auto",
    chase_camera: bool = False,
    chase_look_ahead: float = 0.10,
    chase_distance: float = 0.16,
    chase_vertical_offset: float = 0.09,
    title: str = "Race Car",
) -> "viser.ViserServer":
    """Build the race-car Viser scene; optionally enable a chase camera."""
    import viser.transforms as vtf

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )

    if data is None:
        if simX is not None:
            if t_sim is None:
                raise ValueError("t_sim is required when simX is provided")
            data = extract_race_trajectory_from_sim(
                simX,
                t_sim,
                track_file=track_file,
                trim_warmup=trim_warmup,
            )
        elif results is not None:
            data = extract_race_trajectory(
                results,
                track_file=track_file,
                trim_warmup=trim_warmup,
            )
        else:
            raise ValueError("Provide results, or simX and t_sim, or pre-built data")
    t_arr = data["t"]
    pos = data["pos"]
    vel = data["vel"]
    psi = data["psi"]
    speed = data["speed"]
    throttle = data["throttle"]
    delta = data["delta"]

    if t_arr.size < 2:
        t_arr = np.linspace(0.0, float(t_arr[-1] if t_arr.size else 1.0), max(len(pos), 2))

    colors = compute_velocity_colors(vel, cmap_name="turbo")

    server = _create_track_server()

    _add_lms_track_scene(
        server,
        track_file=track_file,
        lane_width=lane_width,
        distance_marker_step=distance_marker_step,
    )

    trace_line = _add_ghost_and_trace(
        server,
        "/trajectory",
        pos,
        ghost_color=(70, 120, 200),
        trace_color=(255, 180, 60),
    )

    _, update_trail = add_animated_trail(server, pos, colors, point_size=0.018)

    car = _add_race_car(server)
    car_frame = car["frame"]
    body_handle = car["body"]
    wheel_fl = car["wheels"]["fl"]
    wheel_fr = car["wheels"]["fr"]

    vel_arrow = server.scene.add_line_segments(
        "/car/velocity",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.array([80, 255, 160], dtype=np.uint8),
        line_width=3.0,
    )

    hud = {"markdown": None}

    def _throttle_color(d_val: float) -> tuple[int, int, int]:
        if d_val > 0.05:
            return (40, 220, 90)
        if d_val < -0.05:
            return (255, 80, 80)
        return (220, 35, 45)

    def update_trace(frame_idx: int) -> None:
        idx = min(frame_idx + 1, len(pos))
        if idx < 2:
            trace_line.points = np.zeros((1, 2, 3), dtype=np.float32)
            return
        trace_line.points = np.stack([pos[: idx - 1], pos[1:idx]], axis=1).astype(np.float32)

    def update_car(frame_idx: int) -> None:
        p = pos[frame_idx]
        yaw = float(psi[frame_idx])
        rot = vtf.SO3.from_z_radians(yaw)
        car_frame.position = tuple(float(x) for x in p)
        car_frame.wxyz = rot.wxyz

        steer = float(delta[frame_idx])
        steer_rot = vtf.SO3.from_z_radians(steer)
        wheel_fl.wxyz = tuple(float(x) for x in steer_rot.wxyz)
        wheel_fr.wxyz = tuple(float(x) for x in steer_rot.wxyz)

        spd = float(speed[frame_idx])
        arrow_len = 0.06 + 0.14 * min(spd / max(speed.max(), 1e-6), 1.0)
        tip = p + (rot @ np.array([arrow_len, 0.0, 0.0], dtype=np.float64)).astype(np.float32)
        vel_arrow.points = np.array([[p, tip]], dtype=np.float32)

        body_handle.color = _throttle_color(float(throttle[frame_idx]))

        if hud["markdown"] is not None:
            hud["markdown"].content = (
                f"**Lap time:** {t_arr[-1]:.3f} s  \n"
                f"**t:** {t_arr[frame_idx]:.3f} s  \n"
                f"**Speed:** {spd:.2f} m/s  \n"
                f"**Throttle D:** {throttle[frame_idx]:+.2f}  \n"
                f"**Steering δ:** {np.rad2deg(delta[frame_idx]):+.1f}°"
            )

    callbacks = [update_car, update_trace, update_trail]
    update_chase_camera = None
    if chase_camera:
        update_chase_camera = _attach_chase_camera(
            server,
            pos,
            psi,
            look_ahead=chase_look_ahead,
            chase_distance=chase_distance,
            vertical_offset=chase_vertical_offset,
        )
        callbacks.append(update_chase_camera)
    else:
        _frame_overview_camera(server, pos)

    add_animation_controls(server, t_arr, callbacks, loop=loop_animation)

    update_car(0)
    update_trace(0)
    update_trail(0)
    if update_chase_camera is not None:
        update_chase_camera(0)

    cam_note = (
        "Chase camera follows the car during playback."
        if chase_camera
        else "Press **Play** in the Animation folder to replay the lap."
    )
    with server.gui.add_folder(title):
        hud["markdown"] = server.gui.add_markdown(
            f"**Lap time:** {t_arr[-1]:.3f} s  \n**Max speed:** {speed.max():.2f} m/s  \n{cam_note}"
        )

    mode = "chase-cam" if chase_camera else "overview"
    print(
        f"Viser race-car animation ({mode}) ready — open the URL above. "
        "Press Play in the Animation folder to drive the lap."
    )
    return server


def create_race_car_viser_server(
    results=None,
    *,
    simX: np.ndarray | None = None,
    t_sim: np.ndarray | None = None,
    track_file: str = "LMS_Track.txt",
    lane_width: float = 0.12,
    loop_animation: bool = True,
    trim_warmup: bool = True,
    distance_marker_step: float | str | None = "auto",
    title: str = "Race Car",
) -> "viser.ViserServer":
    """Interactive 3D lap replay: track mesh, speed-coloured trail, and posed car."""
    return _create_race_car_viser_server(
        results,
        simX=simX,
        t_sim=t_sim,
        track_file=track_file,
        lane_width=lane_width,
        loop_animation=loop_animation,
        trim_warmup=trim_warmup,
        distance_marker_step=distance_marker_step,
        chase_camera=False,
        title=title,
    )


def _resample_to_common_time(
    datas: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Resample scene arrays onto one clock spanning the slowest lap.

    Each lap is shifted to start at t = 0 and linearly interpolated; past its
    own finish a car simply holds the final sample (parked at the line), so
    the time gap between cars stays visible until the animation ends.
    """
    laps = []
    for d in datas:
        t = d["t"] - d["t"][0]
        psi = np.unwrap(d["psi"])
        laps.append((t, psi, d))

    t_end = max(float(t[-1]) for t, _, _ in laps)
    n_frames = max(len(t) for t, _, _ in laps)
    t_common = np.linspace(0.0, t_end, n_frames)

    resampled = []
    for t, psi, d in laps:

        def interp(sig: np.ndarray, t: np.ndarray = t) -> np.ndarray:
            return np.interp(t_common, t, sig)  # clamps past t[-1]

        resampled.append(
            {
                "t": t_common,
                "lap_time": float(t[-1]),
                "pos": np.column_stack([interp(d["pos"][:, k]) for k in range(3)]).astype(
                    np.float32
                ),
                "psi": interp(psi),
                "speed": interp(d["speed"]),
                "delta": interp(d["delta"]),
            }
        )
    return t_common, resampled


def create_race_car_comparison_viser_server(
    results_list=None,
    labels: list[str] | None = None,
    *,
    simX_list: list[np.ndarray] | None = None,
    t_sim_list: list[np.ndarray] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    track_file: str = "LMS_Track.txt",
    lane_width: float = 0.12,
    loop_animation: bool = True,
    trim_warmup: bool = True,
    distance_marker_step: float | str | None = "auto",
    plot_panels: list[dict] | None = None,
    title: str = "Race Comparison",
) -> "viser.ViserServer":
    """Replay several laps side by side on one track with a shared clock.

    All cars launch together at the start line; each holds at the finish once
    its own lap is done, so the winning margin is visible on screen. Intended
    for A/B comparisons such as the hybrid vs ICE-only laps of
    ``race_car_hybrid.py``, or multi-car MPC races via ``simX_list``.

    Args:
        results_list: Post-processed ``OptimizationResults``, one per car.
        labels: Display name per car (default ``car 0``, ``car 1``, ...).
        simX_list: Alternative to ``results_list``: closed-loop MPC logs, one
            ``(N, 6)`` array per car with columns ``[s, n, α, v, D, δ]`` (same
            layout as :func:`extract_race_trajectory_from_sim`).
        t_sim_list: Time vector per car; required with ``simX_list``.
        colors: Body RGB per car (defaults cycle a small palette).
        distance_marker_step: Metres between "x m" track labels; ``"auto"``
            picks ~9 per lap, ``None`` hides them.
        plot_panels: Optional live plots for the GUI sidebar: dicts with a
            Plotly ``"figure"``, an ``"update"`` callable taking the current
            playback time in seconds (mutate the figure's marker traces), and
            an optional ``"aspect"`` ratio. Updates are throttled — Plotly
            panels re-serialize on every assignment.
    """
    import viser.transforms as vtf

    from openscvx.plotting.viser import add_animation_controls

    if simX_list is not None:
        if t_sim_list is None:
            raise ValueError("t_sim_list is required when simX_list is provided")
        datas = [
            extract_race_trajectory_from_sim(
                simX, t_sim, track_file=track_file, trim_warmup=trim_warmup
            )
            for simX, t_sim in zip(simX_list, t_sim_list)
        ]
    elif results_list is not None:
        datas = [
            extract_race_trajectory(results, track_file=track_file, trim_warmup=trim_warmup)
            for results in results_list
        ]
    else:
        raise ValueError("Provide results_list, or simX_list and t_sim_list")

    if labels is None:
        labels = [f"car {i}" for i in range(len(datas))]
    if colors is None:
        palette = [(220, 35, 45), (90, 140, 235), (240, 190, 50), (120, 200, 120)]
        colors = [palette[i % len(palette)] for i in range(len(datas))]
    t_common, laps = _resample_to_common_time(datas)

    server = _create_track_server()

    _add_lms_track_scene(
        server,
        track_file=track_file,
        lane_width=lane_width,
        distance_marker_step=distance_marker_step,
    )

    cars = []
    for lap, label, color in zip(laps, labels, colors):
        slug = label.replace(" ", "_").replace("/", "_")
        trace_line = _add_ghost_and_trace(
            server,
            f"/trajectory/{slug}",
            lap["pos"],
            ghost_color=tuple(int(0.45 * c) for c in color),  # the car's colour, dimmed
            trace_color=color,
        )
        handles = _add_race_car(server, base_path=f"/cars/{slug}", body_color=color)
        cars.append({"lap": lap, "label": label, "trace": trace_line, **handles})

    hud = {"markdown": None}

    def update_frame(frame_idx: int) -> None:
        status_lines = []
        for car in cars:
            lap = car["lap"]
            p = lap["pos"][frame_idx]
            rot = vtf.SO3.from_z_radians(float(lap["psi"][frame_idx]))
            car["frame"].position = tuple(float(x) for x in p)
            car["frame"].wxyz = rot.wxyz

            steer_rot = vtf.SO3.from_z_radians(float(lap["delta"][frame_idx]))
            car["wheels"]["fl"].wxyz = tuple(float(x) for x in steer_rot.wxyz)
            car["wheels"]["fr"].wxyz = tuple(float(x) for x in steer_rot.wxyz)

            idx = min(frame_idx + 1, len(lap["pos"]))
            if idx >= 2:
                car["trace"].points = np.stack(
                    [lap["pos"][: idx - 1], lap["pos"][1:idx]], axis=1
                ).astype(np.float32)

            finished = t_common[frame_idx] >= lap["lap_time"]
            state = "FINISHED" if finished else f"{lap['speed'][frame_idx]:.2f} m/s"
            status_lines.append(f"**{car['label']}:** {lap['lap_time']:.3f} s — {state}")

        if hud["markdown"] is not None:
            hud["markdown"].content = f"**t:** {t_common[frame_idx]:.3f} s  \n" + "  \n".join(
                status_lines
            )

    # Live GUI plots: each panel owns its figure; we hand it the playback time
    # and push the mutated figure to the client, rate-limited because every
    # push re-serializes the whole figure. The large ``order`` keeps the
    # telemetry folder below the Animation and race-status folders, which are
    # created later with default (insertion-order) positions.
    panel_handles = []
    if plot_panels:
        with server.gui.add_folder("Telemetry", order=1000):
            for panel in plot_panels:
                handle = server.gui.add_plotly(
                    figure=panel["figure"], aspect=float(panel.get("aspect", 1.6))
                )
                panel_handles.append((panel, handle))
    _panel_clock = {"last_t": -np.inf}

    def update_panels(frame_idx: int) -> None:
        t = float(t_common[frame_idx])
        if abs(t - _panel_clock["last_t"]) < 0.15:
            return
        _panel_clock["last_t"] = t
        for panel, handle in panel_handles:
            panel["update"](t)
            handle.figure = panel["figure"]

    _frame_overview_camera(server, np.concatenate([lap["pos"] for lap in laps]))

    callbacks = [update_frame] + ([update_panels] if panel_handles else [])
    add_animation_controls(server, t_common, callbacks, loop=loop_animation)
    update_frame(0)

    lap_times = sorted((lap["lap_time"], label) for lap, label in zip(laps, labels))
    gap_note = "  \n".join(
        f"**{label}:** {lt:.3f} s ({f'+{lt - lap_times[0][0]:.3f} s' if i else 'fastest'})"
        for i, (lt, label) in enumerate(lap_times)
    )
    with server.gui.add_folder(title):
        hud["markdown"] = server.gui.add_markdown(gap_note)

    print(
        "Viser race comparison ready — open the URL above. "
        "Press Play in the Animation folder to race the laps."
    )
    return server


def create_race_car_chase_viser_server(
    results=None,
    *,
    simX: np.ndarray | None = None,
    t_sim: np.ndarray | None = None,
    track_file: str = "LMS_Track.txt",
    lane_width: float = 0.12,
    loop_animation: bool = True,
    trim_warmup: bool = True,
    distance_marker_step: float | str | None = "auto",
    look_ahead: float = 0.10,
    chase_distance: float = 0.16,
    vertical_offset: float = 0.09,
    title: str = "Race Car",
) -> "viser.ViserServer":
    """Same lap replay with a chase camera locked behind the car."""
    return _create_race_car_viser_server(
        results,
        simX=simX,
        t_sim=t_sim,
        track_file=track_file,
        lane_width=lane_width,
        loop_animation=loop_animation,
        trim_warmup=trim_warmup,
        distance_marker_step=distance_marker_step,
        chase_camera=True,
        chase_look_ahead=look_ahead,
        chase_distance=chase_distance,
        chase_vertical_offset=vertical_offset,
        title=title,
    )
