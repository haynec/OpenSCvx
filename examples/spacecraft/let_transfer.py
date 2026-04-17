"""Low-Energy Transfer (LET) setup in Sun-Earth CR3BP with one departure impulse.

Modeling choices:
- Sun-Earth CR3BP rotating-frame dynamics (x shifted so Earth is at x=0)
- Impulsive delta-v at departure and at the final node (arrival burn)
- Fixed initial state, fixed final position, free final velocity
- Free final time with uniform time grid (single global dilation behavior)
- Objective: minimize total impulsive delta-v magnitude
"""

import os
import shutil
import sys
import time as pytime
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

# Add grandparent directory to path to import openscvx without installation.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.algorithms import OptimizationResults
from openscvx.integrators import solve_ivp_diffrax
from openscvx.plotting import plot_projections_2d, plot_states
from openscvx.symbolic.lower import lower_to_jax

# Use float64 in JAX for high-accuracy propagation.
jax.config.update("jax_enable_x64", True)

REFERENCE_DATE = "26 December 2025"
ENABLE_VISER_ANIMATION = True
ENABLE_VISER_INERTIAL_ANIMATION = True
VISER_VISUAL_SCALE = 250.0
VISER_MIN_SPEED_FOR_SAMPLING = 0.01
VISER_TARGET_FPS = 60.0
VISER_MAX_RESAMPLED_POINTS = 120000
VISER_ROTATING_PORT = 8080
VISER_INERTIAL_PORT = 8081
VISER_REQUEST_SHARE_URLS = False
KERNEL_DIR = Path(current_dir) / "ker"
KERNEL_URLS = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de440.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
    "pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
    "gm_de440.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
}
KERNEL_FILENAMES = tuple(KERNEL_URLS.keys())

def _download_kernel(url: str, destination: Path) -> None:
    """Download a single SPICE kernel to destination atomically."""
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        temp_destination.open("wb") as out_file,
    ):
        shutil.copyfileobj(response, out_file)
    temp_destination.replace(destination)


def _ensure_spice_kernels(kernel_dir: Path) -> None:
    """Ensure all required kernels exist in kernel_dir, downloading missing files."""
    kernel_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in KERNEL_FILENAMES if not (kernel_dir / name).is_file()]
    if not missing:
        return

    download_errors = []
    for kernel_name in missing:
        destination = kernel_dir / kernel_name
        try:
            _download_kernel(KERNEL_URLS[kernel_name], destination)
        except Exception as exc:
            part_file = destination.with_suffix(destination.suffix + ".part")
            if part_file.exists():
                part_file.unlink()
            download_errors.append(f"{kernel_name}: {exc}")

    if download_errors:
        raise RuntimeError("Failed to download SPICE kernels: " + "; ".join(download_errors))


def _load_spice_problem_data(reference_date: str) -> dict:
    """Load constants and characteristic distances from SPICE kernels."""
    import spiceypy as spice

    _ensure_spice_kernels(KERNEL_DIR)
    spice.kclear()
    for kernel_name in KERNEL_FILENAMES:
        spice.furnsh(str(KERNEL_DIR / kernel_name))

    et = spice.str2et(reference_date)
    mu_earth_val = spice.bodvrd("Earth", "GM", 1)[1][0]
    mu_sun_val = spice.bodvrd("Sun", "GM", 1)[1][0]
    r_earth_val = spice.bodvrd("Earth", "RADII", 3)[1][0]

    pos_earth = spice.spkezr("Earth", et, "ECLIPJ2000", "NONE", "SSB")[0][:3]
    pos_sun = spice.spkezr("Sun", et, "ECLIPJ2000", "NONE", "SSB")[0][:3]
    pos_moon = spice.spkezr("Moon", et, "ECLIPJ2000", "NONE", "SSB")[0][:3]

    return {
        "mu_earth": float(mu_earth_val),
        "mu_sun": float(mu_sun_val),
        "r_earth": float(r_earth_val),
        "d_earth_sun": float(np.linalg.norm(pos_earth - pos_sun)),
        "d_earth_moon": float(np.linalg.norm(pos_earth - pos_moon)),
        "kernel_dir": str(KERNEL_DIR),
        "reference_date": reference_date,
    }


def _normalized_node_grid(n: int, mode: str) -> np.ndarray:
    """Build a normalized node grid in [0, 1] according to the selected mode."""
    s_uniform = np.linspace(0.0, 1.0, n)
    mode_l = mode.strip().lower()
    if mode_l == "uniform":
        return s_uniform
    if mode_l == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * s_uniform))
    raise ValueError(
        f"Unknown NODE_DISTRIBUTION_MODE={mode!r}. Expected 'uniform' or 'cosine'."
    )


def _add_moon_orbit_overlay(fig, earth_pos: np.ndarray, moon_radius: float) -> None:
    """Overlay Moon orbit projections (XY/XZ/YZ) on the 2D projection figure."""
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    x_orbit = earth_pos[0] + moon_radius * np.cos(theta)
    y_orbit = earth_pos[1] + moon_radius * np.sin(theta)
    z_orbit = np.zeros_like(theta)

    orbit_line = {"color": "rgba(255, 255, 255, 0.55)", "width": 1.5, "dash": "dash"}

    # XY plane
    fig.add_trace(
        go.Scatter(
            x=x_orbit,
            y=y_orbit,
            mode="lines",
            line=orbit_line,
            name="Moon orbit",
            legendgroup="moon_orbit",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # XZ plane (projection of the orbit onto z=0)
    fig.add_trace(
        go.Scatter(
            x=x_orbit,
            y=z_orbit,
            mode="lines",
            line=orbit_line,
            name="Moon orbit",
            legendgroup="moon_orbit",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # YZ plane (projection of the orbit onto z=0)
    fig.add_trace(
        go.Scatter(
            x=y_orbit,
            y=z_orbit,
            mode="lines",
            line=orbit_line,
            name="Moon orbit",
            legendgroup="moon_orbit",
            showlegend=False,
        ),
        row=2,
        col=1,
    )


def _set_projection_axis_labels_km(fig) -> None:
    """Set projection subplot axis labels to km."""
    fig.update_xaxes(title_text="X (km)", row=1, col=1)
    fig.update_yaxes(title_text="Y (km)", row=1, col=1)
    fig.update_xaxes(title_text="X (km)", row=1, col=2)
    fig.update_yaxes(title_text="Z (km)", row=1, col=2)
    fig.update_xaxes(title_text="Y (km)", row=2, col=1)
    fig.update_yaxes(title_text="Z (km)", row=2, col=1)


def _set_projection_speed_colorbar_kms(fig) -> None:
    """Relabel projection colorbar to km/s when velocity-based coloring is used."""
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is not None and getattr(marker, "colorbar", None) is not None:
            marker.colorbar.title = "‖velocity‖ (km/s)"


def _create_let_viser_server(
    trajectory: np.ndarray,
    traj_time_days: np.ndarray,
    earth_pos: np.ndarray,
    sun_pos: np.ndarray,
    moon_radius: float,
    moon_rate_rad_per_day: float,
    guess_trajectory: np.ndarray | None = None,
    port: int = VISER_ROTATING_PORT,
):
    """Create a viser server for LET trajectory playback."""
    try:
        from openscvx.plotting import viser as ox_viser

        pos = np.asarray(trajectory[:, :3], dtype=np.float64)
        vel = np.asarray(trajectory[:, 3:6], dtype=np.float64)

        # Autoscale scene so tiny normalized CR3BP coordinates remain visible.
        scale = max(float(moon_radius), float(np.linalg.norm(pos, axis=1).max()), 1e-9)
        pos_vis = (pos / scale) * VISER_VISUAL_SCALE
        earth_vis = (np.asarray(earth_pos, dtype=np.float64) / scale) * VISER_VISUAL_SCALE
        moon_radius_vis = (float(moon_radius) / scale) * VISER_VISUAL_SCALE

        colors = ox_viser.compute_velocity_colors(vel, fallback_length=pos.shape[0])
        server = ox_viser.create_server(pos_vis, show_grid=True, port=port)

        ox_viser.add_circular_orbit(
            server,
            radius=moon_radius_vis,
            name="moon_orbit",
            center=earth_vis,
            color=(135, 135, 135),
            line_width=1.6,
        )

        earth_radius = max(0.03 * moon_radius_vis, 0.15)
        spacecraft_radius = 0.6 * earth_radius
        server.scene.add_icosphere(
            "/bodies/earth",
            radius=earth_radius,
            color=(80, 160, 255),
            position=earth_vis,
        )
        sun_vis_real = (np.asarray(sun_pos, dtype=np.float64) / scale) * VISER_VISUAL_SCALE
        sun_vis_norm = float(np.linalg.norm(sun_vis_real))
        if sun_vis_norm > 1.8 * VISER_VISUAL_SCALE:
            sun_vis = sun_vis_real * ((1.8 * VISER_VISUAL_SCALE) / sun_vis_norm)
        else:
            sun_vis = sun_vis_real
        server.scene.add_icosphere(
            "/bodies/sun",
            radius=1.15 * earth_radius,
            color=(255, 210, 70),
            position=sun_vis,
        )
        # Phase the Moon so that at final time it rendezvous with the terminal trajectory point.
        traj_time_days = np.asarray(traj_time_days, dtype=np.float64).flatten()
        rel_final = pos_vis[-1] - earth_vis
        rel_final_xy = np.array([rel_final[0], rel_final[1]], dtype=np.float64)
        if np.linalg.norm(rel_final_xy) > 1e-12:
            theta_final = float(np.arctan2(rel_final_xy[1], rel_final_xy[0]))
        else:
            theta_final = -0.5 * np.pi
        theta_0 = theta_final - moon_rate_rad_per_day * float(traj_time_days[-1])
        theta = theta_0 + moon_rate_rad_per_day * traj_time_days
        moon_positions = earth_vis.reshape(1, 3) + moon_radius_vis * np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros_like(theta)]
        )
        moon_handle = server.scene.add_icosphere(
            "/bodies/moon",
            radius=0.6 * earth_radius,
            color=(220, 220, 220),
            position=moon_positions[0],
        )

        if guess_trajectory is not None:
            guess_pos = np.asarray(guess_trajectory[:, :3], dtype=np.float64)
            guess_pos_vis = (guess_pos / scale) * VISER_VISUAL_SCALE
            guess_colors = np.broadcast_to(
                np.array([190, 150, 255], dtype=np.uint8), (guess_pos_vis.shape[0], 3)
            ).copy()
            ox_viser.add_ghost_trajectory(
                server, guess_pos_vis, guess_colors, opacity=0.08, point_size=0.20
            )

        ox_viser.add_ghost_trajectory(server, pos_vis, colors, opacity=0.25, point_size=0.25)
        _, update_trail = ox_viser.add_animated_trail(server, pos_vis, colors, point_size=0.45)
        _, update_marker = ox_viser.add_position_marker(
            server, pos_vis, radius=spacecraft_radius, color=(255, 160, 90)
        )

        def update_moon(frame_idx: int) -> None:
            moon_handle.position = moon_positions[frame_idx]

        ox_viser.add_animation_controls(
            server,
            np.asarray(traj_time_days, dtype=np.float64),
            [update_trail, update_marker, update_moon],
            loop=True,
            folder_name="LET Animation",
            time_label="Time (days)",
        )
        return server
    except Exception as exc:
        print(f"Viser animation unavailable: {exc}")
        return None


def _rotate_about_z(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate Nx3 vectors around +Z by angle array theta (radians)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).flatten()
    c = np.cos(theta)
    s = np.sin(theta)
    out = np.empty_like(vectors)
    out[:, 0] = c * vectors[:, 0] - s * vectors[:, 1]
    out[:, 1] = s * vectors[:, 0] + c * vectors[:, 1]
    out[:, 2] = vectors[:, 2]
    return out


def _hohmann_transfer_metrics(
    mu_central_km3_s2: float,
    r1_km: float,
    r2_km: float,
) -> dict:
    """Compute Earth-centered two-impulse Hohmann delta-v."""
    if r1_km <= 0.0 or r2_km <= 0.0:
        raise ValueError(f"Invalid Hohmann radii: r1={r1_km}, r2={r2_km}.")

    a_t = 0.5 * (r1_km + r2_km)
    v_c1 = np.sqrt(mu_central_km3_s2 / r1_km)
    v_c2 = np.sqrt(mu_central_km3_s2 / r2_km)
    v_t1 = np.sqrt(mu_central_km3_s2 * (2.0 / r1_km - 1.0 / a_t))
    v_t2 = np.sqrt(mu_central_km3_s2 * (2.0 / r2_km - 1.0 / a_t))

    dv1_km_s = abs(v_t1 - v_c1)
    dv2_km_s = abs(v_c2 - v_t2)
    total_dv_km_s = dv1_km_s + dv2_km_s

    return {
        "dv1_km_s": float(dv1_km_s),
        "dv2_km_s": float(dv2_km_s),
        "total_dv_km_s": float(total_dv_km_s),
    }


def _create_let_viser_server_inertial(
    trajectory: np.ndarray,
    traj_time_days: np.ndarray,
    r_ref_km: float,
    d_earth_sun_km: float,
    d_earth_moon_km: float,
    moon_rate_rad_per_day: float,
    kappa_val: float,
    guess_trajectory: np.ndarray | None = None,
    guess_time_days: np.ndarray | None = None,
    port: int = VISER_INERTIAL_PORT,
):
    """Create a Sun-centered inertial-frame viser animation."""
    try:
        from openscvx.plotting import viser as ox_viser

        traj_time_days = np.asarray(traj_time_days, dtype=np.float64).flatten()
        pos_rot = np.asarray(trajectory[:, :3], dtype=np.float64)
        vel_rot = np.asarray(trajectory[:, 3:6], dtype=np.float64)

        tau = traj_time_days * d_2_sec / t_ref
        theta = kappa_val * tau
        rho_es_local = d_earth_sun_km / r_ref_km
        rho_em_local = d_earth_moon_km / r_ref_km

        earth_pos = np.column_stack(
            [rho_es_local * np.cos(theta), rho_es_local * np.sin(theta), np.zeros_like(theta)]
        )
        sat_rel_inertial = _rotate_about_z(pos_rot, theta)
        sat_pos = earth_pos + sat_rel_inertial

        rel_final = sat_pos[-1] - earth_pos[-1]
        rel_final_xy = np.array([rel_final[0], rel_final[1]], dtype=np.float64)
        if np.linalg.norm(rel_final_xy) > 1e-12:
            phi_final = float(np.arctan2(rel_final_xy[1], rel_final_xy[0]))
        else:
            phi_final = 0.0
        phi_0 = phi_final - moon_rate_rad_per_day * float(traj_time_days[-1])
        phi = phi_0 + moon_rate_rad_per_day * traj_time_days
        moon_pos = earth_pos + rho_em_local * np.column_stack(
            [np.cos(phi), np.sin(phi), np.zeros_like(phi)]
        )

        scene_max = max(
            float(np.linalg.norm(sat_pos, axis=1).max()),
            float(np.linalg.norm(earth_pos, axis=1).max()),
            float(np.linalg.norm(moon_pos, axis=1).max()),
            1e-9,
        )
        pos_vis = (sat_pos / scene_max) * VISER_VISUAL_SCALE
        earth_vis = (earth_pos / scene_max) * VISER_VISUAL_SCALE
        moon_vis = (moon_pos / scene_max) * VISER_VISUAL_SCALE
        sun_vis = np.zeros(3, dtype=np.float64)

        colors = ox_viser.compute_velocity_colors(vel_rot, fallback_length=pos_vis.shape[0])
        server = ox_viser.create_server(pos_vis, show_grid=True, port=port)

        sun_radius = max(0.04 * VISER_VISUAL_SCALE, 0.2)
        earth_radius = max(0.0018 * VISER_VISUAL_SCALE, 0.035)
        moon_radius = 0.45 * earth_radius
        spacecraft_radius = 0.22 * earth_radius

        server.scene.add_icosphere(
            "/bodies/sun",
            radius=sun_radius,
            color=(255, 210, 70),
            position=sun_vis,
        )
        earth_handle = server.scene.add_icosphere(
            "/bodies/earth",
            radius=earth_radius,
            color=(80, 160, 255),
            position=earth_vis[0],
        )
        moon_handle = server.scene.add_icosphere(
            "/bodies/moon",
            radius=moon_radius,
            color=(220, 220, 220),
            position=moon_vis[0],
        )

        earth_orbit_pos = np.column_stack(
            [
                rho_es_local * np.cos(np.linspace(0.0, 2.0 * np.pi, 721)),
                rho_es_local * np.sin(np.linspace(0.0, 2.0 * np.pi, 721)),
                np.zeros(721),
            ]
        )
        earth_orbit_vis = (earth_orbit_pos / scene_max) * VISER_VISUAL_SCALE
        earth_orbit_colors = np.broadcast_to(
            np.array([90, 140, 255], dtype=np.uint8), (earth_orbit_vis.shape[0], 3)
        ).copy()
        ox_viser.add_ghost_trajectory(
            server, earth_orbit_vis, earth_orbit_colors, opacity=0.14, point_size=0.20
        )

        moon_orbit_colors = np.broadcast_to(
            np.array([175, 175, 175], dtype=np.uint8), (moon_vis.shape[0], 3)
        ).copy()
        ox_viser.add_ghost_trajectory(
            server, moon_vis, moon_orbit_colors, opacity=0.04, point_size=0.10
        )

        if guess_trajectory is not None:
            guess_pos_rot = np.asarray(guess_trajectory[:, :3], dtype=np.float64)
            if guess_time_days is None:
                guess_time_days = np.linspace(0.0, traj_time_days[-1], guess_pos_rot.shape[0])
            guess_tau = np.asarray(guess_time_days, dtype=np.float64).flatten() * d_2_sec / t_ref
            guess_theta = kappa_val * guess_tau
            guess_earth = np.column_stack(
                [
                    rho_es_local * np.cos(guess_theta),
                    rho_es_local * np.sin(guess_theta),
                    np.zeros_like(guess_theta),
                ]
            )
            guess_sat = guess_earth + _rotate_about_z(guess_pos_rot, guess_theta)
            guess_vis = (guess_sat / scene_max) * VISER_VISUAL_SCALE
            guess_colors = np.broadcast_to(
                np.array([180, 145, 250], dtype=np.uint8), (guess_vis.shape[0], 3)
            ).copy()
            ox_viser.add_ghost_trajectory(
                server, guess_vis, guess_colors, opacity=0.07, point_size=0.20
            )

        ox_viser.add_ghost_trajectory(server, pos_vis, colors, opacity=0.16, point_size=0.12)
        _, update_trail = ox_viser.add_animated_trail(server, pos_vis, colors, point_size=0.18)
        _, update_marker = ox_viser.add_position_marker(
            server, pos_vis, radius=spacecraft_radius, color=(255, 160, 90)
        )

        camera_view_dir: dict[int, np.ndarray] = {}
        camera_view_dist: dict[int, float] = {}

        def _initialize_camera_tracking(client, earth_target: np.ndarray) -> bool:
            try:
                if float(client.camera.update_timestamp) <= 0.0:
                    return False
                rel = np.asarray(client.camera.position) - np.asarray(client.camera.look_at)
                rel_norm = float(np.linalg.norm(rel))
                if rel_norm < 1e-6:
                    rel = np.array(
                        [0.0, -0.20 * VISER_VISUAL_SCALE, 0.08 * VISER_VISUAL_SCALE],
                        dtype=np.float64,
                    )
                    rel_norm = float(np.linalg.norm(rel))
                camera_view_dir[client.client_id] = rel / rel_norm
                camera_view_dist[client.client_id] = rel_norm
                client.camera.position = earth_target + camera_view_dir[client.client_id] * rel_norm
                client.camera.look_at = earth_target
                return True
            except Exception:
                return False

        @server.on_client_connect
        def _on_client_connect(client) -> None:
            _initialize_camera_tracking(client, earth_vis[0])

        @server.on_client_disconnect
        def _on_client_disconnect(client) -> None:
            camera_view_dir.pop(client.client_id, None)
            camera_view_dist.pop(client.client_id, None)

        def update_earth(frame_idx: int) -> None:
            earth_handle.position = earth_vis[frame_idx]
            earth_target = earth_vis[frame_idx]
            for client_id, client in server.get_clients().items():
                try:
                    if client_id not in camera_view_dir:
                        if not _initialize_camera_tracking(client, earth_target):
                            continue
                        continue
                    current_rel = np.asarray(client.camera.position) - np.asarray(
                        client.camera.look_at
                    )
                    current_dist = float(np.linalg.norm(current_rel))
                    if current_dist > 1e-6:
                        camera_view_dist[client_id] = current_dist
                    client.camera.position = (
                        earth_target
                        + camera_view_dir[client_id] * camera_view_dist[client_id]
                    )
                    client.camera.look_at = earth_target
                except Exception as exc:
                    print(
                        (
                            f"[LET visualization] Failed to update camera for client "
                            f"{client_id}: {exc}"
                        ),
                        file=sys.stderr,
                    )
                    continue

        def update_moon(frame_idx: int) -> None:
            moon_handle.position = moon_vis[frame_idx]

        def reset_camera_tracking() -> None:
            camera_view_dir.clear()
            camera_view_dist.clear()
            earth_start = earth_vis[0]
            for client in server.get_clients().values():
                _initialize_camera_tracking(client, earth_start)

        ox_viser.add_animation_controls(
            server,
            traj_time_days,
            [update_trail, update_marker, update_earth, update_moon],
            loop=True,
            folder_name="LET Inertial (Sun-Centered)",
            time_label="Time (days)",
            reset_callbacks=[reset_camera_tracking],
        )
        return server
    except Exception as exc:
        print(f"Sun-centered inertial viser animation unavailable: {exc}")
        return None


def _resample_trajectory_for_viser(
    trajectory: np.ndarray,
    traj_time_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample trajectory for smoother low-speed viser playback."""
    traj_time_days = np.asarray(traj_time_days, dtype=np.float64).flatten()
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if traj_time_days.size < 2 or trajectory.shape[0] < 2:
        return trajectory, traj_time_days

    desired_dt_days = VISER_MIN_SPEED_FOR_SAMPLING / VISER_TARGET_FPS
    horizon_days = float(traj_time_days[-1] - traj_time_days[0])
    n_target = int(np.ceil(horizon_days / max(desired_dt_days, 1e-12))) + 1
    n_target = min(max(n_target, trajectory.shape[0]), VISER_MAX_RESAMPLED_POINTS)
    if n_target <= trajectory.shape[0]:
        return trajectory, traj_time_days

    t_dense = np.linspace(traj_time_days[0], traj_time_days[-1], n_target)
    traj_dense = np.column_stack(
        [np.interp(t_dense, traj_time_days, trajectory[:, i]) for i in range(trajectory.shape[1])]
    )
    return traj_dense, t_dense


spice_data = _load_spice_problem_data(REFERENCE_DATE)
mu_earth = spice_data["mu_earth"]
mu_sun = spice_data["mu_sun"]
r_earth = spice_data["r_earth"]
d_earth_sun = spice_data["d_earth_sun"]
d_earth_moon = spice_data["d_earth_moon"]
spice_source = f"SPICE ({spice_data['kernel_dir']})"

# Sun-Earth normalized CR3BP parameters
mu = mu_earth / (mu_earth + mu_sun)
d_2_sec = 86400.0
n_system = np.sqrt((mu_earth + mu_sun) / d_earth_sun**3)
canonical_t_ref = 1 / n_system
legacy_t_ref = d_2_sec * 365.0 / (2.0 * np.pi)

# Optional nondimensional reference scales:
# - If omitted, defaults preserve the original scaling
REF_LENGTH_KM = None
REF_TIME_S = None

ref_length_km = REF_LENGTH_KM
ref_time_s = REF_TIME_S
use_legacy_dynamics = ref_length_km is None and ref_time_s is None

r_ref = float(d_earth_sun if ref_length_km is None else ref_length_km)
t_ref = float(legacy_t_ref if ref_time_s is None else ref_time_s)
if r_ref <= 0.0 or t_ref <= 0.0:
    raise ValueError(f"Invalid reference scales: L={r_ref}, T={t_ref}. Both must be positive.")

v_ref = r_ref / t_ref
kappa = n_system * t_ref
rho_es = d_earth_sun / r_ref
grav_scale = t_ref**2 / r_ref**3

# Mission setup
h_earth = 1500.0
v_moon = np.sqrt(mu_earth/d_earth_moon) / v_ref
r_0 = r_earth + h_earth
# Earth-centered rotating coordinates: Earth at x=0, Sun at x=-1.
pos_earth_rot = np.array([0.0, 0.0, 0.0])
rot_deg             = 0.0
rot_rad             = np.deg2rad(rot_deg)
rot_mat             = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0], 
                                [np.sin(rot_rad), np.cos(rot_rad), 0],
                                  [0, 0, 1]])
pos_0 = pos_earth_rot + rot_mat @ np.array([r_0 / r_ref, 0.0, 0.0])
vel_0 = rot_mat @ np.array([0.0, 7.8 * np.sqrt(2.0) * 0.9085 / v_ref, 0.0])

x0_seed = np.concatenate([pos_0, vel_0])

pos_f = pos_earth_rot + np.array([0.0, -d_earth_moon / r_ref, 0.0])
vel_f = np.array([ np.sqrt(mu_earth / d_earth_moon) / v_ref, 0.0, 0.0])
t_f_guess_days = 78.0
t_f_guess = t_f_guess_days * d_2_sec / t_ref

# Initial impulse guess.
v_circular      = np.sqrt(mu_earth / r_0) / v_ref
v_circular_vect = rot_mat @ np.array([0, v_circular, 0])
delta_v0_guess  = vel_0 - v_circular_vect

n_nodes = 45
integration_tol = 1e-10
integration_max_steps = 3000

# Guess-node distribution toggle:
# - "uniform": evenly spaced nodes in [0, 1]
# - "cosine": denser near interval endpoints
NODE_DISTRIBUTION_MODE = "cosine"

# Build symbolic CR3BP model once and reuse it for optimization and propagation.
position = ox.State("position", shape=(3,))
velocity = ox.State("velocity", shape=(3,))
fuel = ox.State("fuel", shape=(1,))

# Assign slices for standalone lowering/evaluation on [x, y, z, vx, vy, vz].
position._slice = slice(0, 3)
velocity._slice = slice(3, 6)

x_e = position[0]
y_e = position[1]
z_e = position[2]

# In this shifted frame: Earth is at x=0 and Sun is at x=-1.
# For general reference distance L, Sun is at x = -d_earth_sun / L = -rho_es.
sun_dx = x_e + rho_es
earth_dx = x_e

d_sun = ox.Sqrt(sun_dx**2 + y_e**2 + z_e**2)
d_earth = ox.Sqrt(earth_dx**2 + y_e**2 + z_e**2)

ax = (
    2.0 * kappa * velocity[1]
    + kappa**2 * (x_e + rho_es * (1.0 - mu))
    - grav_scale * (mu_sun * sun_dx / d_sun**3 + mu_earth * earth_dx / d_earth**3)
)
ay = (
    -2.0 * kappa * velocity[0]
    + kappa**2 * y_e
    - grav_scale * (mu_sun * y_e / d_sun**3 + mu_earth * y_e / d_earth**3)
)
az = -grav_scale * (mu_sun * z_e / d_sun**3 + mu_earth * z_e / d_earth**3)

velocity_dot = ox.Concat(ax, ay, az)
dynamics = {
    "position": velocity,
    "velocity": velocity_dot,
    "fuel": 0.0,
}

delta_v = ox.Control(
    "delta_v",
    shape=(3,),
    parameterization="impulsive",
    nodes=[0, n_nodes - 1],
)

eps_impulse = 1e-12
dynamics_discrete = {
    "position": position,
    "velocity": velocity + delta_v,
    "fuel": fuel - ox.linalg.Norm(delta_v + eps_impulse),
}

cr3bp_rhs = lower_to_jax(ox.Concat(velocity, velocity_dot))
# Dense propagation for an initialization trajectory.
guess_dense = np.asarray(
    solve_ivp_diffrax(
        lambda t, x: cr3bp_rhs(x, jnp.zeros((0,), dtype=x.dtype), 0, {}),
        tau_final=t_f_guess,
        y_0=jnp.asarray(x0_seed, dtype=jnp.float64),
        args=(),
        tau_0=0.0,
        num_substeps=3000,
        solver_name="Dopri8",
        rtol=integration_tol,
        atol=integration_tol,
    ),
    dtype=float,
)

# Build nodal guess and apply the pre-impulse state offset at node 0.
s_uniform = np.linspace(0.0, 1.0, n_nodes)
node_grid = _normalized_node_grid(n_nodes, NODE_DISTRIBUTION_MODE)
node_idx = np.round((guess_dense.shape[0] - 1) * node_grid).astype(int)
nodal_guess = guess_dense[node_idx].copy()
# nodal_guess[0, 3:6] -= delta_v0_guess

# Broad bounds (required by OpenSCvx for robust scaling/bounding).
position.min = np.array([-2.0, -2.0, -2.0])
position.max = np.array([2.0, 2.0, 2.0])
velocity.min = np.array([-3.0, -3.0, -3.0])
velocity.max = np.array([3.0, 3.0, 3.0])
fuel.min = np.array([0.95])
fuel.max = np.array([1.00])

# Boundary conditions
position.initial = pos_0
velocity.initial = vel_0
fuel.initial = np.array([1.0])

position.final = [
    ox.Free(float(pos_f[0])),
    ox.Free(float(pos_f[1])),
    ox.Free(float(pos_f[2])),
]
velocity.final = [
    ox.Free(float(vel_f[0])),
    ox.Free(float(vel_f[1])),
    ox.Free(float(vel_f[2])),
]
fuel.final = [("maximize", 0.95)]

# Guesses
position.guess = nodal_guess[:, :3]
velocity.guess = nodal_guess[:, 3:6]
fuel.guess = np.ones((n_nodes, 1))

delta_v.min = -np.ones(3)
delta_v.max = np.ones(3)
delta_v_guess = np.zeros((n_nodes, 3))
delta_v.guess = delta_v_guess

time_guess = (t_f_guess * node_grid).reshape(-1, 1)
time = ox.Time(
    initial=0.0,
    final=ox.Free(float(t_f_guess)),
    min=0.0,
    max=3.0 * t_f_guess,
    guess=time_guess,
    time_dilation_min=0.01 * t_f_guess,
    time_dilation_max=3.0 * t_f_guess,
    uniform_time_grid=False,
)
dtdtau_guess = np.gradient(time_guess[:, 0], s_uniform)
dtdtau_guess = np.clip(dtdtau_guess, 0.01 * t_f_guess, 3.0 * t_f_guess)
time.time_dilation_guess = dtdtau_guess.reshape(-1, 1)

states = [position, velocity, fuel]
controls = [delta_v]

discretizer = {
    "ode_solver": "Dopri8",
    "diffrax_kwargs": {"atol": integration_tol, "rtol": integration_tol},
}
algorithm = {
    "k_max": 150,
    "lam_prox": 5e-2,
    "lam_vc": 3e1,
    "lam_vb": 2e-1,
    "lam_cost": 5e-1,
    "ep_tr": 1e-9,
    "ep_vc": 1e-6,
    "autotuner": ox.AugmentedLagrangian(),
}

constraints = []
# Enforce final distance from Earth in normalized Sun-Earth rotating frame.
final_radius_target = d_earth_moon / r_ref
eps_radius          = 1e-4
constraints += [
    (ox.linalg.Norm(position - pos_earth_rot) <= final_radius_target).at([n_nodes - 1]).convex(),
]
constraints += [
    (ox.linalg.Norm(position - pos_earth_rot) >= (1+eps_radius)*final_radius_target)
                                                                                .at([n_nodes - 1]),
]

# Final orbit tangency: radius and velocity orthogonal at terminal node.
constraints += [
    (ox.Sum((position - pos_earth_rot) * velocity) >= 0.0).at([n_nodes - 1]),
]
constraints += [
    (ox.Sum((position - pos_earth_rot) * velocity) <= 0.0).at([n_nodes - 1]),
]

# Final speed magnitude: velocity should match the moon one
constraints += [
    (ox.linalg.Norm(velocity) - v_moon >= 0.0).at([n_nodes - 1]),
]
constraints += [
    (ox.linalg.Norm(velocity) - v_moon <= 0.0).at([n_nodes - 1]).convex(),
]


problem = Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n_nodes,
    discretizer=discretizer,
    algorithm=algorithm,
    float_dtype="float64",
    solver={"cvx_solver": "CLARABEL"},
)

# Keep post-process propagation tolerances aligned with discretization.
problem.settings.prp.solver = "Dopri8"
problem.settings.prp.atol = integration_tol
problem.settings.prp.rtol = integration_tol

if __name__ == "__main__":
    print(f"Ephemeris source: {spice_source}")
    print(f"Reference date: {REFERENCE_DATE}")
    print(f"Reference scales: L={r_ref:.6f} km, T={t_ref:.6f} s, v_ref={v_ref:.9f} km/s")
    hohmann_metrics = _hohmann_transfer_metrics(
        mu_central_km3_s2=mu_earth,
        r1_km=r_0,
        r2_km=d_earth_moon,
    )

    x0_guess_post = x0_seed.copy()
    traj_guess = np.asarray(
        solve_ivp_diffrax(
            lambda t, x: cr3bp_rhs(x, jnp.zeros((0,), dtype=x.dtype), 0, {}),
            tau_final=t_f_guess,
            y_0=jnp.asarray(x0_guess_post, dtype=jnp.float64),
            args=(),
            tau_0=0.0,
            num_substeps=3000,
            solver_name="Dopri8",
            rtol=integration_tol,
            atol=integration_tol,
        ),
        dtype=float,
    )

    guess_results = OptimizationResults(converged=True, t_final=float(t_f_guess))
    guess_results.trajectory = {
        "time": np.linspace(0.0, t_f_guess, traj_guess.shape[0]).reshape(-1, 1),
        "position": traj_guess[:, :3] * r_ref,
        "velocity": traj_guess[:, 3:6] * v_ref,
    }
    guess_results.nodes = {
        "time": time_guess,
        "position": nodal_guess[:, :3] * r_ref,
        "velocity": nodal_guess[:, 3:6] * v_ref,
    }
    fig_guess = plot_projections_2d(guess_results, velocity_var_name="velocity")
    _set_projection_axis_labels_km(fig_guess)
    _set_projection_speed_colorbar_kms(fig_guess)
    fig_guess.update_layout(title="LET Initial Guess - XY, XZ, YZ Projections (km)")
    fig_guess.show()

    problem.initialize()

    results = problem.solve()
    results = problem.post_process()

    fig_states = plot_states(results, ["position", "velocity", "fuel"], cols=3)
    fig_states.update_layout(title_text="LET Solution - State Evolution")
    fig_states.update_xaxes(title_text="Time (normalized)")
    fig_states.show()

    t_f_opt = float(np.asarray(results.nodes["time"][-1]).squeeze())
    dv0_opt = np.asarray(results.nodes["delta_v"][0], dtype=float)
    dvf_opt = np.asarray(results.nodes["delta_v"][-1], dtype=float)

    x0_opt_pre = np.concatenate([results.nodes["position"][0], results.nodes["velocity"][0]])
    x0_opt_post = x0_opt_pre.copy()
    x0_opt_post[3:6] += dv0_opt
    traj_solution = np.asarray(
        solve_ivp_diffrax(
            lambda t, x: cr3bp_rhs(x, jnp.zeros((0,), dtype=x.dtype), 0, {}),
            tau_final=t_f_opt,
            y_0=jnp.asarray(x0_opt_post, dtype=jnp.float64),
            args=(),
            tau_0=0.0,
            num_substeps=3000,
            solver_name="Dopri8",
            rtol=integration_tol,
            atol=integration_tol,
        ),
        dtype=float,
    )

    solution_results = OptimizationResults(converged=bool(results.converged), t_final=t_f_opt)
    solution_results.trajectory = {
        "time": np.linspace(0.0, t_f_opt, traj_solution.shape[0]).reshape(-1, 1),
        "position": traj_solution[:, :3] * r_ref,
        "velocity": traj_solution[:, 3:6] * v_ref,
    }
    solution_results.nodes = {
        "time": results.nodes["time"],
        "position": np.asarray(results.nodes["position"], dtype=float) * r_ref,
        "velocity": np.asarray(results.nodes["velocity"], dtype=float) * v_ref,
    }
    fig_solution = plot_projections_2d(solution_results, velocity_var_name="velocity")
    _set_projection_axis_labels_km(fig_solution)
    _set_projection_speed_colorbar_kms(fig_solution)
    fig_solution.update_layout(title="LET Solution - XY, XZ, YZ Projections (km)")
    _add_moon_orbit_overlay(
        fig_solution,
        earth_pos=pos_earth_rot * r_ref,
        moon_radius=d_earth_moon,
    )
    fig_solution.show()

    final_pos = traj_solution[-1, :3]
    final_vel = traj_solution[-1, 3:6]
    final_radius_vec = final_pos - pos_earth_rot
    final_radius_norm = float(np.linalg.norm(final_radius_vec))
    final_speed_norm = float(np.linalg.norm(final_vel))
    final_distance_norm = final_radius_norm
    final_distance_km = final_distance_norm * r_ref
    dv0_guess_norm = float(np.linalg.norm(delta_v0_guess))
    dv0_opt_norm = float(np.linalg.norm(dv0_opt))
    dvf_opt_norm = float(np.linalg.norm(dvf_opt))
    total_dv_opt_norm = dv0_opt_norm + dvf_opt_norm
    total_dv_with_guess_norm = dv0_guess_norm + total_dv_opt_norm

    print(f"Converged: {bool(results.converged)}")
    print(f"Final time (days): {t_f_opt * t_ref / d_2_sec:.6f}")
    print(f"||delta_v0_guess|| (km/s): {dv0_guess_norm * v_ref:.9f}")
    print(f"Initial delta-v (km/s): {dv0_opt * v_ref}")
    print(f"||Initial delta-v|| (km/s): {dv0_opt_norm * v_ref:.9f}")
    print(f"Final delta-v (km/s): {dvf_opt * v_ref}")
    print(f"||Final delta-v|| (km/s): {dvf_opt_norm * v_ref:.9f}")
    print(f"Total ||delta-v|| (solution only, km/s): {total_dv_opt_norm * v_ref:.9f}")
    print(f"Total ||delta-v|| (+delta_v0_guess, km/s): {total_dv_with_guess_norm * v_ref:.9f}")
    print(f"Hohmann dv1 (km/s): {hohmann_metrics['dv1_km_s']:.9f}")
    print(f"Hohmann dv2 (km/s): {hohmann_metrics['dv2_km_s']:.9f}")
    print(f"Hohmann total delta-v (km/s): {hohmann_metrics['total_dv_km_s']:.9f}")

    if ENABLE_VISER_ANIMATION:
        traj_time_days = np.linspace(0.0, t_f_opt * t_ref / d_2_sec, traj_solution.shape[0])
        traj_solution_vis, traj_time_days_vis = _resample_trajectory_for_viser(
            traj_solution, traj_time_days
        )
        traj_guess_time_days = np.linspace(0.0, t_f_guess * t_ref / d_2_sec, traj_guess.shape[0])
        traj_guess_vis, traj_guess_time_days_vis = _resample_trajectory_for_viser(
            traj_guess, traj_guess_time_days
        )
        moon_rate_rad_per_day = np.sqrt(mu_earth / d_earth_moon**3) * d_2_sec
        viser_server = _create_let_viser_server(
            trajectory=traj_solution_vis,
            traj_time_days=traj_time_days_vis,
            earth_pos=pos_earth_rot,
            sun_pos=np.array([-rho_es, 0.0, 0.0], dtype=float),
            moon_radius=d_earth_moon / r_ref,
            moon_rate_rad_per_day=moon_rate_rad_per_day,
            guess_trajectory=traj_guess_vis,
            port=VISER_ROTATING_PORT,
        )
        inertial_server = None
        if ENABLE_VISER_INERTIAL_ANIMATION:
            inertial_server = _create_let_viser_server_inertial(
                trajectory=traj_solution_vis,
                traj_time_days=traj_time_days_vis,
                r_ref_km=r_ref,
                d_earth_sun_km=d_earth_sun,
                d_earth_moon_km=d_earth_moon,
                moon_rate_rad_per_day=moon_rate_rad_per_day,
                kappa_val=kappa,
                guess_trajectory=traj_guess_vis,
                guess_time_days=traj_guess_time_days_vis,
                port=VISER_INERTIAL_PORT,
            )

        if viser_server is not None or inertial_server is not None:
            rotating_url = f"http://localhost:{VISER_ROTATING_PORT}"
            inertial_url = f"http://localhost:{VISER_INERTIAL_PORT}"
            print("Launching viser animation server(s) (Ctrl+C to exit)...")
            if viser_server is not None:
                print(f"Rotating frame viewer: {rotating_url}")
                if VISER_REQUEST_SHARE_URLS:
                    try:
                        rotating_share_url = viser_server.request_share_url(verbose=True)
                        if rotating_share_url is not None:
                            print(f"Rotating frame public URL: {rotating_share_url}")
                    except Exception as exc:
                        print(f"Rotating frame share URL unavailable: {exc}")
            if inertial_server is not None:
                print(f"Sun-centered inertial viewer: {inertial_url}")
                if VISER_REQUEST_SHARE_URLS:
                    try:
                        inertial_share_url = inertial_server.request_share_url(verbose=True)
                        if inertial_share_url is not None:
                            print(f"Sun-centered inertial public URL: {inertial_share_url}")
                    except Exception as exc:
                        print(f"Sun-centered inertial share URL unavailable: {exc}")
            while True:
                pytime.sleep(1.0)
