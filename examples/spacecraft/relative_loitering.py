"""Relative loitering in Earth-Moon CR3BP with CTCS box-violation regularization.

- Relative dynamics around a target halo-like trajectory in CR3BP rotating frame
- Three impulsive burns at nodes [0, 1, 2]
- Free final time with maximization objective
- Nodal keep-in-zone (KIZ) box constraints plus CTCS smooth-ReLU penalties
- Strict local SPICE loading from `examples/spacecraft/ker`
- Target initial condition x0_t loaded from `ker/halo_orbit_x0_t.npz`;
  if missing/invalid, computed once via `halo_orbit.py` and saved there
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add grandparent directory to path to import openscvx without installation.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.integrators import solve_ivp_diffrax

try:
    from openscvx.plotting import plot_states
except Exception:
    plot_states = None

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception:
    mpl = None
    plt = None


def _enable_jax_x64() -> None:
    """Enable float64 in JAX for high-accuracy propagation."""
    jax.config.update("jax_enable_x64", True)


reference_date = "2024-08-28T00:00:00"
kernel_dir = Path(current_dir) / "ker"
kernel_urls = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de440.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
    "pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
    "gm_de440.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
}
kernel_filenames = tuple(kernel_urls.keys())
halo_x0_file = kernel_dir / "halo_orbit_x0_t.npz"

# Problem constants (normalized CR3BP units)
n_nodes = 4
impulse_nodes = [0, 1, 2]
tf_guess = 1.522
time_max_factor = 6.0
time_max = time_max_factor * tf_guess
max_deltav = 2.5e-4
r_ref_km = 384400.0
kiz_box_width_km = 15.0
kiz_box_width_dyn = kiz_box_width_km / r_ref_km

integration_tol = 1e-14
target_num_substeps = 10000

# Figure output settings for the matplotlib 3D + 2D projection plots.
_FIGURE_DIR = Path(current_dir) / "figures"
_FIGURE_DPI = 600

# Shared styling palette for the matplotlib 3D + 2D projection plots.
_KIZ_FACE = (76 / 255.0, 175 / 255.0, 80 / 255.0, 0.08)
_KIZ_EDGE = (76 / 255.0, 175 / 255.0, 80 / 255.0, 0.85)
_KIZ_EDGE_WIDTH = 0.6
_IMPULSE_COLORS = ("#D62728", "#FF7F0E", "#9467BD")

# CAPSTONE mesh assets for impulse markers in the 3D plot.
_CAPSTONE_ASSET_DIR = Path(current_dir) / "capstone"
_CAPSTONE_GLTF_PATH = _CAPSTONE_ASSET_DIR / "capstone.gltf"
_CAPSTONE_MESH_CACHE_PATH = _CAPSTONE_ASSET_DIR / "capstone_plot_mesh.npz"
_CAPSTONE_TARGET_LENGTH_KM = 5.0


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
    missing = [name for name in kernel_filenames if not (kernel_dir / name).is_file()]
    if not missing:
        return

    download_errors = []
    for kernel_name in missing:
        destination = kernel_dir / kernel_name
        try:
            _download_kernel(kernel_urls[kernel_name], destination)
        except Exception as exc:
            part_file = destination.with_suffix(destination.suffix + ".part")
            if part_file.exists():
                part_file.unlink()
            download_errors.append(f"{kernel_name}: {exc}")

    if download_errors:
        raise RuntimeError("Failed to download SPICE kernels: " + "; ".join(download_errors))


def _load_spice_mu_from_local_kernels(kernel_dir: Path) -> tuple[float, float]:
    """Load Earth/Moon GM constants from local SPICE kernels only."""
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError(
            "spiceypy is required for examples/spacecraft/relative_loitering.py. "
            "Install it with: pip install spiceypy"
        ) from exc

    _ensure_spice_kernels(kernel_dir)

    spice.kclear()
    try:
        for kernel_name in kernel_filenames:
            spice.furnsh(str(kernel_dir / kernel_name))

        mu_earth_val = float(spice.bodvrd("EARTH", "GM", 1)[1][0])
        mu_moon_val = float(spice.bodvrd("MOON", "GM", 1)[1][0])
        return mu_earth_val, mu_moon_val
    finally:
        spice.kclear()


def _cr3bp_rhs_numeric(x: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Absolute CR3BP dynamics in rotating frame for numeric integration."""
    x_e, y_e, z_e, vx_e, vy_e, vz_e = x

    r1x = x_e + mu
    r2x = x_e - (1.0 - mu)
    d1 = jnp.sqrt(r1x**2 + y_e**2 + z_e**2)
    d2 = jnp.sqrt(r2x**2 + y_e**2 + z_e**2)

    ax = 2.0 * vy_e + x_e - (1.0 - mu) * r1x / d1**3 - mu * r2x / d2**3
    ay = -2.0 * vx_e + y_e - (1.0 - mu) * y_e / d1**3 - mu * y_e / d2**3
    az = -(1.0 - mu) * z_e / d1**3 - mu * z_e / d2**3

    return jnp.array([vx_e, vy_e, vz_e, ax, ay, az], dtype=x.dtype)


def _cr3bp_accel_expr(x_e, y_e, z_e, vx_e, vy_e, vz_e, mu: float):
    """Absolute CR3BP acceleration as symbolic expressions."""
    r1x = x_e + mu
    r2x = x_e - (1.0 - mu)
    d1 = ox.Sqrt(r1x**2 + y_e**2 + z_e**2)
    d2 = ox.Sqrt(r2x**2 + y_e**2 + z_e**2)

    ax = 2.0 * vy_e + x_e - (1.0 - mu) * r1x / d1**3 - mu * r2x / d2**3
    ay = -2.0 * vx_e + y_e - (1.0 - mu) * y_e / d1**3 - mu * y_e / d2**3
    az = -(1.0 - mu) * z_e / d1**3 - mu * z_e / d2**3
    return ax, ay, az


def _build_target_reference(
    mu: float, x0_t: np.ndarray, t_horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate target absolute trajectory over [0, t_horizon]."""
    traj = np.asarray(
        solve_ivp_diffrax(
            lambda t, x: _cr3bp_rhs_numeric(x, mu),
            tau_final=t_horizon,
            y_0=jnp.asarray(x0_t, dtype=jnp.float64),
            args=(),
            tau_0=0.0,
            num_substeps=target_num_substeps,
            solver_name="Dopri8",
            rtol=integration_tol,
            atol=integration_tol,
        ),
        dtype=float,
    )
    t_grid = np.linspace(0.0, t_horizon, traj.shape[0], dtype=float)
    return t_grid, traj


def _load_halo_x0_from_file(file_path: Path) -> np.ndarray:
    """Load halo initial condition from local NPZ file and validate shape."""
    with np.load(file_path, allow_pickle=False) as data:
        if "x0_t" not in data.files:
            raise KeyError(f"Missing 'x0_t' in {file_path}. Found keys: {data.files}")
        x0_t = np.asarray(data["x0_t"], dtype=np.float64).reshape(-1)
    if x0_t.shape != (6,):
        raise ValueError(f"Invalid x0_t shape in {file_path}: expected (6,), got {x0_t.shape}")
    return x0_t


def _get_target_x0_t(
    *,
    force_recompute_halo: bool,
    verbose: bool,
) -> tuple[np.ndarray, str]:
    """Load target x0_t from ker file when available, otherwise compute and cache it."""
    if not force_recompute_halo and halo_x0_file.is_file():
        try:
            x0_t = _load_halo_x0_from_file(halo_x0_file)
            if verbose:
                print(f"Loaded target x0_t from file: {halo_x0_file}")
            return x0_t, f"file:{halo_x0_file}"
        except Exception as exc:
            if verbose:
                print(f"Failed to read {halo_x0_file} ({exc}). Recomputing x0_t.")

    from examples.spacecraft.halo_orbit import get_halo_initial_condition

    x0_t = get_halo_initial_condition(force_recompute=force_recompute_halo, verbose=verbose)
    halo_x0_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(halo_x0_file, x0_t=np.asarray(x0_t, dtype=np.float64))
    if verbose:
        print(f"Saved target x0_t to file: {halo_x0_file}")
    return np.asarray(x0_t, dtype=np.float64), f"solve+save:{halo_x0_file}"


def build_relative_loitering_problem(
    *,
    force_recompute_halo: bool = False,
    verbose: bool = True,
) -> tuple[Problem, dict]:
    """Create the relative-loitering OpenSCvx problem and context dictionary."""
    mu_earth, mu_moon = _load_spice_mu_from_local_kernels(kernel_dir)
    mu = mu_moon / (mu_earth + mu_moon)

    x0_t, x0_t_source = _get_target_x0_t(
        force_recompute_halo=force_recompute_halo,
        verbose=verbose,
    )

    target_time_grid, target_traj = _build_target_reference(mu=mu, x0_t=x0_t, t_horizon=time_max)

    position = ox.State("position", shape=(3,))
    velocity = ox.State("velocity", shape=(3,))

    position._slice = slice(0, 3)
    velocity._slice = slice(3, 6)

    position.min = np.array([-2.0, -2.0, -2.0])
    position.max = np.array([2.0, 2.0, 2.0])
    velocity.min = np.array([-4.0, -4.0, -4.0])
    velocity.max = np.array([4.0, 4.0, 4.0])

    position.initial = np.array([1e-6, 0.0, 0.0])
    velocity.initial = np.array([0.0, 0.0, 0.0])

    position.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]
    velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

    position.guess = np.linspace(position.initial, np.zeros(3), n_nodes)
    velocity.guess = np.zeros((n_nodes, 3))

    delta_v = ox.Control(
        "delta_v",
        shape=(3,),
        parameterization="impulsive",
        nodes=impulse_nodes,
    )
    delta_v.min = -max_deltav * np.ones(3)
    delta_v.max = max_deltav * np.ones(3)
    delta_v.guess = np.zeros((n_nodes, 3))

    time_guess = np.linspace(0.0, tf_guess, n_nodes).reshape(-1, 1)
    time = ox.Time(
        initial=0.0,
        final=ox.Maximize(float(tf_guess)),
        min=0.0,
        max=time_max,
        guess=time_guess,
        time_dilation_min=0.1,
        time_dilation_max=time_max,
    )

    t_expr = time[0]
    x_t = ox.Cinterp(t_expr, target_time_grid, target_traj[:, 0])
    y_t = ox.Cinterp(t_expr, target_time_grid, target_traj[:, 1])
    z_t = ox.Cinterp(t_expr, target_time_grid, target_traj[:, 2])
    vx_t = ox.Cinterp(t_expr, target_time_grid, target_traj[:, 3])
    vy_t = ox.Cinterp(t_expr, target_time_grid, target_traj[:, 4])
    vz_t = ox.Cinterp(t_expr, target_time_grid, target_traj[:, 5])

    x_c = position[0] + x_t
    y_c = position[1] + y_t
    z_c = position[2] + z_t
    vx_c = velocity[0] + vx_t
    vy_c = velocity[1] + vy_t
    vz_c = velocity[2] + vz_t

    ax_c, ay_c, az_c = _cr3bp_accel_expr(x_c, y_c, z_c, vx_c, vy_c, vz_c, mu)
    ax_t, ay_t, az_t = _cr3bp_accel_expr(x_t, y_t, z_t, vx_t, vy_t, vz_t, mu)

    dynamics = {
        "position": velocity,
        "velocity": ox.Concat(ax_c - ax_t, ay_c - ay_t, az_c - az_t),
    }

    dynamics_discrete = {
        "position": position,
        "velocity": velocity + delta_v,
    }

    constraints = []
    all_nodes = list(range(n_nodes))

    constraints.append((position <= kiz_box_width_dyn).convex().at(all_nodes))
    constraints.append((position >= -kiz_box_width_dyn).convex().at(all_nodes))

    for node in impulse_nodes:
        constraints.append((ox.linalg.Norm(delta_v) <= max_deltav).convex().at([node]))

    ctcs_intervals = [(0, 1), (1, 2), (2, 3)]
    for idx, interval in enumerate(ctcs_intervals):
        for axis in range(3):
            constraints.append(
                (position[axis] - kiz_box_width_dyn <= 0.0).over(
                    interval,
                    penalty="smooth_relu",
                    idx=idx,
                )
            )
            constraints.append(
                (-position[axis] - kiz_box_width_dyn <= 0.0).over(
                    interval,
                    penalty="smooth_relu",
                    idx=idx,
                )
            )

    problem = Problem(
        dynamics=dynamics,
        dynamics_discrete=dynamics_discrete,
        states=[position, velocity],
        controls=[delta_v],
        time=time,
        constraints=constraints,
        N=n_nodes,
        licq_min=0.0,
        licq_max=1e-8,
        discretizer={
            "ode_solver": "Dopri8",
            "diffrax_kwargs": {"atol": integration_tol, "rtol": integration_tol},
        },
        algorithm={
            "k_max": 100,
            "lam_prox": 5e-1,
            "lam_vc": 2e2,
            "lam_cost": 1e0,
            "ep_vc": 5e-8,
            "ep_tr": 5e-8,
            "autotuner": ox.AugmentedLagrangian(),
        },
        solver={"cvx_solver": "MOSEK", "solver_args": {}},
        float_dtype="float64",
    )

    problem.settings.prp.solver = "Dopri8"
    problem.settings.prp.atol = integration_tol
    problem.settings.prp.rtol = integration_tol

    context = {
        "mu_earth": float(mu_earth),
        "mu_moon": float(mu_moon),
        "mu": float(mu),
        "x0_t": np.asarray(x0_t, dtype=float),
        "x0_t_source": x0_t_source,
        "target_time_grid": target_time_grid,
        "target_traj": target_traj,
    }
    return problem, context


def _print_solution_summary(results, context: dict) -> None:
    """Print compact mission and constraint diagnostics."""
    t_final = float(np.asarray(results.nodes["time"][-1]).squeeze())
    delta_v_nodes = np.asarray(results.nodes["delta_v"], dtype=float)

    impulse_norms = np.linalg.norm(delta_v_nodes[impulse_nodes], axis=1)
    total_impulse = float(np.sum(impulse_norms))

    pos_nodes = np.asarray(results.nodes["position"], dtype=float)
    max_abs_pos = np.max(np.abs(pos_nodes), axis=0)

    print(f"Reference date: {reference_date}")
    print(f"SPICE kernels: {kernel_dir}")
    print(f"Target x0_t source: {context['x0_t_source']}")
    print(f"Converged: {bool(results.converged)}")
    print(f"Final time: {t_final:.9f}")

    for local_idx, node in enumerate(impulse_nodes):
        print(f"||delta_v|| at node {node}: {impulse_norms[local_idx]:.9e}")
    print(f"Total ||delta_v|| over impulse nodes: {total_impulse:.9e}")

    print(
        "Max |position| at nodes: "
        f"x={max_abs_pos[0]:.9e}, y={max_abs_pos[1]:.9e}, z={max_abs_pos[2]:.9e} "
        f"(KIZ bound={kiz_box_width_dyn:.9e})"
    )

    for idx in range(3):
        ctcs_name = f"_ctcs_aug_{idx}"
        if ctcs_name in results.nodes:
            ctcs_vals = np.asarray(results.nodes[ctcs_name], dtype=float).flatten()
            print(f"{ctcs_name} max: {np.max(ctcs_vals):.9e}")


def _apply_kiz_limits_to_state_plot(fig, kiz_half_width: float) -> None:
    """Clamp position subplots to KIZ scale and draw explicit KIZ bounds."""
    y_lim = 1.1 * kiz_half_width
    for col in (1, 2, 3):
        fig.update_yaxes(range=[-y_lim, y_lim], row=1, col=col)
        fig.add_hrect(
            y0=-kiz_half_width,
            y1=kiz_half_width,
            fillcolor="rgba(76, 175, 80, 0.12)",
            line_width=0,
            layer="below",
            row=1,
            col=col,
        )
        fig.add_hline(
            y=kiz_half_width,
            line_dash="dash",
            line_color="rgba(200, 0, 0, 0.9)",
            line_width=2,
            row=1,
            col=col,
        )
        fig.add_hline(
            y=-kiz_half_width,
            line_dash="dash",
            line_color="rgba(200, 0, 0, 0.9)",
            line_width=2,
            row=1,
            col=col,
        )


def _savefig_hi_dpi(fig, basename: str) -> None:
    """Save fig as a high-DPI PNG in ``_FIGURE_DIR``."""
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = _FIGURE_DIR / f"{basename}.png"
    fig.savefig(png_path, dpi=_FIGURE_DPI, bbox_inches="tight", transparent=True)
    print(f"Saved {png_path}")


def _ensure_texlive_on_path() -> None:
    """Prepend a detected TeX Live bin dir to PATH so text.usetex=True works."""
    if shutil.which("latex") is not None:
        return
    candidates = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2024/bin/universal-darwin",
        "/usr/local/texlive/2024/bin/x86_64-darwin",
        "/usr/local/texlive/2023/bin/universal-darwin",
        "/opt/homebrew/bin",
    ]
    for d in candidates:
        if Path(d, "latex").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return


def _apply_serif_mpl_rc() -> None:
    """Shared mpl rcParams: real LaTeX rendering with Computer Modern serif."""
    _ensure_texlive_on_path()
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def _turbo_line_collection_3d(xyz: np.ndarray, t: np.ndarray, linewidth: float = 2.4):
    """Per-segment Line3DCollection colored by t via viridis colormap."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pts = xyz.reshape(-1, 1, 3)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = mpl.colors.Normalize(vmin=float(t[0]), vmax=float(t[-1]))
    cmap = mpl.colormaps["viridis"]
    t_mid = 0.5 * (t[:-1] + t[1:])
    return Line3DCollection(segments, colors=cmap(norm(t_mid)), linewidth=linewidth), norm, cmap


def _turbo_line_collection_2d(xy: np.ndarray, t: np.ndarray, linewidth: float = 2.4):
    """Per-segment 2D LineCollection colored by t via viridis colormap."""
    from matplotlib.collections import LineCollection

    pts = xy.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = mpl.colors.Normalize(vmin=float(t[0]), vmax=float(t[-1]))
    cmap = mpl.colormaps["viridis"]
    t_mid = 0.5 * (t[:-1] + t[1:])
    return LineCollection(segments, colors=cmap(norm(t_mid)), linewidth=linewidth), norm, cmap


def _load_capstone_plot_mesh() -> tuple[np.ndarray, np.ndarray] | None:
    """Load CAPSTONE mesh for 3D impulse markers, scaled to plot units (km)."""
    if _CAPSTONE_MESH_CACHE_PATH.is_file():
        try:
            with np.load(_CAPSTONE_MESH_CACHE_PATH, allow_pickle=False) as cached:
                cache_scale = None
                if "target_length_km" in cached.files:
                    cache_scale = float(np.asarray(cached["target_length_km"]).reshape(-1)[0])
                scale_matches = (
                    cache_scale is not None
                    and abs(cache_scale - _CAPSTONE_TARGET_LENGTH_KM) < 1e-12
                )
                if scale_matches:
                    return np.asarray(cached["vertices"], dtype=np.float64), np.asarray(
                        cached["faces"], dtype=np.int64
                    )
        except Exception:
            pass

    if not _CAPSTONE_GLTF_PATH.is_file():
        print(
            "CAPSTONE model file not found at "
            f"{_CAPSTONE_GLTF_PATH}; using point markers for impulses."
        )
        return None

    try:
        import trimesh
    except ImportError:
        print("Install trimesh to render CAPSTONE CAD markers (pip install trimesh).")
        return None

    try:
        loaded = trimesh.load(str(_CAPSTONE_GLTF_PATH), force="scene")
        mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    except Exception as exc:
        print(f"Failed to load CAPSTONE mesh ({exc}); using point markers for impulses.")
        return None

    if vertices.size == 0 or faces.size == 0:
        print("CAPSTONE mesh is empty; using point markers for impulses.")
        return None

    vertices = vertices - np.mean(vertices, axis=0)
    length = float(np.max(np.ptp(vertices, axis=0)))
    if length > 1e-12:
        vertices = vertices * (_CAPSTONE_TARGET_LENGTH_KM / length)

    try:
        np.savez(
            _CAPSTONE_MESH_CACHE_PATH,
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.uint32),
            target_length_km=np.array([_CAPSTONE_TARGET_LENGTH_KM], dtype=np.float64),
        )
    except Exception:
        pass

    return vertices, faces


def _rotation_matrix_from_vectors(source_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix that maps source_vec direction onto target_vec."""
    src = np.asarray(source_vec, dtype=np.float64).reshape(3)
    dst = np.asarray(target_vec, dtype=np.float64).reshape(3)

    src_norm = float(np.linalg.norm(src))
    dst_norm = float(np.linalg.norm(dst))
    if src_norm < 1e-12 or dst_norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    src_u = src / src_norm
    dst_u = dst / dst_norm
    cross = np.cross(src_u, dst_u)
    dot = float(np.clip(np.dot(src_u, dst_u), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-deg rotation: choose any axis orthogonal to source.
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src_u[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = ortho - np.dot(ortho, src_u) * src_u
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            ortho = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            axis = ortho - np.dot(ortho, src_u) * src_u
            axis_norm = float(np.linalg.norm(axis))
        axis = axis / axis_norm
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    k = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + k + (k @ k) * ((1.0 - dot) / (cross_norm**2))


def _add_spacecraft_mesh_marker(
    ax,
    center_km,
    vertices_km,
    faces,
    major_axis,
    impulse_direction,
) -> None:
    """Draw one white CAPSTONE mesh marker centered at a 3D point."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rot = _rotation_matrix_from_vectors(major_axis, impulse_direction)
    verts_rot = np.asarray(vertices_km, dtype=np.float64) @ rot.T
    tris = verts_rot[faces] + np.asarray(center_km, dtype=np.float64).reshape(1, 1, 3)
    fc = mpl.colors.to_rgba("#FFFFFF", alpha=0.96)
    ec = (0.1, 0.1, 0.1, 0.5)
    marker = Poly3DCollection(
        tris,
        facecolors=fc,
        edgecolors=ec,
        linewidths=0.08,
        zorder=10,
    )
    ax.add_collection3d(marker)


def _plot_3d_kiz(results) -> None:
    """Render 3D nominal trajectory in KIZ with robust-plot styling."""
    if (mpl is None) or (plt is None):
        print("Skipping 3D projection plot because Matplotlib is unavailable.")
        return
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _apply_serif_mpl_rc()

    nom_pos = np.asarray(results.trajectory["position"], dtype=float)
    t_full = np.asarray(results.trajectory["time"], dtype=float).reshape(-1)
    node_pos = np.asarray(results.nodes["position"], dtype=float)
    delta_v_nodes = np.asarray(results.nodes["delta_v"], dtype=float)

    scale_km = float(r_ref_km)
    nom_scaled = nom_pos * scale_km
    node_scaled = node_pos * scale_km
    nominal_imp_pts = node_scaled[np.asarray(impulse_nodes, dtype=int), :]

    fig = plt.figure(figsize=(11.0, 9.0), facecolor="none")
    fig.patch.set_alpha(0.0)
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    except TypeError:
        ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.grid(True, alpha=0.25)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

    w = float(kiz_box_width_km)
    corners = np.array(
        [
            [-w, -w, -w],
            [w, -w, -w],
            [w, w, -w],
            [-w, w, -w],
            [-w, -w, w],
            [w, -w, w],
            [w, w, w],
            [-w, w, w],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]],
    ]
    kiz_surface = Poly3DCollection(faces, facecolors=_KIZ_FACE, edgecolors="none", zorder=1)
    ax.add_collection3d(kiz_surface)
    for e0, e1 in edges:
        ax.plot(
            [corners[e0, 0], corners[e1, 0]],
            [corners[e0, 1], corners[e1, 1]],
            [corners[e0, 2], corners[e1, 2]],
            color=_KIZ_EDGE,
            linewidth=_KIZ_EDGE_WIDTH,
            alpha=0.9,
        )

    lc3d, norm, cmap = _turbo_line_collection_3d(nom_scaled, t_full, linewidth=2.4)
    ax.add_collection3d(lc3d)

    capstone_mesh = _load_capstone_plot_mesh()
    capstone_major_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if capstone_mesh is not None:
        centered = capstone_mesh[0] - np.mean(capstone_mesh[0], axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            major_axis = np.asarray(vh[0], dtype=np.float64)
            major_axis_norm = float(np.linalg.norm(major_axis))
            if major_axis_norm > 1e-12:
                capstone_major_axis = major_axis / major_axis_norm
                if capstone_major_axis[0] < 0.0:
                    capstone_major_axis = -capstone_major_axis
        except Exception:
            pass

    legend_handles = []
    arrow_length_km = 0.35 * w
    for q, node_q in enumerate(impulse_nodes):
        color = _IMPULSE_COLORS[q % len(_IMPULSE_COLORS)]
        impulse_direction = np.asarray(delta_v_nodes[node_q], dtype=np.float64)
        impulse_norm = float(np.linalg.norm(impulse_direction))
        if impulse_norm > 1e-14:
            dir_unit = impulse_direction / impulse_norm
            ax.quiver(
                nominal_imp_pts[q, 0],
                nominal_imp_pts[q, 1],
                nominal_imp_pts[q, 2],
                dir_unit[0],
                dir_unit[1],
                dir_unit[2],
                length=arrow_length_km,
                normalize=True,
                arrow_length_ratio=0.25,
                color=color,
                linewidths=2.0,
            )

    for q, node_q in enumerate(impulse_nodes):
        color = _IMPULSE_COLORS[q % len(_IMPULSE_COLORS)]
        impulse_direction = np.asarray(delta_v_nodes[node_q], dtype=np.float64)
        if capstone_mesh is None:
            ax.scatter(
                nominal_imp_pts[q, 0],
                nominal_imp_pts[q, 1],
                nominal_imp_pts[q, 2],
                marker="o",
                s=180,
                color=color,
                edgecolors="black",
                linewidths=0.9,
                zorder=10,
            )
        else:
            _add_spacecraft_mesh_marker(
                ax=ax,
                center_km=nominal_imp_pts[q],
                vertices_km=capstone_mesh[0],
                faces=capstone_mesh[1],
                major_axis=capstone_major_axis,
                impulse_direction=impulse_direction,
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=9,
                label=f"Impulse {q}",
            )
        )

    lim = max(1.05 * w, 1.05 * float(np.max(np.abs(nom_scaled))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=10.0, azim=-10.0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=-0.04, shrink=0.7)
    cbar.set_label(r"$t\;[\mathrm{TU}]$", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.20, 0.83),
        frameon=False,
        fontsize=16,
    )
    fig.tight_layout()
    _savefig_hi_dpi(fig, "relative_loitering_3d")
    plt.show()


def _plot_2d_projections(results) -> None:
    """XY and XZ projections with robust-plot styling."""
    if (mpl is None) or (plt is None):
        print("Skipping 2D projection plot because Matplotlib is unavailable.")
        return
    from matplotlib.patches import Rectangle

    _apply_serif_mpl_rc()

    nom_pos = np.asarray(results.trajectory["position"], dtype=float)
    t_full = np.asarray(results.trajectory["time"], dtype=float).reshape(-1)
    node_pos = np.asarray(results.nodes["position"], dtype=float)

    scale_km = float(r_ref_km)
    nom_km = nom_pos * scale_km
    node_km = node_pos * scale_km
    w = float(kiz_box_width_km)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor="none")
    fig.patch.set_alpha(0.0)
    axis_pairs = [(0, 1), (0, 2)]
    labels = [
        (r"$x\;[\mathrm{km}]$", r"$y\;[\mathrm{km}]$"),
        (r"$x\;[\mathrm{km}]$", r"$z\;[\mathrm{km}]$"),
    ]
    titles = [
        r"$xy$-projection",
        r"$xz$-projection",
    ]

    last_norm = None
    last_cmap = None
    for ax, (ai, aj), (xlbl, ylbl), title in zip(axes, axis_pairs, labels, titles):
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
        ax.grid(True, alpha=0.25)

        ax.add_patch(
            Rectangle(
                (-w, -w),
                2 * w,
                2 * w,
                facecolor=_KIZ_FACE,
                edgecolor=_KIZ_EDGE,
                linewidth=_KIZ_EDGE_WIDTH,
                zorder=1,
            )
        )

        xy = np.column_stack([nom_km[:, ai], nom_km[:, aj]])
        lc, norm, cmap = _turbo_line_collection_2d(xy, t_full, linewidth=2.2)
        lc.set_zorder(3)
        ax.add_collection(lc)
        last_norm, last_cmap = norm, cmap

        for q, node_q in enumerate(impulse_nodes):
            ax.scatter(
                node_km[node_q, ai],
                node_km[node_q, aj],
                marker="o",
                s=160,
                color=_IMPULSE_COLORS[q % len(_IMPULSE_COLORS)],
                edgecolors="black",
                linewidths=0.9,
                zorder=10,
                label=f"Impulse {q}" if ai == 0 and aj == 1 else None,
            )

        all_pts = nom_km[:, [ai, aj]]
        lim = max(1.05 * w, 1.05 * float(np.max(np.abs(all_pts))))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlbl, fontsize=15)
        ax.set_ylabel(ylbl, fontsize=15)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=12)

    if last_norm is not None:
        sm = mpl.cm.ScalarMappable(norm=last_norm, cmap=last_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02, shrink=0.85)
        cbar.set_label(r"$t\;[\mathrm{TU}]$", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    axes[0].legend(loc="upper right", frameon=False, fontsize=14)
    fig.suptitle(
        r"Relative Loitering: $xy$ and $xz$ projections",
        fontsize=16,
    )
    _savefig_hi_dpi(fig, "relative_loitering_xy_xz")
    plt.show()


if __name__ == "__main__":
    _enable_jax_x64()
    problem, context = build_relative_loitering_problem(force_recompute_halo=False, verbose=True)

    # Plot-only oversampling: shrink the post-processing save step so the
    # propagated trajectory used for plotting is denser.
    # Must be set BEFORE initialize() so max_tau_len is sized accordingly.
    plot_oversample_factor = 10
    problem.settings.prp.dt = problem.settings.prp.dt / plot_oversample_factor

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    _print_solution_summary(results, context)

    if (mpl is None) or (plt is None):
        print(
            "Skipping 3D/2D projection plotting because Matplotlib is unavailable. "
            "Check NumPy/Matplotlib compatibility in your environment."
        )
    else:
        _plot_3d_kiz(results)
        _plot_2d_projections(results)

    if plot_states is None:
        print(
            "Skipping state-time plotting because plotting dependencies failed to import. "
            "Check NumPy/Plotly compatibility in your environment."
        )
    else:
        fig_states = plot_states(results, ["position", "velocity", "time"], cols=3)
        _apply_kiz_limits_to_state_plot(fig_states, kiz_box_width_dyn)
        fig_states.update_layout(title_text="Relative Loitering - State Evolution")
        fig_states.show()
