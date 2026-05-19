"""Relative loitering in Earth-Moon CR3BP with CTCS box-violation regularization.

Port of `p_06_BRO_CTCS` to OpenSCvx with the following design:
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
    from openscvx.plotting import plot_projections_2d, plot_states
except Exception:
    plot_projections_2d = None
    plot_states = None


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
target_num_substeps = 4000


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
            "k_max": 150,
            "lam_prox": 4e-1,
            "lam_vc": 1e2,
            "lam_cost": 1e0,
            "ep_vc": 1e-8,
            "ep_tr": 1e-8,
            "autotuner": ox.AugmentedLagrangian(),
        },
        solver={"cvx_solver": "CLARABEL", "solver_args": {}},
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


if __name__ == "__main__":
    _enable_jax_x64()
    problem, context = build_relative_loitering_problem(force_recompute_halo=False, verbose=True)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    _print_solution_summary(results, context)

    if (plot_projections_2d is None) or (plot_states is None):
        print(
            "Skipping plotting because plotting dependencies failed to import. "
            "Check NumPy/Matplotlib compatibility in your environment."
        )
    else:
        fig_proj = plot_projections_2d(results, velocity_var_name="velocity")
        fig_proj.update_layout(title="Relative Loitering Solution - XY, XZ, YZ Projections")
        fig_proj.show()

        fig_states = plot_states(results, ["position", "velocity", "time"], cols=3)
        _apply_kiz_limits_to_state_plot(fig_states, kiz_box_width_dyn)
        fig_states.update_layout(title_text="Relative Loitering - State Evolution")
        fig_states.show()
