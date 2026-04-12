"""Low-Energy Transfer (LET) setup in Sun-Earth CR3BP with one departure impulse.

Modeling choices:
- Sun-Earth CR3BP rotating-frame dynamics
- Impulsive delta-v at departure and at the final node (arrival burn)
- Fixed initial state, fixed final position, free final velocity
- Free final time with uniform time grid (single global dilation behavior)
- Objective: minimize total impulsive delta-v magnitude
"""

import os
import shutil
import sys
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
from openscvx.plotting import plot_projections_2d
from openscvx.symbolic.lower import lower_to_jax

# Use float64 in JAX for high-accuracy propagation.
jax.config.update("jax_enable_x64", True)

REFERENCE_DATE = "26 December 2025"
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
    with urllib.request.urlopen(url, timeout=120) as response, temp_destination.open("wb") as out_file:
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

spice_data = _load_spice_problem_data(REFERENCE_DATE)
mu_earth = spice_data["mu_earth"]
mu_sun = spice_data["mu_sun"]
r_earth = spice_data["r_earth"]
d_earth_sun = spice_data["d_earth_sun"]
d_earth_moon = spice_data["d_earth_moon"]
spice_source = f"SPICE ({spice_data['kernel_dir']})"

# Sun-Earth normalized CR3BP parameters
mu = mu_earth / (mu_earth + mu_sun)
r_ref = d_earth_sun
t_ref = 86400.0 * 365.0 / (2.0 * np.pi)
v_ref = r_ref / t_ref

# Mission setup
h_earth = 1500.0
r_0 = r_earth + h_earth
pos_earth_rot = np.array([1.0 - mu, 0.0, 0.0])
pos_0 = pos_earth_rot + np.array([r_0 / r_ref, 0.0, 0.0])
vel_0 = np.array([0.0, 7.8 * np.sqrt(2.0) * 0.8866 / v_ref, 0.0])
x0_seed = np.concatenate([pos_0, vel_0])

pos_f = pos_earth_rot + np.array([0.0, -d_earth_moon / r_ref, 0.0])
vel_f = np.array([ np.sqrt(mu_earth / d_earth_moon) / v_ref, 0.0, 0.0])
t_f_guess = 66.0 / 365.0 * (2.0 * np.pi)

# Initial impulse guess.
v_circular = np.sqrt(mu_earth / r_0) / v_ref
delta_v0_guess = np.array([0.0, vel_0[1] - v_circular, 0.0])

n_nodes = 30
integration_tol = 1e-12

# Guess-node distribution toggle:
# - "uniform": evenly spaced nodes in [0, 1]
# - "cosine": denser near interval endpoints
NODE_DISTRIBUTION_MODE = "cosine"


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

# Build symbolic CR3BP model once and reuse it for optimization and propagation.
position = ox.State("position", shape=(3,))
velocity = ox.State("velocity", shape=(3,))
fuel = ox.State("fuel", shape=(1,))

# Assign slices for standalone lowering/evaluation on [x, y, z, vx, vy, vz].
position._slice = slice(0, 3)
velocity._slice = slice(3, 6)

r1x = position[0] + mu
r1y = position[1]
r1z = position[2]
r2x = position[0] - (1.0 - mu)
r2y = position[1]
r2z = position[2]

d1 = ox.Sqrt(r1x**2 + r1y**2 + r1z**2)
d2 = ox.Sqrt(r2x**2 + r2y**2 + r2z**2)

ax = 2.0 * velocity[1] + position[0] - (1.0 - mu) * r1x / d1**3 - mu * r2x / d2**3
ay = -2.0 * velocity[0] + position[1] - (1.0 - mu) * r1y / d1**3 - mu * r2y / d2**3
az = -(1.0 - mu) * r1z / d1**3 - mu * r2z / d2**3

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
    "fuel": fuel + ox.linalg.Norm(delta_v + eps_impulse),
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
nodal_guess[0, 3:6] -= delta_v0_guess

# Broad bounds (required by OpenSCvx for robust scaling/bounding).
position.min = np.array([-2.0, -2.0, -2.0])
position.max = np.array([2.0, 2.0, 2.0])
velocity.min = np.array([-3.0, -3.0, -3.0])
velocity.max = np.array([3.0, 3.0, 3.0])
fuel.min = np.array([0.0])
fuel.max = np.array([5.0])

# Boundary conditions.
position.initial = pos_0
velocity.initial = vel_0
fuel.initial = np.array([0.0])

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
fuel.final = [("minimize", 1.0)]

# Guesses.
position.guess = nodal_guess[:, :3]
velocity.guess = nodal_guess[:, 3:6]
fuel.guess = np.zeros((n_nodes, 1))

delta_v.min = -np.ones(3)
delta_v.max = np.ones(3)
delta_v_guess = np.zeros((n_nodes, 3))
delta_v_guess[0, :] = delta_v0_guess
delta_v.guess = delta_v_guess

time_guess = (t_f_guess * node_grid).reshape(-1, 1)
time = ox.Time(
    initial=0.0,
    final=ox.Free(float(t_f_guess)),
    min=0.0,
    max=3.0 * t_f_guess,
    guess=time_guess,
    time_dilation_min=0.05 * t_f_guess,
    time_dilation_max=3.0 * t_f_guess,
    uniform_time_grid=False,
)
dtdtau_guess = np.gradient(time_guess[:, 0], s_uniform)
dtdtau_guess = np.clip(dtdtau_guess, 0.05 * t_f_guess, 3.0 * t_f_guess)
time.time_dilation_guess = dtdtau_guess.reshape(-1, 1)

states = [position, velocity, fuel]
controls = [delta_v]

discretizer = {
    "ode_solver": "Dopri8",
    "diffrax_kwargs": {"atol": integration_tol, "rtol": integration_tol},
}
algorithm = {
    "k_max": 50,
    "lam_prox": 1e0,
    "lam_vc": 1e1,
    "lam_vb": 1e0,
    "lam_cost": 1.0,
    "ep_tr": 1e-9,
    "ep_vc": 1e-6,
    "autotuner": ox.AugmentedLagrangian(),
}

# Enforce final distance from Earth in normalized Sun-Earth rotating frame.
final_radius_target = d_earth_moon / r_ref
constraints = [
    (ox.linalg.Norm(position - pos_earth_rot) <= final_radius_target).at([n_nodes - 1]),
]
constraints += [
    (ox.linalg.Norm(position - pos_earth_rot) >= final_radius_target).at([n_nodes - 1]),
]
# Final orbit tangency: radius and velocity orthogonal at terminal node.
constraints += [
    (ox.Sum((position - pos_earth_rot) * velocity) >= 0.0).at([n_nodes - 1]),
]
constraints += [
    (ox.Sum((position - pos_earth_rot) * velocity) <= 0.0).at([n_nodes - 1]),
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
    solver={"cvx_solver": "MOSEK", "solver_args":{}},

)

# Keep post-process propagation tolerances aligned with discretization.
problem.settings.prp.solver = "Dopri8"
problem.settings.prp.atol = integration_tol
problem.settings.prp.rtol = integration_tol

if __name__ == "__main__":
    print(f"Ephemeris source: {spice_source}")
    print(f"Reference date: {REFERENCE_DATE}")

    x0_guess_post = x0_seed.copy()
    # x0_guess_post[3:6] += delta_v0_guess
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
        "position": traj_guess[:, :3],
        "velocity": traj_guess[:, 3:6],
    }
    guess_results.nodes = {
        "time": time_guess,
        "position": nodal_guess[:, :3],
        "velocity": nodal_guess[:, 3:6],
    }
    fig_guess = plot_projections_2d(guess_results, velocity_var_name="velocity")
    fig_guess.update_layout(title="LET Initial Guess - XY, XZ, YZ Projections")
    fig_guess.show()

    problem.initialize()

    results = problem.solve()
    results = problem.post_process()

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
        "position": traj_solution[:, :3],
        "velocity": traj_solution[:, 3:6],
    }
    solution_results.nodes = {
        "time": results.nodes["time"],
        "position": results.nodes["position"],
        "velocity": results.nodes["velocity"],
    }
    fig_solution = plot_projections_2d(solution_results, velocity_var_name="velocity")
    fig_solution.update_layout(title="LET Solution - XY, XZ, YZ Projections")
    _add_moon_orbit_overlay(
        fig_solution,
        earth_pos=pos_earth_rot,
        moon_radius=d_earth_moon / r_ref,
    )
    fig_solution.show()

    final_pos = traj_solution[-1, :3]
    final_vel = traj_solution[-1, 3:6]
    final_radius_vec = final_pos - pos_earth_rot
    final_radius_norm = float(np.linalg.norm(final_radius_vec))
    final_speed_norm = float(np.linalg.norm(final_vel))
    final_speed_pos_orth = float(
        np.dot(final_radius_vec, final_vel) / max(final_radius_norm * final_speed_norm, 1e-12)
    )
    final_distance_norm = final_radius_norm
    final_distance_km = final_distance_norm * r_ref
    final_distance_error_km = final_distance_km - d_earth_moon
    moon_distance_match = bool(np.isclose(final_distance_km, d_earth_moon, atol=100.0))

    print(f"Converged: {bool(results.converged)}")
    print(f"Final time (normalized): {t_f_opt:.6f}")
    print(f"Initial delta-v (normalized): {dv0_opt}")
    print(f"Initial delta-v (km/s): {dv0_opt * v_ref}")
    print(f"Final delta-v (normalized): {dvf_opt}")
    print(f"Final delta-v (km/s): {dvf_opt * v_ref}")

