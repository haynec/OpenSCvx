"""Compute a halo-orbit initial condition x0_t."""

import os
import sys
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
from openscvx.symbolic.lower import lower_to_jax

# Use float64 in JAX for high-accuracy propagation.
jax.config.update("jax_enable_x64", True)

kernel_dir = Path(current_dir) / "ker"
kernel_filenames = ("gm_de440.tpc",)


def _load_spice_mu_from_local_kernels(kernel_dir: Path) -> tuple[float, float]:
    """Load Earth/Moon GM constants from local SPICE kernels."""
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError(
            "spiceypy is required for examples/spacecraft/halo_orbit.py. "
            "Install it with: pip install spiceypy"
        ) from exc

    missing = [name for name in kernel_filenames if not (kernel_dir / name).is_file()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required SPICE kernel files in '{kernel_dir}': {missing_str}"
        )

    spice.kclear()
    try:
        for kernel_name in kernel_filenames:
            spice.furnsh(str(kernel_dir / kernel_name))

        mu_earth_val = float(spice.bodvrd("EARTH", "GM", 1)[1][0])
        mu_moon_val = float(spice.bodvrd("MOON", "GM", 1)[1][0])
        return mu_earth_val, mu_moon_val
    finally:
        spice.kclear()


r0 = np.array([0.98736, 0.0, 0.00877])
v0 = np.array([0.0, 1.63446, 0.0])
x0_seed = np.concatenate([r0, v0])

# Earth-Moon mass ratio used in normalized CR3BP
mu_earth, mu_moon = _load_spice_mu_from_local_kernels(kernel_dir)
mu = mu_moon / (mu_earth + mu_moon)
t_f = 1.522
t_opt = 6.0 * t_f  # 6 revolutions
n_nodes = 2
integration_tol = 1e-10

# Build symbolic CR3BP model once and reuse it for both optimization and propagation.
position = ox.State("position", shape=(3,))
velocity = ox.State("velocity", shape=(3,))

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
}
cr3bp_rhs = lower_to_jax(ox.Concat(velocity, velocity_dot))

# Dense high-accuracy propagation used only to build a reliable guess.
guess_dense = np.asarray(
    solve_ivp_diffrax(
        lambda t, x: cr3bp_rhs(x, jnp.zeros((0,), dtype=x.dtype), 0, {}),
        tau_final=t_opt,
        y_0=jnp.asarray(x0_seed, dtype=jnp.float64),
        args=(),
        tau_0=0.0,
        num_substeps=1000,
        solver_name="Dopri8",
        rtol=integration_tol,
        atol=integration_tol,
    ),
    dtype=float,
)

# Keep optimization decision vector at two nodes only: initial and final.
nominal_guess = np.vstack([guess_dense[0], guess_dense[-1]])

position.min = np.array([-2.0, -2.0, -2.0])
position.max = np.array([2.0, 2.0, 2.0])
velocity.min = np.array([-3.0, -3.0, -3.0])
velocity.max = np.array([3.0, 3.0, 3.0])

# Initial/final conditions:
position.initial = [ox.Free(float(x0_seed[0])), 0.0, ox.Free(float(x0_seed[2]))]
velocity.initial = [0.0, ox.Free(float(x0_seed[4])), 0.0]

# Final state is mostly free; terminal objective handles y,vx,vz.
x_tf_guess = nominal_guess[-1]
position.final = [
    ox.Free(float(x_tf_guess[0])),
    0,
    ox.Free(float(x_tf_guess[2])),
]
velocity.final = [
    0,
    ox.Free(float(x_tf_guess[4])),
    0,
]

# Guesses
position.guess = nominal_guess[:, :3]
velocity.guess = nominal_guess[:, 3:]

states = [position, velocity]

time = ox.Time(initial=0.0, final=t_opt, min=0.0, max=t_opt)
discretizer = {
    "ode_solver": "Dopri8",
    "diffrax_kwargs": {"atol": integration_tol, "rtol": integration_tol},
}
algorithm = {
    "k_max": 400,
    "lam_prox": 1e0,
    "lam_vc": 2.5e-1,
    "lam_cost": 5e-1,
    "ep_vc": 1e-6,
    "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0),
}

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=[],
    time=time,
    constraints=[],
    N=n_nodes,
    discretizer=discretizer,
    algorithm=algorithm,
    float_dtype="float64",
)

# Keep post-process propagation tolerances aligned with discretization.
problem.settings.prp.solver = "Dopri8"
problem.settings.prp.atol = integration_tol
problem.settings.prp.rtol = integration_tol


def _solve_halo_orbit() -> np.ndarray:
    """Solve halo initialization problem and return x0_t."""
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    pos = np.asarray(results.trajectory["position"], dtype=float)
    vel = np.asarray(results.trajectory["velocity"], dtype=float)
    x0_opt = np.concatenate([pos[0], vel[0]])
    return x0_opt


def get_halo_initial_condition(
    *,
    force_recompute: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Return halo initial condition x0_t as a simple API."""
    # `force_recompute` is kept for API stability with older callers.
    _ = force_recompute
    x0_t = _solve_halo_orbit()
    if verbose:
        print("Computed halo target initial state.")
    return x0_t


def get_halo_target_initial_state(
    *,
    force_recompute: bool = False,
    use_cache: bool = True,
    cache_file: Path | None = None,
    verbose: bool = False,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, str]]:
    """Backward-compatible wrapper; cache options are ignored."""
    _ = use_cache
    _ = cache_file
    x0_t = get_halo_initial_condition(force_recompute=force_recompute, verbose=verbose)
    if return_metadata:
        return x0_t, {"source": "solve", "cache_file": "none"}
    return x0_t


if __name__ == "__main__":
    x0_t = get_halo_initial_condition(verbose=True)
    np.set_printoptions(precision=8, suppress=True)
    print(x0_t)
