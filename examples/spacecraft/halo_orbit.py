"""Southern L2 Halo orbit initial conditions definition

- Earth-Moon CR3BP rotating-frame dynamics
- Symmetry structure at initial node: y0 = vx0 = vz0 = 0
- Free design variables at initial node: x0, z0, vy0
- Terminal objective to drive (y, vx, vz) toward zero after n revolutions

"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

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

r0 = np.array([0.98736, 0.0, 0.00877])
v0 = np.array([0.0, 1.63446, 0.0])
x0_seed = np.concatenate([r0, v0])

# Earth-Moon mass ratio used in normalized CR3BP
mu_earth = 398600.4354360959
mu_moon = 4902.800066163796
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

# Dense high-accuracy propagation used only to build a reliable guess and for plotting.
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

# Broad bounds (required by OpenSCvx for all variables)
position.min = np.array([-2.0, -2.0, -2.0])
position.max = np.array([2.0, 2.0, 2.0])
velocity.min = np.array([-3.0, -3.0, -3.0])
velocity.max = np.array([3.0, 3.0, 3.0])

# Initial/final conditions:
# y0 = vx0 = vz0 = 0 (fixed), x0/z0/vy0 free around the seed values.
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
    "atol": integration_tol,
    "rtol": integration_tol,
}
algorithm = {
    "k_max": 400,
    "lam_prox": 2e-1,
    "lam_vc": 5e-2,
    "ep_vc": 1e-6,
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

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    x0_opt = np.concatenate([pos[0], vel[0]])
    xf_opt = np.concatenate([pos[-1], vel[-1]])

    # Match the post-optimization check style in main_orbs_2.py:
    # integrate only to t_f from optimized initial state and export xyz.
    traj_plot = np.asarray(
        solve_ivp_diffrax(
            lambda t, x: cr3bp_rhs(x, jnp.zeros((0,), dtype=x.dtype), 0, {}),
            tau_final=t_opt,
            y_0=jnp.asarray(x0_opt, dtype=jnp.float64),
            args=(),
            tau_0=0.0,
            num_substeps=2000,
            solver_name="Dopri8",
            rtol=integration_tol,
            atol=integration_tol,
        ),
        dtype=float,
    )

    # Plot guess and solution using the exact plot_projections_2d style.
    guess_results = OptimizationResults(converged=True, t_final=float(t_opt))
    guess_results.trajectory = {
        "time": np.linspace(0.0, t_opt, guess_dense.shape[0]).reshape(-1, 1),
        "position": guess_dense[:, :3],
        "velocity": guess_dense[:, 3:6],
    }
    guess_results.nodes = {
        "time": np.array([0.0, t_opt]).reshape(-1, 1),
        "position": nominal_guess[:, :3],
        "velocity": nominal_guess[:, 3:6],
    }
    fig_guess = plot_projections_2d(guess_results, velocity_var_name="velocity")
    fig_guess.update_layout(title="Initial Guess - XY, XZ, YZ Projections")
    fig_guess.show()

    solution_results = OptimizationResults(converged=bool(results.converged), t_final=float(t_opt))
    solution_results.trajectory = {
        "time": np.linspace(0.0, t_opt, traj_plot.shape[0]).reshape(-1, 1),
        "position": traj_plot[:, :3],
        "velocity": traj_plot[:, 3:6],
    }
    solution_results.nodes = {
        "time": results.nodes["time"],
        "position": results.nodes["position"],
        "velocity": results.nodes["velocity"],
    }
    fig_solution = plot_projections_2d(solution_results, velocity_var_name="velocity")
    fig_solution.update_layout(title="Solution - XY, XZ, YZ Projections")
    fig_solution.show()
