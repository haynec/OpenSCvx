import os
import sys

import jax

# use float64
jax.config.update("jax_enable_x64", True)

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx.plotting import plot_controls, plot_states

n = 40
time_final = 1000.0
time_dilation_min = 1e-4
time_dilation_max = 1e4

x = ox.State("x", shape=(1,))
x.min = [0]
x.max = [50]
x.scaling_max = [1.5]
x.scaling_min = [0.0]
x.initial = [1.5]
x.final = [1]
x.guess = np.linspace(0.0, 0.0, n).reshape(-1, 1)

cost = ox.State("cost", shape=(1,))
cost.min = [0]
cost.max = [100]
cost.scaling_max = [1e0]
cost.initial = [0]
cost.final = [ox.Minimize(0.0)]
cost.guess = np.linspace(0, 0, n).reshape(-1, 1)

u = ox.Control("u", shape=(1,))
u.min = [-50]
u.max = [50]
u.scaling_max = [1.0]
u.scaling_min = [-0.5]
u.guess = np.linspace(0.0, 0.0, n).reshape(-1, 1)

# Cosine time-grid guess: denser nodes near the endpoints.
s_uniform = np.linspace(0.0, 1.0, n)
node_grid = 0.5 * (1.0 - np.cos(np.pi * s_uniform))
time_guess = (time_final * node_grid).reshape(-1, 1)

dtdtau_guess = np.gradient(time_guess[:, 0], s_uniform)
dtdtau_guess = np.clip(dtdtau_guess, time_dilation_min, time_dilation_max)

time = ox.Time(
    initial=0.0,
    final=time_final,
    min=0.0,
    max=time_final,
    guess=time_guess,
    time_dilation_min=time_dilation_min,
    time_dilation_max=time_dilation_max,
    time_dilation_guess=dtdtau_guess.reshape(-1, 1),
    uniform_time_grid=False,
)


dynamics = {
    "x": -x + u,
    "cost": 0.5 * (x**2 + u**2),
}

states = [x, cost]
controls = [u]
constraints = []
for state in states:
    constraints.extend(
        [ox.ctcs(state <= state.max, penalty="huber"), ox.ctcs(state.min <= state, penalty="huber")]
    )

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    constraints=constraints,
    N=n,
    time=time,
    licq_max=1e-8,
    algorithm={
        "lam_prox": 1e0,
        "lam_cost": {"cost": 3e0},
        "lam_vc": 4e0,
        "autotuner": ox.AugmentedLagrangian(ep=1e-1),
    },
    discretizer={
        "dis_type": "ZOH",
        "ode_solver": "Dopri8",
        "diffrax_kwargs": {"atol": 1e-12, "rtol": 1e-12},
    },
    float_dtype="float64",
)

problem.settings.prp.atol = 1e-12
problem.settings.prp.rtol = 1e-12


def _show_plot(fig):
    try:
        fig.show()
    except PermissionError as exc:
        print(f"Skipping plot display: {exc}")


if __name__ == "__main__":
    problem.initialize()
    problem.solve()
    results = problem.post_process()

    _show_plot(plot_states(results))
    _show_plot(plot_controls(results))
    # plot_scp_iterations(results).show()
