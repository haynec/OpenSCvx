"""Chen–Allgöwer unstable OCP (OpenSCvx port).

Port of the acados Chen–Allgöwer example
(``export_chen_allgoewer_model`` + feasibility / unconstrained OCP setup).

The system is a 2-state unstable plant with scalar control ``u`` and parameter
``mu = 0.7``:

    x1_dot = x2 + u * (mu + (1 - mu) * x2)
    x2_dot = x1 + u * (mu - 4 * (1 - mu) * x2)

Two modes mirror acados ``SOLVE_FEASIBILITY_PROBLEM``:

* **Feasibility** (default): drive from a parametric initial state to the
  terminal point ``[0, 0.03]`` over ``T_f = 5`` s with ``|u| <= 1.5``. No
  running cost (``lam_cost = 0``).
* **Unconstrained OCP**: quadratic tracking cost on ``x`` and ``u`` with soft
  control limits at ``|u| <= 1`` (barrier-style penalty in the integrand).

Horizon: ``N = 21`` nodes (acados ``N_horizon = 20`` intervals).

If ``chen_allgoewer_initial_guess.npy`` sits next to this script (acados format:
first array ``X_init`` shape ``(2, N+1)``, second ``U_init`` shape ``(1, N)``),
it is loaded as the warm start; otherwise a linear state interpolation from
``x0`` to ``x_f`` with zero control is used.
"""

from __future__ import annotations

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states

# ── Problem parameters (match acados defaults) ───────────────────────────────
N = 21  # nodes (acados N_horizon = 20 intervals)
TF = 5.0
MU = 0.7
ONE_MINUS_MU = 1.0 - MU

# Feasibility mode
U_MAX_FEAS = 1.5
X_TERMINAL = np.array([0.0, 0.03])

# Unconstrained OCP mode
Q_X = np.diag([0.5, 0.5])
R_U = np.array([[0.8]])
P_X = np.diag([10.0, 10.0])
U_MAX_SOFT = 1.0
TAU_SOFT = 100.0

DEFAULT_INITIAL_CONDITIONS = (
    np.array([0.42, 0.45]),
    np.array([0.42, 0.5]),
)

INITIAL_GUESS_PATH = os.path.join(current_dir, "chen_allgoewer_initial_guess.npy")


def _load_initial_guess(n: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.isfile(INITIAL_GUESS_PATH):
        return None
    with open(INITIAL_GUESS_PATH, "rb") as f:
        x_init = np.load(f)
        u_init = np.load(f)
    if x_init.shape != (2, n):
        raise ValueError(f"Expected X_init shape (2, {n}); got {x_init.shape}.")
    if u_init.shape == (1, n - 1):
        # acados stores one control per shooting interval; pad for OpenSCvx nodes.
        u_init = np.pad(u_init, ((0, 0), (0, 1)), mode="edge")
    elif u_init.shape != (1, n):
        raise ValueError(f"Expected U_init shape (1, {n}) or (1, {n - 1}); got {u_init.shape}.")
    return x_init, u_init


def _default_initial_guess(x0: np.ndarray, xf: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    tau = np.linspace(0.0, 1.0, n)
    x_init = (1.0 - tau) * x0[:, None] + tau * xf[:, None]
    u_init = np.zeros((1, n))
    return x_init, u_init


def _apply_initial_guess(
    x1: ox.State,
    x2: ox.State,
    control: ox.Control,
    x0: np.ndarray,
    xf: np.ndarray,
    n: int,
) -> None:
    loaded = _load_initial_guess(n)
    if loaded is None:
        x_init, u_init = _default_initial_guess(x0, xf, n)
    else:
        x_init, u_init = loaded

    x1.guess = x_init[0, :].reshape(-1, 1)
    x2.guess = x_init[1, :].reshape(-1, 1)
    control.guess = u_init[0, :].reshape(-1, 1)


def build_problem(
    *, solve_feasibility: bool = True
) -> tuple[Problem, ox.State, ox.State, ox.Control]:
    u_max = U_MAX_FEAS if solve_feasibility else U_MAX_SOFT

    x1 = ox.State("x1", shape=(1,))
    x1.min = np.array([-5.0])
    x1.max = np.array([5.0])
    x1.initial = np.array([DEFAULT_INITIAL_CONDITIONS[0][0]])
    x1.final = [X_TERMINAL[0]] if solve_feasibility else [ox.Free(0.0)]

    x2 = ox.State("x2", shape=(1,))
    x2.min = np.array([-5.0])
    x2.max = np.array([5.0])
    x2.initial = np.array([DEFAULT_INITIAL_CONDITIONS[0][1]])
    x2.final = [X_TERMINAL[1]] if solve_feasibility else [ox.Free(0.0)]

    control = ox.Control("u", shape=(1,), parameterization="ZOH")
    control.min = np.array([-u_max])
    control.max = np.array([u_max])

    states: list[ox.State] = [x1, x2]
    controls = [control]

    dynamics: dict = {
        "x1": x2[0] + control[0] * (ox.Constant(MU) + ox.Constant(ONE_MINUS_MU) * x2[0]),
        "x2": x1[0] + control[0] * (ox.Constant(MU) - ox.Constant(4.0 * ONE_MINUS_MU) * x2[0]),
    }

    if not solve_feasibility:
        soft_upper = ox.Constant(TAU_SOFT) * ox.Max(ox.Constant(0.0), control[0] - U_MAX_SOFT) ** 2
        soft_lower = ox.Constant(TAU_SOFT) * ox.Min(ox.Constant(0.0), control[0] + U_MAX_SOFT) ** 2
        stage_cost = ox.State("stage_cost", shape=(1,))
        stage_cost.min = np.array([0.0])
        stage_cost.max = np.array([1e4])
        stage_cost.initial = np.array([0.0])
        stage_cost.final = [ox.Minimize(0.0)]
        dynamics["stage_cost"] = (
            ox.Constant(0.5) * (Q_X[0, 0] * x1[0] ** 2 + Q_X[1, 1] * x2[0] ** 2)
            + ox.Constant(0.5) * R_U[0, 0] * control[0] ** 2
            + soft_upper
            + soft_lower
        )
        states.append(stage_cost)

    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

    if solve_feasibility:
        _apply_initial_guess(
            x1,
            x2,
            control,
            DEFAULT_INITIAL_CONDITIONS[0],
            X_TERMINAL,
            N,
        )
    else:
        _apply_initial_guess(
            x1,
            x2,
            control,
            DEFAULT_INITIAL_CONDITIONS[0],
            np.zeros(2),
            N,
        )

    time = ox.Time(
        initial=0.0,
        final=TF,
        min=0.0,
        max=TF,
        uniform_time_grid=True,
    )

    algorithm: dict = {
        "lam_prox": 1e0,
        "lam_vc": 1e1,
        "autotuner": ox.RampProximalWeight(ramp_factor=1.04, lam_prox_max=1e3),
    }
    if solve_feasibility:
        algorithm["lam_cost"] = 0e0
    else:
        algorithm["lam_cost"] = 0e0

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=N,
        # float_dtype="float64",
        algorithm=algorithm,
        discretizer=ox.LinearizeDiscretizeSparse(dis_type="ZOH"),
        # solver = {
        #     "cvx_solver": "PIQP",
        #     "solver_args": {"canon_backend": "COO", "enforce_dpp": True},
        # }
        solver={
            "cvx_solver": "qocogen",
            "solver_args": {},
            "cvxpygen": True,
        },
    )
    problem.settings.dev.printing = False
    return problem, x1, x2, control


def _show_plot(fig):
    try:
        fig.show()
    except PermissionError as exc:
        print(f"Skipping plot display: {exc}")


def solve_for_initial_condition(
    problem: Problem,
    x1: ox.State,
    x2: ox.State,
    control: ox.Control,
    x0: np.ndarray,
    *,
    plot_solution: bool = False,
) -> ox.OptimizationResults:
    x1.initial = np.array([x0[0]])
    x2.initial = np.array([x0[1]])
    _apply_initial_guess(x1, x2, control, x0, X_TERMINAL, N)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print(f"x0 = {x0}")
    print(f"  x1(T) = {nodes['x1'][-1, 0]:.6f}  (target {X_TERMINAL[0]:.6f})")
    print(f"  x2(T) = {nodes['x2'][-1, 0]:.6f}  (target {X_TERMINAL[1]:.6f})")
    print(f"  converged: {results.converged}")

    if plot_solution:
        _show_plot(plot_states(results))
        _show_plot(plot_controls(results))

    return results


if __name__ == "__main__":
    SOLVE_FEASIBILITY_PROBLEM = True
    PLOT_SOLUTION = os.environ.get("OPENSCVX_NO_PLOT") is None

    problem, x1, x2, control = build_problem(solve_feasibility=SOLVE_FEASIBILITY_PROBLEM)

    for x0 in DEFAULT_INITIAL_CONDITIONS:
        solve_for_initial_condition(problem, x1, x2, control, x0, plot_solution=PLOT_SOLUTION)
