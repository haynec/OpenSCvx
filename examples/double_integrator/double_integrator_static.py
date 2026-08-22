"""1D double integrator with box bounds and terminal goal (OpenSCvx port).

Port of the RobotZoo / TrajectoryOptimization.jl ``DoubleIntegrator`` setup::

    x0 = [0, 0], xf = [1, 0], tf = 2 s, N = 21
    LQR stage cost  (x - xf)' Q (x - xf) + u' R u  with Q = I, R = 0.1 I
    |u| <= 3, |v| <= 0.6, position unconstrained
    terminal goal at the final node

Julia ``SolverOptions``: ``penalty_scaling=1000``, ``penalty_initial=1``.
"""

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

# ── Problem parameters (match Julia DoubleIntegrator) ─────────────────────────
N = 21
TF = 2.0
DT = TF / (N - 1)

X0 = np.array([0.0, 0.0])
XF = np.array([1.0, 0.0])

Q_POS = 1.0
Q_VEL = 1.0
R_U = 0.1

U_BND = 3.0
V_BND = 0.6

# ── States ───────────────────────────────────────────────────────────────────
position = ox.State("position", shape=(1,))
# Julia uses Inf (unbounded); OpenSCvx needs finite box limits for CTCS.
position.min = np.array([-100.0])
position.max = np.array([100.0])
position.initial = np.array([X0[0]])
position.final = np.array([XF[0]])

velocity = ox.State("velocity", shape=(1,))
velocity.min = np.array([-V_BND])
velocity.max = np.array([V_BND])
velocity.initial = np.array([X0[1]])
velocity.final = np.array([XF[1]])

stage_cost = ox.State("stage_cost", shape=(1,))
stage_cost.min = np.array([0.0])
stage_cost.max = np.array([1e3])
stage_cost.initial = np.array([0.0])
stage_cost.final = [ox.Minimize(0.0)]

# ── Control ──────────────────────────────────────────────────────────────────
force = ox.Control("force", shape=(1,), parameterization="ZOH")
force.min = np.array([-U_BND])
force.max = np.array([U_BND])
force.guess = np.zeros((N, 1))

states = [position, velocity, stage_cost]
controls = [force]

# ── Dynamics ─────────────────────────────────────────────────────────────────
dynamics = {
    "position": velocity[0],
    "velocity": force[0],
    "stage_cost": (
        Q_POS * (position[0] - ox.Constant(XF[0])) ** 2
        + Q_VEL * (velocity[0] - ox.Constant(XF[1])) ** 2
        + R_U * force[0] ** 2
    ),
}

# ── Constraints ──────────────────────────────────────────────────────────────
constraints: list = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
constraints.extend([ox.ctcs(force <= force.max), ox.ctcs(force.min <= force)])

# ── Initial guess (straight-line rollout analogue) ───────────────────────────
tau = np.linspace(0.0, 1.0, N)
position.guess = ((1.0 - tau) * X0[0] + tau * XF[0]).reshape(-1, 1)
velocity.guess = np.zeros((N, 1))
t_guess = np.linspace(0.0, TF, N)
stage_cost.guess = np.cumsum(
    (
        Q_POS * (position.guess[:, 0] - XF[0]) ** 2
        + Q_VEL * velocity.guess[:, 0] ** 2
        + R_U * force.guess[:, 0] ** 2
    )
    * np.gradient(t_guess)
).reshape(-1, 1)

time = ox.Time(
    initial=0.0,
    final=TF,
    min=0.0,
    max=TF,
    uniform_time_grid=True,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N,
    float_dtype="float64",
    algorithm={
        # Julia SolverOptions: penalty_scaling=1000, penalty_initial=1
        "lam_prox": 1.0,
        "lam_vc": 1e3,
        "lam_cost": 1.0,
    },
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print(f"dt = {DT:.4f} s")
    print(f"Final position: {nodes['position'][-1, 0]:.6f} (target {XF[0]})")
    print(f"Final velocity: {nodes['velocity'][-1, 0]:.6f} (target {XF[1]})")
    print(f"Integrated stage cost: {nodes['stage_cost'][-1, 0]:.6f}")

    plot_states(results).show()
    plot_controls(results).show()
