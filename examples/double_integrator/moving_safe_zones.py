"""2D double integrator with STL moving safe zones.

The vehicle must stay inside at least one of several moving circular safe zones
while traveling from a start point to a goal. Each zone center follows a
time-varying path (PCHIP spline in physical time), and the specification is

    Always( Or(in_ball_0, ..., in_ball_N) )

over the full horizon, enforced with ``ox.stl`` operators.

The balls are designed as a time-segmented relay: each ball covers a different
time window along a feasible hand-off path, but no single ball spans the full
journey. Only ball 0 covers the start and only ball 5 covers the goal, so the
vehicle must switch zones to reach the finish.
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from examples.double_integrator._plotting import plot_moving_safe_zones
from openscvx import Problem

# Discretization
N = 30
TOTAL_TIME = 10.0

START = np.array([-4.0, 0.0])
GOAL = np.array([4.0, 0.0])

BALL_RADIUS = 0.6

# Knot times aligned with relay hand-offs (non-uniform so ball paths can
# diverge between segments).
T_KNOTS = np.array([0.0, 2.0, 3.5, 5.0, 7.5, 10.0])

# Feasible relay anchors (each lies inside exactly one ball at that time).
RELAY_KEYFRAMES = [
    START,
    np.array([-2.0, 0.9]),
    np.array([-0.8, 0.2]),
    np.array([0.5, -1.0]),
    np.array([2.5, -0.7]),
    GOAL,
]
RELAY_NODES = [0, 6, 10, 15, 22, N - 1]

# Six relay balls — each covers part of the journey; no ball spans start→goal.
BALL_PATHS = [
    # Ball 0: early high lane (only ball at t=0)
    (
        np.array([-4.0, -2.0, -0.5, 1.0, 2.2, 2.8]),
        np.array([0.0, 0.9, 1.1, 1.3, 1.4, 1.5]),
    ),
    # Ball 1: high→mid transition (~2–5 s)
    (
        np.array([-3.5, -2.2, -0.8, 0.2, 1.3, 2.2]),
        np.array([-0.4, 0.55, 0.2, -0.15, -0.35, -0.55]),
    ),
    # Ball 2: alternate mid-high (never at start/goal)
    (
        np.array([-2.8, -1.4, 0.3, 1.6, 2.8, 3.4]),
        np.array([1.25, 1.15, 1.0, 0.9, 0.8, 0.7]),
    ),
    # Ball 3: low trench (~3.5–8 s)
    (
        np.array([-2.5, -1.4, 0.0, 0.5, 2.5, 3.1]),
        np.array([-1.1, -1.0, -0.95, -1.0, -0.7, -0.65]),
    ),
    # Ball 4: late high option (never at start/goal)
    (
        np.array([-1.5, -0.2, 1.2, 2.3, 3.1, 3.5]),
        np.array([1.15, 1.05, 0.95, 0.75, 0.55, 0.5]),
    ),
    # Ball 5: finisher (only ball at t=10)
    (
        np.array([-1.2, 0.3, 1.4, 2.4, 3.4, 4.0]),
        np.array([-0.55, -0.45, -0.35, -0.25, -0.1, 0.0]),
    ),
]

# States
position = ox.State("position", shape=(2,))
position.min = np.array([-6.0, -2.5])
position.max = np.array([6.0, 2.5])
position.initial = START
position.final = GOAL

velocity = ox.State("velocity", shape=(2,))
velocity.min = np.array([-8.0, -8.0])
velocity.max = np.array([8.0, 8.0])
velocity.initial = np.array([0.0, 0.0])
velocity.final = [ox.Free(0.0), ox.Free(0.0)]

# Fixed final time so ball centers stay synchronized with physical time.
time = ox.Time(
    initial=0.0,
    final=TOTAL_TIME,
    min=0.0,
    max=TOTAL_TIME,
    uniform_time_grid=True,
)

force = ox.Control("force", shape=(2,))
A_MAX = 12.0
force.max = np.array([A_MAX, A_MAX])
force.min = np.array([-A_MAX, -A_MAX])
force.guess = np.zeros((N, 2))

states = [position, velocity, time]
controls = [force]
m = 1.0

ball_radius = ox.Parameter("ball_radius", shape=(), value=BALL_RADIUS)

in_ball_predicates = []
for ball_x, ball_y in BALL_PATHS:
    center = ox.Concat(
        ox.Cinterp(time[0], T_KNOTS, ball_x, method="pchip"),
        ox.Cinterp(time[0], T_KNOTS, ball_y, method="pchip"),
    )
    delta = position - center
    in_ball_predicates.append(ox.Sum(delta * delta) <= ball_radius * ball_radius)

in_some_ball = ox.stl.Or(*in_ball_predicates)

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

constraints.append(ox.stl.Always(in_some_ball, (0, N - 1)).over())

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * force,
    "time": 1.0,
}

position.guess = ox.init.linspace(keyframes=RELAY_KEYFRAMES, nodes=RELAY_NODES)
velocity.guess = np.gradient(position.guess, TOTAL_TIME / (N - 1), axis=0)
force.guess = m * np.gradient(velocity.guess, TOTAL_TIME / (N - 1), axis=0)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_prox": 5e-3,
        "lam_vc": 2e1,
        "lam_cost": 1e-3,
    },
    float_dtype="float64",
)

plotting_data = {
    "ball_radius": BALL_RADIUS,
    "t_knots": T_KNOTS,
    "ball_paths": BALL_PATHS,
    "ball_interp_method": "pchip",
    "start": START,
    "goal": GOAL,
}

if __name__ == "__main__":
    n_balls = len(BALL_PATHS)
    print("2D Double Integrator — Moving Safe Zones (STL Or + Always)")
    print("=" * 60)
    print(f"Start: {START}, Goal: {GOAL}")
    print(f"Balls: {n_balls}, radius: {BALL_RADIUS}")
    print("Design: time-segmented relay (mandatory ball hand-offs)")
    print(f"Spec: Always( Or(in_ball_0, ..., in_ball_{n_balls - 1}) ) over the horizon")
    print("=" * 60)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_data)

    plot_moving_safe_zones(results).show()