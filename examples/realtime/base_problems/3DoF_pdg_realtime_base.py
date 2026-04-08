"""3-DOF Powered Descent Guidance (PDG) for planetary landing.

This example demonstrates optimal trajectory generation for a rocket performing
powered descent guidance, similar to SpaceX Falcon 9 or Blue Origin landings.
The problem includes:

- 3D position and velocity dynamics
- Fuel-optimal mass minimization
- Thrust magnitude and pointing constraints
- Glideslope constraint for safe landing approach
"""

import contextlib
import io
import os
import queue
import sys
import time as pytime

import numpy as np

# Add repo root to path so `examples.*` imports resolve when run as a script.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(repo_root_dir)

import openscvx as ox
from openscvx import Free

from examples.plotting_viser import (
    create_pdg_animated_plotting_server,
    create_scp_animated_plotting_server,
)

from openscvx import Problem
from openscvx.plotting import plot_controls, plot_projections_2d, plot_states, plot_vector_norm
from openscvx.utils import printing as _openscvx_printing


def _silence_openscvx_console_printing() -> None:
    """Disable ASCII banner, problem box, and post-process results box for this example."""
    _openscvx_printing.intro = lambda: None
    _openscvx_printing.print_problem_summary = lambda *args, **kwargs: None
    _openscvx_printing.print_results_summary = lambda *args, **kwargs: None


def _scp_intermediate_drain_only(problem: Problem):
    """Consume iteration emits without printing rows (keeps queue from growing)."""

    def _run(print_queue, params, columns) -> None:
        hz = 30.0
        while True:
            t_start = pytime.time()
            try:
                data = print_queue.get(timeout=1.0 / hz)
                problem._scp_last_emit = data
            except queue.Empty:
                pass
            pytime.sleep(max(0.0, 1.0 / hz - (pytime.time() - t_start)))

    return _run


_gpq_patched = False


def _patch_get_print_queue_data_for_pdg() -> None:
    """Prefer latest SCP emit stored on the problem (drain thread) for Viser metrics."""
    global _gpq_patched
    if _gpq_patched:
        return
    import examples.plotting_viser as _pv

    _orig_gpq = _pv.get_print_queue_data

    def _gpq(problem):
        emit = getattr(problem, "_scp_last_emit", None)
        if emit is not None:
            return {
                "dis_time": emit.get("dis_time", 0.0),
                "prob_stat": emit.get("prob_stat", "--"),
                "cost": emit.get("cost", 0.0),
            }
        return _orig_gpq(problem)

    _pv.get_print_queue_data = _gpq
    _gpq_patched = True


def install_pdg_scp_console_style(problem: Problem) -> None:
    """Show SCP column header once; suppress per-iteration table rows."""
    problem.settings.dev.printing = True
    _openscvx_printing.intermediate = _scp_intermediate_drain_only(problem)
    _patch_get_print_queue_data_for_pdg()


def initialize_problem_quiet(problem: Problem) -> None:
    """Call ``initialize()`` without writing OpenSCvx banner/timing lines to stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        problem.initialize()


def print_scp_table_header_once(problem: Problem) -> None:
    """Print the SCP iteration table header (column names) once."""
    cols = getattr(problem, "_columns", None)
    if cols is not None:
        _openscvx_printing.header(cols)


_silence_openscvx_console_printing()

n = 10
total_time = 95.0  # Total simulation time

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
v_max = 500 * 1e3 / 3600  # Maximum velocity in m/s (800 km/h converted to m/s)
position.max = np.array([3000, 3000, 3000])
position.min = np.array([-3000, -3000, 0])
position.initial = [Free(2000), Free(0), Free(1500)]
position.final = [Free(0), Free(0), 0]
position.guess = np.linspace(position.initial, position.final, n)

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([v_max, v_max, v_max])
velocity.min = np.array([-v_max, -v_max, -v_max])
velocity.initial = np.array([80, 30, -75])
velocity.final = np.array([0, 0, 0])
velocity.guess = np.linspace(velocity.initial, velocity.final, n)

mass = ox.State("mass", shape=(1,))  # Vehicle mass
mass.max = np.array([1905])
mass.min = np.array([1505])
mass.initial = np.array([1905])
mass.final = [("maximize", 1700)]
mass.scaling_min = np.array([1700])
# mass.scaling_max = np.array([1700])
mass.guess = np.linspace(mass.initial, 1690, n).reshape(-1, 1)

# Define control — thrust force vector [Tx, Ty, Tz]
thrust = ox.Control("thrust", shape=(3,), parameterization="ZOH")

T_bar = 3.1 * 1e3
# T1 = 0.3 * T_bar
T2_init = 0.8 * T_bar
n_eng = 6

# Set bounds on control
thrust.min = n_eng * np.array([-T_bar, -T_bar, -T_bar])
thrust.max = n_eng * np.array([T_bar, T_bar, T_bar])

# Set initial control guess
thrust.guess = np.repeat(np.expand_dims(np.array([0, 0, n_eng * (T2_init) / 2]), axis=0), n, axis=0)

# Define list of all states and controls
states = [position, velocity, mass]
controls = [thrust]


# Define Parameters for physical constants
g_e = 9.807  # Gravitational acceleration on Earth in m/s^2

# Create parameters for the problem
I_sp = ox.Parameter("I_sp", value=225.0)
g = ox.Parameter("g", value=3.7114)
theta = ox.Parameter("theta", value=27 * np.pi / 180)
glideslope_angle = ox.Parameter("glideslope_angle", value=86 * np.pi / 180)
thrust_pointing_angle = ox.Parameter("thrust_pointing_angle", value=40 * np.pi / 180)
T1 = ox.Parameter("T1", value=0.3 * T_bar)
T2 = ox.Parameter("T2", value=0.8 * T_bar)
initial_position = ox.Parameter("initial_position", shape=(3,), value=np.array([2000, 0, 1500]))
final_position = ox.Parameter("final_position", shape=(2,), value=np.array([0, 0]))

# Keep these symbolic so live parameter updates are reflected in constraints.
rho_min_expr = n_eng * T1 * ox.Cos(theta)  # Minimum thrust bound
rho_max_expr = n_eng * T2 * ox.Cos(theta)  # Maximum thrust bound

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, idx=0),
            ox.ctcs(state.min <= state, idx=0),
        ]
    )

# Thrust magnitude constraints
constraints.extend(
    [
        ox.ctcs(rho_min_expr <= ox.linalg.Norm(thrust), idx=1),
        ox.ctcs(ox.linalg.Norm(thrust) <= rho_max_expr, idx=1),
    ]
)

# Thrust pointing constraint (thrust cant angle)
constraints.append(
    ox.ctcs(ox.Cos(thrust_pointing_angle) <= thrust[2] / ox.linalg.Norm(thrust), idx=2)
)

# Glideslope constraint
constraints.append(
    ox.ctcs(ox.linalg.Norm(position[:2]) <= ox.Tan(glideslope_angle) * position[2], idx=3)
)

# Initial position constraint
constraints.append(
    (position == initial_position).convex().at([0])
)

# Terminal position constraint
constraints.append(
    (position[0:2] == final_position).convex().at([n-1])
)


# Define dynamics as dictionary mapping state names to their derivatives
g_vec = np.array([0, 0, 1], dtype=np.float64) * g  # Gravitational acceleration vector

dynamics = {
    "position": velocity,
    "velocity": thrust / mass[0] - g_vec,
    "mass": -ox.linalg.Norm(thrust) / (I_sp * g_e * ox.Cos(theta)),
}

# Build the problem
time = ox.Time(
    initial=0.0,
    final=("free", total_time),
    min=0.0,
    max=2e2,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": 1 / 3,
        "lam_vc": 2e0,
        "lam_prox": 1e0,
    },
    discretizer={"ode_solver": "Dopri8"},
)

install_pdg_scp_console_style(problem)

plotting_dict = {
    "rho_min": n_eng * T1.value * np.cos(theta.value),
    "rho_max": n_eng * T2.value * np.cos(theta.value),
    "glideslope_angle_deg": 86.0,
}

if __name__ == "__main__":
    initialize_problem_quiet(problem)
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Create PDG trajectory visualization
    scene_scale = 100
    traj_server = create_pdg_animated_plotting_server(
        results,
        thrust_key="thrust",
        glideslope_angle_deg=86.0,
        scene_scale=1.0,
    )

    # Create SCP iteration visualization
    scp_server = create_scp_animated_plotting_server(
        results,
        frame_duration_ms=200,
        scene_scale=100.0,
    )

    # Keep servers running
    traj_server.sleep_forever()
