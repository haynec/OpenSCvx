"""6-DoF powered descent guidance base problem for realtime OpenSCvx demos.

Adapted from the SCvxGEN repository by Abhi Kamath, https://scvxgen.mintlify.app/introduction.
"""

import contextlib
import io
import os
import queue
import sys
import time as pytime

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(repo_root_dir)

import openscvx as ox
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
)
from openscvx import Problem
from openscvx.plotting.viser.coordinates import model_vec_to_viser_xyz
from openscvx.utils import printing as _openscvx_printing


def _silence_openscvx_console_printing() -> None:
    _openscvx_printing.intro = lambda: None
    _openscvx_printing.print_problem_summary = lambda *args, **kwargs: None
    _openscvx_printing.print_results_summary = lambda *args, **kwargs: None


def _scp_intermediate_drain_only(problem: Problem):
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
    problem.settings.dev.printing = True
    _openscvx_printing.intermediate = _scp_intermediate_drain_only(problem)
    _patch_get_print_queue_data_for_pdg()


def initialize_problem_quiet(problem: Problem) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        problem.initialize()


def print_scp_table_header_once(problem: Problem) -> None:
    cols = getattr(problem, "_columns", None)
    if cols is not None:
        _openscvx_printing.header(cols)


_silence_openscvx_console_printing()

# Alias for scripts that imported POSITION_STATE_SLICE from this module.
POSITION_STATE_SLICE = slice(1, 4)


def remap_optimization_results_for_viser_xyz(
    results,
    position_slice: slice = POSITION_STATE_SLICE,
) -> None:
    """After ``post_process()``, remap stored trajectories and SCP history for Viser axes."""
    traj = results.trajectory
    if traj.get("position") is not None:
        traj["position"] = model_vec_to_viser_xyz(np.asarray(traj["position"]))
    if traj.get("velocity") is not None:
        traj["velocity"] = model_vec_to_viser_xyz(np.asarray(traj["velocity"]))

    idx_list = list(range(position_slice.start, position_slice.stop, position_slice.step or 1))
    if len(idx_list) != 3:
        idx_list = None

    def _remap_state_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64, copy=True)
        sl = position_slice
        X[:, sl] = model_vec_to_viser_xyz(X[:, sl])
        return X

    for i in range(len(results.X)):
        results.X[i] = _remap_state_rows(results.X[i])

    if getattr(results, "x_full", None) is not None:
        results.x_full = _remap_state_rows(results.x_full)

    if idx_list is None or not results.discretization_history:
        return

    n_x = int(results.X[-1].shape[1]) if results.X else 0
    n_u = int(results.U[-1].shape[1]) if results.U else 0
    if n_x <= 0:
        return

    i4 = n_x + n_x * n_x + 2 * n_x * n_u
    i0, i1, i2 = idx_list[0], idx_list[1], idx_list[2]

    for k in range(len(results.discretization_history)):
        V = np.asarray(results.discretization_history[k], dtype=np.float64, copy=True)
        n_timesteps = V.shape[1]
        n_segments = V.shape[0] // i4
        for seg in range(n_segments):
            b = seg * i4
            for t in range(n_timesteps):
                old = np.array([V[b + i0, t], V[b + i1, t], V[b + i2, t]], dtype=np.float64)
                new = model_vec_to_viser_xyz(old)
                V[b + i0, t] = new[0]
                V[b + i1, t] = new[1]
                V[b + i2, t] = new[2]
        results.discretization_history[k] = V


n = 5

# Runtime-updatable constants (see problem.parameters["name"] after building the problem)
gI = ox.Parameter("gI", value=1.0)
l_arm = ox.Parameter("l", value=0.25)
J_diag = ox.Parameter(
    "J_diag",
    shape=(3,),
    value=np.array([0.168 * 2e-2, 0.168, 0.168]),
)
J_mat = ox.Diag(J_diag)
J_inv_mat = ox.Inv(ox.Diag(J_diag))
g0 = ox.Parameter("g0", value=1.0)
Isp = ox.Parameter("Isp", value=30.0)
m_dry = ox.Parameter("m_dry", value=1.0)
v_max = ox.Parameter("v_max", value=3.0)
w_max = ox.Parameter("w_max", value=0.3752)
del_max = ox.Parameter("del_max", value=20.0)
theta_max = ox.Parameter("theta_max", value=75.0)
T_min = ox.Parameter("T_min", value=1.5)
T_max = ox.Parameter("T_max", value=6.5)
gamma = ox.Parameter("gamma", value=75.0)
beta = ox.Parameter("beta", value=0.01)
c_ax = ox.Parameter("c_ax", value=0.5)
c_ayz = ox.Parameter("c_ayz", value=1.0)
S_a = ox.Parameter("S_a", value=0.5)
rho = ox.Parameter("rho", value=1.0)
l_p = ox.Parameter("l_p", value=0.05)
initial_position = ox.Parameter("initial_position", shape=(3,), value=np.array([7.5, 4.5, 2.5]))
final_position = ox.Parameter("final_position", shape=(2,), value=np.array([0.0, 0.0]))

# Concat (not Stack): JAX lowering stacks scalars as (3,1); Diag needs a length-3 vector (3,).
CA = ox.Diag(ox.Concat(c_ax, c_ayz, c_ayz))
r_arm = ox.Concat(-l_arm, 0.0, 0.0)
r_cp = ox.Concat(l_p, 0.0, 0.0)

mass = ox.State("mass", shape=(1,))
mass.max = [2.0]
mass.min = [1.0]
mass.initial = [2.0]
mass.final = [ox.Maximize(1.5)]

position = ox.State("position", shape=(3,))
position.max = [10.0, 10.0, 10.0]
position.min = [-10.0, -10.0, -10.0]
position.initial = [
    ox.Free(float(initial_position.value[0])),
    ox.Free(float(initial_position.value[1])),
    ox.Free(float(initial_position.value[2])),
]
position.final = [0.0, ox.Free(0.0), ox.Free(0.0)]

velocity = ox.State("velocity", shape=(3,))
velocity.max = [v_max.value, v_max.value, v_max.value]
velocity.min = [-v_max.value, -v_max.value, -v_max.value]
velocity.initial = [-0.5, -2.8, 0.0]
velocity.final = [-0.1, 0.0, 0.0]

attitude = ox.State("attitude", shape=(4,))
attitude.max = [1.0, 1.0, 1.0, 1.0]
attitude.min = [-1.0, -1.0, -1.0, -1.0]
attitude.initial = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0), ox.Free(1.0)]
attitude.final = [0.0, 0.0, 0.0, 1.0]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [w_max.value, w_max.value, w_max.value]
angular_velocity.min = [-w_max.value, -w_max.value, -w_max.value]
angular_velocity.initial = [1e-8, 0.0, 0.0]
angular_velocity.final = [1e-8, 0.0, 0.0]

thrust = ox.Control("thrust", shape=(3,))
thrust.max = [T_max.value, T_max.value, T_max.value]
thrust.min = [-T_max.value, -T_max.value, -T_max.value]
thrust.guess = np.linspace(
    np.array([gI.value * mass.initial[0], 0, 0]),
    np.array([gI.value * m_dry.value, 0, 0]),
    n,
).reshape(-1, 3)

# Extract quaternion components
q1 = attitude[0]
q2 = attitude[1]
q3 = attitude[2]
q4 = attitude[3]

# Direction cosine matrix (DCM) from quaternion
CBI = ox.Block(
    [
        [q4**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q4 * q3), 2 * (q4 * q2 + q1 * q3)],
        [2 * (q4 * q3 + q1 * q2), q4**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q4 * q1)],
        [2 * (q1 * q3 - q4 * q2), 2 * (q4 * q1 + q2 * q3), q4**2 - q1**2 - q2**2 + q3**2],
    ]
).T  # Transpose to get inertial to body frame


def cross(a, b):
    """Cross product of two vectors"""
    return ox.Concat(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )


w1 = angular_velocity[0]
w2 = angular_velocity[1]
w3 = angular_velocity[2]

q1_dot = 0.5 * (w1 * q4 - w2 * q3 + w3 * q2)
q2_dot = 0.5 * (w1 * q3 - w3 * q1 + w2 * q4)
q3_dot = 0.5 * (w2 * q1 - w1 * q2 + w3 * q4)
q4_dot = -0.5 * (w1 * q1 + w2 * q2 + w3 * q3)

attitude_dot = ox.Concat(q1_dot, q2_dot, q3_dot, q4_dot)

# Aerodynamic force in body frame (computed symbolically)
A = -0.5 * rho * ox.linalg.Norm(velocity) * S_a * CA @ CBI @ velocity

dynamics = {
    "mass": -(1 / (Isp * g0)) * ox.linalg.Norm(thrust) - beta,
    "position": velocity,
    "velocity": CBI.T @ (thrust + A) / mass[0] + ox.Concat(-gI, 0.0, 0.0),
    "attitude": attitude_dot,
    "angular_velocity": J_inv_mat
    @ (cross(r_arm, thrust) + cross(r_cp, A) - cross(angular_velocity, (J_mat @ angular_velocity))),
}


states = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust]

constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Boundary Constraints
# Initial position constraint
constraint_exprs.append((position == initial_position).convex().at([0]))

# Terminal position constraint
constraint_exprs.append((position[1:3] == final_position).convex().at([n - 1]))


constraint_exprs.append(ox.ctcs(1.0 * (mass - m_dry) >= 0))
constraint_exprs.append(
    ox.ctcs(0.1 * ox.linalg.Norm(position[1:]) - ox.Tan(gamma * np.pi / 180.0) * position[0] <= 0)
)
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(velocity) ** 2 - v_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(1.0 * ox.Cos(theta_max * np.pi / 180.0) - 1.0 + 2.0 * (q2**2 + q3**2) <= 0)
)
constraint_exprs.append(ox.ctcs(1.0 * ox.linalg.Norm(angular_velocity) ** 2 - w_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(0.1 * ox.linalg.Norm(thrust) - thrust[0] / ox.Cos(del_max * np.pi / 180.0) <= 0)
)
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(thrust) ** 2 - T_max**2 <= 0))
constraint_exprs.append(ox.ctcs(0.1 * T_min**2 - ox.linalg.Norm(thrust) ** 2 <= 0))

# Nominal final time (must match free-final guess). Old API used
# time_dilation_factor_min/max as factors times this value for absolute bounds.
t_final_guess = 10.0
time = ox.Time(
    initial=0.0,
    final=ox.Free(t_final_guess),
    min=0.0,
    max=10.0,
    time_dilation_min=0.2 * t_final_guess,
    time_dilation_max=2.0 * t_final_guess,
)

problem = Problem(
    N=n,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraint_exprs,
    time=time,
    float_dtype="float64",
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": 1e-2,
        "lam_vc": 1e1,
        "lam_prox": 1e0,
        "ep_tr": 5e-3,
        "ep_vc": 1e-6,
    },
)

problem.solver.solver_args = {"abstol": 1e-7, "reltol": 1e-7}

install_pdg_scp_console_style(problem)

# Alias for receding-horizon scripts that follow the 3DoF ``total_time`` naming.
total_time = t_final_guess

plotting_dict: dict = {}


if __name__ == "__main__":
    initialize_problem_quiet(problem)
    result = problem.solve()
    result = problem.post_process()
    result.update(plotting_dict)
    remap_optimization_results_for_viser_xyz(result)

    # Create PDG trajectory visualization
    traj_server = create_animated_plotting_server(
        result,
        thrust_key="thrust",
    )

    # Create SCP iteration visualization
    scp_server = create_scp_animated_plotting_server(
        result,
        frame_duration_ms=50.0,
    )

    # Keep both servers running
    traj_server.sleep_forever()
