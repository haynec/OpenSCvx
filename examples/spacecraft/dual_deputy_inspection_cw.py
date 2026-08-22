"""Dual-deputy relative-orbit inspection with CW dynamics.

Two deputies inspect a non-maneuvering chief (CW origin) under:

- Circular Earth-orbit Clohessy-Wiltshire translational dynamics
- Impulsive body-frame delta-v at every node (RCS burns), rotated into CW
  via QDCM(q) so thruster directions follow attitude
- Continuous reaction-wheel torque for full 6DoF attitude / FOV pointing
- Thin annular keep-in band about the chief (min / max range)
- Collinearity of the two deputies with the chief (symbolic cross product)
- Opposite-side constraint so the formation stays dual-sided
- Continuous sensor FOV cones keeping the chief in each camera frame
- Minimum total delta-v objective (accumulated impulse cost state)
- Uniform LoS-sweep heuristic: dual "clocks" drive the inspection axis on S²
  at constant azimuth φ and colatitude θ rates (equal dwell in each direction),
  including out-of-plane views — no discrete waypoints

Mission: sweep the dual viewing axis uniformly over the chief's viewing sphere
on a fixed horizon while staying in the annulus, keeping LoS / FOV, and limiting
total Δv. Antipodal pair (r2 = -r1) means one axis direction n = r1/‖r1‖ covers
both sides; sweeping θ ∈ [π/2, 0] and φ ∈ [0, π] traces a hemisphere (π sr).

!!! tip "The CW frame convention:"
    - x: radial direction (outward from Earth)
    - y: along-track direction (velocity direction)
    - z: cross-track direction (normal to orbit plane)
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import create_animated_plotting_server
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states
from openscvx.plotting.viser import (
    add_animated_trail,
    add_animation_controls,
    add_attitude_frame,
    add_ghost_trajectory,
    add_position_marker,
    add_viewcone,
    compute_velocity_colors,
)
from openscvx.solvers.cvxpy_ptr_solver import CVXPyPTRSolver

# Dual-vehicle CTCS (shared aug state) fills discrete A_d across both blocks, but
# symbolic sparsity stays block-diagonal. Drop sparse Parameter patterns before
# Problem() lowers into CVXPy (create_variables runs at construction time).
# Keep N modest — dense params OOM around N≳20 on typical laptops.
_orig_create_variables = CVXPyPTRSolver.create_variables


def _create_variables_dense(
    self, *args, dynamics_sparsity=None, constraint_sparsity=None, **kwargs
):
    return _orig_create_variables(
        self,
        *args,
        dynamics_sparsity=None,
        constraint_sparsity=None,
        **kwargs,
    )


CVXPyPTRSolver.create_variables = _create_variables_dense

# =============================================================================
# Problem parameters
# =============================================================================

N = 30
total_time = 3000.0  # Fixed horizon [s] (~0.44 of an ISS orbital period)

# Circular LEO (ISS-like) so CW is valid — matches proxops_cw.py
mu = 3.986004418e14  # Earth gravitational parameter [m^3/s^2]
a_orbit = 6.778e6  # Semi-major axis [m]
n_mean = np.sqrt(mu / a_orbit**3)  # Mean motion [rad/s]

R_min = 48.0  # Annular keep-in inner radius [m]
R_max = 52.0  # Annular keep-in outer radius [m]
R_nom = 0.5 * (R_min + R_max)

eps_colinear = 75.0  # ‖r1 × r2‖ tolerance [m^2]; slightly loose for a long tour
dv_max = 0.5  # Per-axis body-frame impulsive delta-v bound [m/s]
cost_ub = 30.0  # Upper bound / minimize guess for total Δv [m/s]
v_max = 2.0  # Relative velocity box [m/s]
w_max = 0.5  # Angular rate box [rad/s]
torque_max = 1.0  # Reaction-wheel torque box [N·m]
J_b = jnp.array([1.0, 1.0, 1.0])  # Deputy inertia diagonal [kg·m^2]

# Spherical uniform-sweep reference rates for deputy-1 axis n = r1/‖r1‖.
#   θ = colatitude from +x (radial),  φ = azimuth in the yz-plane.
# Antipodal pair: θ sweeps π/2 → 0 and φ sweeps 0 → π covers one hemisphere.
theta_init = 0.5 * np.pi
theta_final = 0.0
phi_sweep = np.pi
omega_phi_ref = phi_sweep / total_time
omega_theta_ref = (theta_final - theta_init) / total_time  # negative: toward +x pole

impulse_nodes = list(range(N))

# Sensor / FOV (dr_vp-style cone; half-angle = pi / alpha)
# Wider than the drone racing cone — inspection at ~50 m needs a looser FOV
# for the dual-6DoF NLP to remain tractable.
alpha_x = 16.0
alpha_y = 16.0
A_cone = np.diag(
    [
        1.0 / np.tan(np.pi / alpha_x),
        1.0 / np.tan(np.pi / alpha_y),
        0.0,
    ]
)
c_boresight = jnp.array([0.0, 0.0, 1.0])
norm_type = 2
R_sb = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# =============================================================================
# Helpers
# =============================================================================


def cross(a, b):
    """Right-handed cross product (matches examples/rocket/6DoF_pdg.py:196–200)."""
    return ox.Concat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _pointing_attitudes(positions: np.ndarray) -> np.ndarray:
    """Quaternions [qw, qx, qy, qz] that point the sensor boresight at the origin."""
    b = np.asarray(R_sb @ np.array([0.0, 1.0, 0.0]), dtype=float)
    attitudes = np.zeros((positions.shape[0], 4))
    for k, pos in enumerate(positions):
        a = -np.asarray(pos, dtype=float)  # chief (origin) relative to deputy
        if la.norm(a) < 1e-9:
            attitudes[k] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        q_xyz = np.cross(b, a)
        q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
        q = np.hstack((q_w, q_xyz))
        attitudes[k] = q / la.norm(q)
    return attitudes


def _great_circle_yz(n_nodes: int, radius: float, sweep_rad: float = 2.0 * np.pi) -> np.ndarray:
    """Great-circle path on the yz-plane starting at +y, sweeping about +x."""
    theta = np.linspace(0.0, sweep_rad, n_nodes)
    return np.stack(
        [
            np.zeros(n_nodes),
            radius * np.cos(theta),
            radius * np.sin(theta),
        ],
        axis=1,
    )


def _sphere_tour_guess(
    n_nodes: int,
    radius: float,
    *,
    theta_start: float = 0.5 * np.pi,
    theta_end: float = 0.0,
    phi_start: float = 0.0,
    phi_end: float = np.pi,
) -> np.ndarray:
    """Spherical tour: θ = colatitude from +x, φ = azimuth in yz-plane.

    Default IC matches pos1_init = [0, R, 0] (θ=π/2, φ=0) and sweeps out-of-plane
    toward the +x radial while advancing azimuth.
    """
    theta = np.linspace(theta_start, theta_end, n_nodes)
    phi = np.linspace(phi_start, phi_end, n_nodes)
    sin_theta = np.sin(theta)
    return np.stack(
        [
            radius * np.cos(theta),
            radius * sin_theta * np.cos(phi),
            radius * sin_theta * np.sin(phi),
        ],
        axis=1,
    )


def _make_deputy_states(suffix: str, pos_init: np.ndarray):
    """Create position / velocity / attitude / rate states for one deputy.

    Terminal position is free so the LoS-sweep objective can drive a full
    tour around the chief rather than a fixed 90 deg reorientation.
    """
    position = ox.State(f"position_{suffix}", shape=(3,))
    position.max = np.array([100.0, 100.0, 100.0])
    position.min = np.array([-100.0, -100.0, -100.0])
    position.initial = pos_init
    position.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

    velocity = ox.State(f"velocity_{suffix}", shape=(3,))
    velocity.max = np.full(3, v_max)
    velocity.min = np.full(3, -v_max)
    velocity.initial = [("free", 0.0), ("free", 0.0), ("free", 0.0)]
    velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

    attitude = ox.State(f"attitude_{suffix}", shape=(4,))
    attitude.max = np.ones(4)
    attitude.min = -np.ones(4)
    attitude.initial = [("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]
    attitude.final = [("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]

    angular_velocity = ox.State(f"angular_velocity_{suffix}", shape=(3,))
    angular_velocity.max = np.full(3, w_max)
    angular_velocity.min = np.full(3, -w_max)
    angular_velocity.initial = [("free", 0.0), ("free", 0.0), ("free", 0.0)]
    angular_velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

    return position, velocity, attitude, angular_velocity


def _yz_azimuth_rate(position, velocity):
    """Signed azimuth rate in the yz-plane: φ̇ = (y v_z − z v_y)/(y²+z²)."""
    y = position[1]
    z = position[2]
    return (y * velocity[2] - z * velocity[1]) / (y * y + z * z + 1e-6)


def _colatitude_rate(position, velocity):
    """Colatitude rate from +x: θ̇ = (ρ̇ x − ρ v_x)/‖r‖² with ρ = ‖(y,z)‖."""
    x = position[0]
    y = position[1]
    z = position[2]
    vx = velocity[0]
    vy = velocity[1]
    vz = velocity[2]
    rho = ox.linalg.Norm(position[1:])
    rho_dot = (y * vy + z * vz) / (rho + 1e-6)
    r_sq = ox.Sum(position * position)
    return (rho_dot * x - rho * vx) / (r_sq + 1e-6)


# =============================================================================
# States / controls
# =============================================================================

pos1_init = np.array([0.0, R_nom, 0.0])
pos2_init = -pos1_init

position_1, velocity_1, attitude_1, angular_velocity_1 = _make_deputy_states("1", pos1_init)
position_2, velocity_2, attitude_2, angular_velocity_2 = _make_deputy_states("2", pos2_init)

cost = ox.State("cost", shape=(1,))
cost.initial = np.array([0.0])
cost.final = [("minimize", cost_ub)]
cost.min = np.array([0.0])
cost.max = np.array([cost_ub])
cost.guess = np.zeros((N, 1))

# Unwrapped spherical coordinates of deputy-1 inspection axis (report-only).
view_azim = ox.State("view_azim", shape=(1,))
view_azim.initial = np.array([0.0])
view_azim.final = [ox.Free(0.0)]
view_azim.min = np.array([-0.5])
view_azim.max = np.array([phi_sweep + 0.5])

view_colat = ox.State("view_colat", shape=(1,))
view_colat.initial = np.array([theta_init])
view_colat.final = [ox.Free(0.0)]
view_colat.min = np.array([theta_final - 0.5])
view_colat.max = np.array([theta_init + 0.5])

# Mean squared fractional rate errors for φ̇ and θ̇ about their reference clocks.
uniformity = ox.State("uniformity", shape=(1,))
uniformity.initial = np.array([0.0])
uniformity.final = [ox.Minimize(0.0)]
uniformity.min = np.array([0.0])
uniformity.max = np.array([10.0])

delta_v_1 = ox.Control(
    "delta_v_1",
    shape=(3,),
    parameterization="impulsive",
    nodes=impulse_nodes,
)
delta_v_1.min = -dv_max * np.ones(3)
delta_v_1.max = dv_max * np.ones(3)
delta_v_1.guess = np.zeros((N, 3))

delta_v_2 = ox.Control(
    "delta_v_2",
    shape=(3,),
    parameterization="impulsive",
    nodes=impulse_nodes,
)
delta_v_2.min = -dv_max * np.ones(3)
delta_v_2.max = dv_max * np.ones(3)
delta_v_2.guess = np.zeros((N, 3))

torque_1 = ox.Control("torque_1", shape=(3,))
torque_1.min = -torque_max * np.ones(3)
torque_1.max = torque_max * np.ones(3)
torque_1.guess = np.zeros((N, 3))

torque_2 = ox.Control("torque_2", shape=(3,))
torque_2.min = -torque_max * np.ones(3)
torque_2.max = torque_max * np.ones(3)
torque_2.guess = np.zeros((N, 3))

states = [
    position_1,
    velocity_1,
    attitude_1,
    angular_velocity_1,
    position_2,
    velocity_2,
    attitude_2,
    angular_velocity_2,
    cost,
    view_azim,
    view_colat,
    uniformity,
]
controls = [delta_v_1, delta_v_2, torque_1, torque_2]

# =============================================================================
# Guesses
# =============================================================================

pos1_guess = _sphere_tour_guess(
    N,
    R_nom,
    theta_start=theta_init,
    theta_end=theta_final,
    phi_start=0.0,
    phi_end=phi_sweep,
)
pos2_guess = -pos1_guess
dt = total_time / max(N - 1, 1)
vel1_guess = np.gradient(pos1_guess, dt, axis=0)
vel2_guess = np.gradient(pos2_guess, dt, axis=0)
att1_guess = _pointing_attitudes(pos1_guess)
att2_guess = _pointing_attitudes(pos2_guess)

position_1.guess = pos1_guess
position_2.guess = pos2_guess
velocity_1.guess = vel1_guess
velocity_2.guess = vel2_guess
attitude_1.guess = att1_guess
attitude_2.guess = att2_guess
angular_velocity_1.guess = np.zeros((N, 3))
angular_velocity_2.guess = np.zeros((N, 3))
phi_guess = np.linspace(0.0, phi_sweep, N)
theta_guess = np.linspace(theta_init, theta_final, N)
view_azim.guess = phi_guess.reshape(-1, 1)
view_colat.guess = theta_guess.reshape(-1, 1)
uniformity.guess = np.zeros((N, 1))

# =============================================================================
# Dynamics
# =============================================================================

J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)


def _cw_accel(position, velocity):
    return ox.Concat(
        3.0 * n_mean**2 * position[0] + 2.0 * n_mean * velocity[1],
        -2.0 * n_mean * velocity[0],
        -(n_mean**2) * position[2],
    )


def _attitude_dynamics(attitude, angular_velocity, torque):
    q_norm = ox.linalg.Norm(attitude)
    attitude_normalized = attitude / q_norm
    attitude_dot = 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized
    angular_velocity_dot = ox.linalg.Diag(J_b_inv) @ (
        torque - cross(angular_velocity, J_b_diag @ angular_velocity)
    )
    return attitude_dot, angular_velocity_dot


att1_dot, w1_dot = _attitude_dynamics(attitude_1, angular_velocity_1, torque_1)
att2_dot, w2_dot = _attitude_dynamics(attitude_2, angular_velocity_2, torque_2)

_phi_dot_1 = _yz_azimuth_rate(position_1, velocity_1)
_theta_dot_1 = _colatitude_rate(position_1, velocity_1)
_phi_err = (_phi_dot_1 - omega_phi_ref) / omega_phi_ref
_theta_err = (_theta_dot_1 - omega_theta_ref) / omega_theta_ref

dynamics = {
    "position_1": velocity_1,
    "velocity_1": _cw_accel(position_1, velocity_1),
    "attitude_1": att1_dot,
    "angular_velocity_1": w1_dot,
    "position_2": velocity_2,
    "velocity_2": _cw_accel(position_2, velocity_2),
    "attitude_2": att2_dot,
    "angular_velocity_2": w2_dot,
    "cost": 0.0,
    "view_azim": _phi_dot_1,
    "view_colat": _theta_dot_1,
    "uniformity": (_phi_err * _phi_err + _theta_err * _theta_err) / total_time,
}

# Body-frame impulsive Δv → CW: QDCM(q) maps body → CW/LVLH (same convention
# as the drone examples and the FOV residual below).
eps_impulse = 1e-6
att1_n = attitude_1 / ox.linalg.Norm(attitude_1)
att2_n = attitude_2 / ox.linalg.Norm(attitude_2)
dynamics_discrete = {
    "position_1": position_1,
    "velocity_1": velocity_1 + ox.spatial.QDCM(att1_n) @ delta_v_1,
    "attitude_1": attitude_1,
    "angular_velocity_1": angular_velocity_1,
    "position_2": position_2,
    "velocity_2": velocity_2 + ox.spatial.QDCM(att2_n) @ delta_v_2,
    "attitude_2": attitude_2,
    "angular_velocity_2": angular_velocity_2,
    "cost": cost
    + ox.linalg.Norm(delta_v_1 + eps_impulse)
    + ox.linalg.Norm(delta_v_2 + eps_impulse),
    "view_azim": view_azim,
    "view_colat": view_colat,
    "uniformity": uniformity,
}

# =============================================================================
# Constraints
# =============================================================================

# Prefer nodal convex/nonconvex constraints over CTCS for this dual-vehicle
# problem: a single shared CTCS channel + cross-vehicle residuals was numerically
# unstable (huge aug-state growth). Nodal enforcement at every node is enough
# for the example; continuous FOV between nodes can be revisited later.
constraints = []

for state in states:
    constraints.append(ox.ctcs((state <= state.max)))
    constraints.append(ox.ctcs((state.min <= state)))

for pos in (position_1, position_2):
    constraints.append(ox.ctcs((ox.linalg.Norm(pos) <= R_max)))
    # Reverse-convex keep-out / min-range (nonconvex)
    constraints.append(ox.ctcs(R_min <= ox.linalg.Norm(pos)))

cross_12 = cross(position_1, position_2)
# Squared residual — ‖r1×r2‖ is singular at the antipodal guess (cross=0).
constraints.append(ox.ctcs(ox.Sum(cross_12 * cross_12) <= eps_colinear**2))
# Bilinear opposite-side inequality is nonconvex (not a convex quadratic).
constraints.append(ox.ctcs(ox.Sum(position_1 * position_2) <= 0.0))


def g_vp(p_target_I, x_pos, x_quat):
    p_s = R_sb @ ox.spatial.QDCM(x_quat).T @ (p_target_I - x_pos)
    # Epsilon avoids ‖·‖ singularity when the chief lies exactly on the boresight.
    return ox.linalg.Norm(A_cone @ p_s + 1e-8) - (c_boresight.T @ p_s)


chief = np.zeros(3)
constraints.append(ox.ctcs(g_vp(chief, position_1, attitude_1) <= 0.0))
constraints.append(ox.ctcs(g_vp(chief, position_2, attitude_2) <= 0.0))

for node in impulse_nodes:
    constraints.append((ox.linalg.Norm(delta_v_1) <= dv_max).convex().at([node]))
    constraints.append((ox.linalg.Norm(delta_v_2) <= dv_max).convex().at([node]))

# =============================================================================
# Problem
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=total_time,
    min=0.0,
    max=total_time,
    guess=np.linspace(0.0, total_time, N).reshape(-1, 1),
)

problem = Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N,
    float_dtype="float64",
    licq_max=1e-10,
    algorithm={
        "lam_prox": 5e0,
        "lam_vc": 1e1,
        "lam_cost": {"cost": 1e-1, "uniformity": 2.0},
        "autotuner": ox.ConstantProximalWeight(),
        # "autotuner": ox.AugmentedLagrangian(eta_lambda=1e2),
    },
    discretizer={"diffrax_kwargs": {"atol": 1e-6, "rtol": 1e-6}},
)

plotting_dict = {
    "mean_motion": n_mean,
    "R_min": R_min,
    "R_max": R_max,
    "R_sb": np.asarray(R_sb),
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "norm_type": norm_type,
    "init_poses": np.array([[0.0, 0.0, 0.0]]),
}


def _print_summary(results) -> None:
    cost_nodes = np.asarray(results.nodes["cost"], dtype=float).reshape(-1)
    azim_nodes = np.asarray(results.nodes["view_azim"], dtype=float).reshape(-1)
    colat_nodes = np.asarray(results.nodes["view_colat"], dtype=float).reshape(-1)
    unif_nodes = np.asarray(results.nodes["uniformity"], dtype=float).reshape(-1)
    dv1 = np.asarray(results.nodes["delta_v_1"], dtype=float)
    dv2 = np.asarray(results.nodes["delta_v_2"], dtype=float)
    pos1 = np.asarray(results.nodes["position_1"], dtype=float)
    pos2 = np.asarray(results.nodes["position_2"], dtype=float)

    print(f"Converged: {bool(results.converged)}")
    print(f"Final accumulated Δv (cost): {cost_nodes[-1]:.6e} m/s")
    print(f"Sum ‖Δv1‖ over nodes: {np.linalg.norm(dv1, axis=1).sum():.6e} m/s")
    print(f"Sum ‖Δv2‖ over nodes: {np.linalg.norm(dv2, axis=1).sum():.6e} m/s")
    print(
        f"LoS azimuth φ: {azim_nodes[-1]:.3f} rad "
        f"({np.rad2deg(azim_nodes[-1]):.1f} deg) / target {np.rad2deg(phi_sweep):.1f} deg"
    )
    print(
        f"LoS colatitude θ: {colat_nodes[-1]:.3f} rad "
        f"({np.rad2deg(colat_nodes[-1]):.1f} deg) / target {np.rad2deg(theta_final):.1f} deg"
    )
    print(
        f"Spherical sweep uniformity (RMS frac. rate error): "
        f"{np.sqrt(max(unif_nodes[-1], 0.0)):.4f} (0 = uniform); "
        f"ω_φ={omega_phi_ref:.3e}, ω_θ={omega_theta_ref:.3e} rad/s"
    )
    x1 = pos1[:, 0]
    print(
        f"Out-of-plane range |z1|: [{np.abs(pos1[:, 2]).min():.3f}, "
        f"{np.abs(pos1[:, 2]).max():.3f}] m; "
        f"x1 (radial offset): [{x1.min():.3f}, {x1.max():.3f}] m"
    )
    print(
        f"‖r1‖ range: [{np.linalg.norm(pos1, axis=1).min():.3f}, "
        f"{np.linalg.norm(pos1, axis=1).max():.3f}] m"
    )
    print(
        f"‖r2‖ range: [{np.linalg.norm(pos2, axis=1).min():.3f}, "
        f"{np.linalg.norm(pos2, axis=1).max():.3f}] m"
    )
    cross_norm = np.linalg.norm(np.cross(pos1, pos2), axis=1)
    print(f"‖r1 × r2‖ max: {cross_norm.max():.6e} m^2 (eps={eps_colinear})")
    print(f"max r1·r2: {np.sum(pos1 * pos2, axis=1).max():.6e} m^2")


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    _print_summary(results)
    plot_states(results).show()
    plot_controls(results).show()

    # Build deputy-1 scene without starting playback, then attach deputy 2 and
    # a shared diameter line before wiring the animation GUI.
    handle = create_animated_plotting_server(
        results,
        position_key="position_1",
        velocity_key="velocity_1",
        attitude_key="attitude_1",
        thrust_key="delta_v_1",
        show_viewcone=True,
        viewcone_scale=R_max,
        show_control_norm_plot="delta_v_1",
        show_grid=False,
        target_radius=2.0,
        controls="manual",
    )

    # Deputy 2 shares the server with deputy 1, so each primitive gets its own scene path.
    pos2_traj = np.asarray(results.trajectory["position_2"], dtype=np.float64)
    att2_traj = np.asarray(results.trajectory["attitude_2"], dtype=np.float64)
    vel2_traj = np.asarray(results.trajectory["velocity_2"], dtype=np.float64)
    colors_2 = compute_velocity_colors(vel2_traj)

    add_ghost_trajectory(handle.server, pos2_traj, colors_2, opacity=0.35, name="/deputy_2/ghost")
    _, update_trail_2 = add_animated_trail(
        handle.server, pos2_traj, colors_2, point_size=0.25, name="/deputy_2/trail"
    )
    _, update_marker_2 = add_position_marker(
        handle.server, pos2_traj, radius=1.2, color=(255, 160, 60), name="/deputy_2/marker"
    )
    _, update_body_2 = add_attitude_frame(
        handle.server,
        pos2_traj,
        att2_traj,
        axes_length=3.0,
        axes_radius=0.08,
        name="/deputy_2/body",
    )
    _, update_viewcone_2 = add_viewcone(
        handle.server,
        pos2_traj,
        att2_traj,
        half_angle_x=np.pi / alpha_x,
        half_angle_y=np.pi / alpha_y,
        scale=R_max,
        norm_type=norm_type,
        R_sb=np.asarray(R_sb, dtype=np.float64),
        color=(255, 140, 40),
        opacity=0.35,
        name="/deputy_2/viewcone",
    )
    handle.update_callbacks += [
        update_trail_2,
        update_marker_2,
        update_body_2,
        update_viewcone_2,
    ]

    # Collinear diameter through the chief (deputy 1 ↔ origin ↔ deputy 2)
    pos1_traj = np.asarray(results.trajectory["position_1"], dtype=np.float64)
    diameter = handle.server.scene.add_line_segments(
        "/diameter",
        points=np.array([[pos1_traj[0], pos2_traj[0]]], dtype=np.float32),
        colors=(220, 220, 220),
        line_width=2.5,
    )

    def update_diameter(frame_idx: int) -> None:
        diameter.points = np.array([[pos1_traj[frame_idx], pos2_traj[frame_idx]]], dtype=np.float32)

    handle.update_callbacks.append(update_diameter)

    # Annular shell (inner / outer); chief is already marked via init_poses
    handle.server.scene.add_icosphere(
        "/annulus/r_min",
        radius=R_min,
        color=(80, 180, 120),
        opacity=0.08,
    )
    handle.server.scene.add_icosphere(
        "/annulus/r_max",
        radius=R_max,
        color=(80, 140, 200),
        opacity=0.08,
    )

    add_animation_controls(
        handle.server,
        handle.traj_time,
        handle.update_callbacks,
        loop=True,
    )
    handle.step(0)
    handle.server.sleep_forever()
