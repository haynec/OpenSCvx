"""UR5e laser tracing — boresight path following on a work surface via frax dynamics.

The arm counterpart of ``examples/drone/openscvx_logo.py``: instead of a
quadrotor aiming its boresight at a moving logo target in the air, a UR5e
points a tool-mounted laser at a target moving along a drawing path on the
work surface the robot is mounted on (the plane z = 0).

Formulation
-----------
- ``FraxDynamics`` — full rigid-body joint-space dynamics from the UR5e URDF.
- A moving target ``p_target(t / T)`` sweeps the drawing path exactly once
  over the (fixed) mission time.
- A ``misalignment`` state integrates ``1 - cos(theta)``, where ``theta`` is
  the angle between the tool axis (EE local +z, the laser) and the line of
  sight to the target (BYOF dynamics, since it needs frax FK). Its state
  upper bound caps the average pointing error through CTCS, and its final
  value is minimized.
- The initial guess is the "clever" part: sequential damped-least-squares IK
  through frax's own FK tracks the path at a standoff height with the tool
  axis normal to the surface, so the guess already points at the target and
  SCP only has to trade pointing accuracy against the dynamics. The same
  initialization works for any drawing path, not just the circle used here.

Requires
--------
    pip install openscvx[frax]
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import frax
except ImportError:
    print(
        "frax is not installed. Install with: pip install openscvx[frax]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox

# =============================================================================
# Robot and dynamics
# =============================================================================

robot = frax.Manipulator(os.path.join(current_dir, "ur5e_assets", "ur5e.urdf"))
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls
n_j = robot.num_joints  # 6

# =============================================================================
# Drawing path on the work surface
# =============================================================================
# The work surface is the plane z = 0 (the robot is mounted on it). The
# drawing path is a closed circle in front of the base, parameterized by
# normalized path progress s in [0, 1]. Swap ``path_xy`` for any other
# arc-length-parameterized curve (e.g. an SVG outline) to trace a different
# figure.

path_center = np.array([0.45, 0.0])
path_radius = 0.15


def path_xy(s):
    """Drawing path in table coordinates: s in [0, 1] -> (x, y) in metres."""
    ang = 2.0 * jnp.pi * s
    return jnp.stack(
        [
            path_center[0] + path_radius * jnp.cos(ang),
            path_center[1] + path_radius * jnp.sin(ang),
        ]
    )


def target_position(s):
    """Moving laser target on the work surface at path progress s in [0, 1]."""
    xy = path_xy(s)
    return jnp.array([xy[0], xy[1], 0.0])


def path_progress(t_frac):
    """Eased path progress s(t/T): zero target speed at both lap ends.

    The arm starts and ends at rest, so an easing profile lets it track the
    target through the whole lap instead of chasing a step in target speed.
    """
    return t_frac - jnp.sin(2.0 * jnp.pi * t_frac) / (2.0 * jnp.pi)


def target_at_time(t):
    """Target position at mission time t (composition used everywhere below)."""
    return target_position(path_progress(t / total_time))


# =============================================================================
# Discretization
# =============================================================================

n = 40
total_time = 10.0  # target completes the path exactly once

# =============================================================================
# Initial guess: IK tracking of the path at a standoff height
# =============================================================================
# Desired EE poses hover ``standoff`` metres above the moving target with the
# tool axis pointing straight down at it. Sequential damped-least-squares IK
# (each node seeded by the previous solution) through frax FK turns those
# poses into a smooth joint-space trajectory on a single IK branch.

standoff = 0.25
_R_down = jnp.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_q_seed = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def _so3_log(R):
    """SO(3) logarithm: rotation matrix -> rotation vector (axis * angle)."""
    cos_a = jnp.clip((jnp.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_a)
    skew = (R - R.T) / 2.0
    omega = jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    safe_angle = jnp.where(jnp.abs(angle) < 1e-10, 1.0, angle)
    scale = jnp.where(jnp.abs(angle) < 1e-10, 1.0, safe_angle / jnp.sin(safe_angle))
    return omega * scale


def _pose_error(q_val, p_des):
    """6D error between the EE pose and (p_des, tool-down orientation)."""
    T = robot.ee_transform(q_val)
    return jnp.concatenate([T[:3, 3] - p_des, _so3_log(_R_down.T @ T[:3, :3])])


@jax.jit
def _ik_step(q_val, p_des):
    """One damped-least-squares Newton step toward the desired EE pose."""
    err = _pose_error(q_val, p_des)
    J = jax.jacfwd(_pose_error)(q_val, p_des)
    return q_val - J.T @ jnp.linalg.solve(J @ J.T + 1e-6 * jnp.eye(6), err)


def ik_track_path(n_nodes: int, iters: int = 50) -> np.ndarray:
    """Joint trajectory whose EE follows the path at the standoff height."""
    q_traj = np.zeros((n_nodes, n_j))
    q_val = jnp.array(_q_seed)
    for k in range(n_nodes):
        t_k = total_time * k / (n_nodes - 1)
        p_des = target_at_time(t_k) + jnp.array([0.0, 0.0, standoff])
        for _ in range(iters):
            q_val = _ik_step(q_val, p_des)
        q_traj[k] = np.asarray(q_val)
    return q_traj


q_guess = ik_track_path(n)
qd_guess = np.gradient(q_guess, np.linspace(0.0, total_time, n), axis=0)

q.initial = q_guess[0]
q.final = [("free", 0.0)] * n_j
qd.initial = np.zeros(n_j)
qd.final = np.zeros(n_j)

q.guess = q_guess
qd.guess = qd_guess
tau.guess = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q_guess])

# =============================================================================
# Pointing metric state (the arm counterpart of the drone logo's angle_metric)
# =============================================================================
# d(misalignment)/dt = 1 - cos(theta), where theta is the angle between the
# laser and the line of sight to the target, so the state accumulates pointing
# error over the mission. The upper bound caps it continuously (CTCS) and the
# final value is minimized. 1 - cos(theta) ~= theta^2 / 2 near alignment and,
# unlike the angle itself, has a bounded Jacobian there — with arccos the
# linearization degenerates at alignment and SCP stalls on phantom defects.

misalignment = ox.State("misalignment", shape=(1,))
misalignment.min = np.array([0.0])
misalignment.max = np.array([0.02])  # over the mission; 0.02 ~= 3.6 deg RMS
misalignment.initial = np.array([0.0])
misalignment.final = [ox.Minimize(0.0)]
misalignment.guess = np.zeros((n, 1))

time = ox.Time(initial=0.0, final=total_time, min=0.0, max=total_time)


def laser_cos_to_target(q_val, t_val):
    """Cosine of the angle between the tool axis (EE local +z) and the sight line."""
    T = robot.ee_transform(q_val)
    p_ee = jnp.asarray(T)[:3, 3]
    laser = jnp.asarray(T)[:3, 2]
    to_target = target_at_time(t_val) - p_ee
    return jnp.dot(to_target, laser) / jnp.linalg.norm(to_target)


def laser_angle_to_target(q_val, t_val):
    """Pointing angle in radians (for reporting; not used in the optimization)."""
    return jnp.arccos(jnp.clip(laser_cos_to_target(q_val, t_val), -1.0, 1.0))


def _misalignment_dynamics(x, u, node, params):
    return jnp.array([1.0 - laser_cos_to_target(x[q.slice], x[time.slice][0])])


# --- CTCS: EE stays above the work surface ----------------------------------
def _table_clearance_ctcs(x, u, node, params):
    ee_z = jnp.asarray(robot.ee_transform(x[q.slice]))[2, 3]
    return 0.05 - ee_z  # <= 0 when the EE keeps 5 cm of clearance


byof: ox.ByofSpec = {
    "dynamics": {"misalignment": _misalignment_dynamics},
    "ctcs_constraints": [{"constraint_fn": _table_clearance_ctcs}],
}

# =============================================================================
# Constraints and Problem
# =============================================================================

states = [*dyn.states, misalignment]

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

problem = ox.Problem(
    dynamics=dyn,
    states=states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    byof=byof,
    N=n,
    algorithm={
        # The terminal misalignment is O(1e-4) — the default lam_cost of 1e-2
        # leaves it under-priced against the proximal term, and the solution
        # lags the target at peak path speed.
        "lam_cost": 1e1,
        "lam_vb": 1e0,
        # lam_vc must price virtual control well above the pointing cost:
        # otherwise the solver "erases" accumulated misalignment through
        # defects instead of steering the arm.
        "lam_vc": 1e4,
        "lam_prox": 1e1,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
    # misalignment is O(1e-4..1e-2), far below the default diffrax atol of
    # 1e-3 — tighten so the pointing metric is actually resolved between nodes.
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-8, "rtol": 1e-8}),
)

# =============================================================================
# Post-processing: where the laser actually hits the surface
# =============================================================================


def laser_trace(q_traj: np.ndarray) -> np.ndarray:
    """Intersect the laser ray with the work surface (z = 0) along a trajectory.

    Points where the laser is parallel to the surface or pointing away from it
    are returned as NaN so plots show a gap rather than a spurious dot.
    """
    trace = np.full((len(q_traj), 3), np.nan)
    for k, q_val in enumerate(np.asarray(q_traj)):
        T = np.asarray(robot.ee_transform(q_val))
        p_ee, laser = T[:3, 3], T[:3, 2]
        if laser[2] < -1e-8:
            trace[k] = p_ee - (p_ee[2] / laser[2]) * laser
    return trace


# =============================================================================
# Visualization
# =============================================================================


def visualize(results) -> None:
    """Animate in Viser: stick-model arm, laser ray, and the trace it draws."""
    from openscvx.plotting.viser import add_animation_controls, create_server

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])
    prop = results.multishot_propagation()
    if prop is not None:
        q_traj, t_vec = prop.state("q")

    n_frames = len(q_traj)
    trace = laser_trace(q_traj)
    target_path = np.array([np.asarray(target_position(s)) for s in np.linspace(0, 1, 200)])

    keypoints = np.zeros((n_frames, n_j + 2, 3))
    ee_T = np.zeros((n_frames, 4, 4))
    for k in range(n_frames):
        links = np.asarray(robot.link_to_world_transforms(q_traj[k]))
        ee_T[k] = np.asarray(robot.ee_transform(q_traj[k]))
        keypoints[k, 1 : 1 + n_j] = links[:, :3, 3]
        keypoints[k, -1] = ee_T[k][:3, 3]

    server = create_server(keypoints[:, -1], show_grid=False)
    server.scene.add_grid("/table", width=1.6, height=1.6, cell_size=0.2)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    # Target path (thin line) drawn on the surface
    server.scene.add_line_segments(
        "/target_path",
        points=np.stack([target_path[:-1], target_path[1:]], axis=1),
        colors=np.full((len(target_path) - 1, 2, 3), (90, 200, 120), dtype=np.uint8),
        line_width=2.0,
    )

    link_rgb = np.linspace([80, 100, 180], [255, 120, 80], n_j + 1).astype(np.uint8)
    link_colors = np.stack([link_rgb, link_rgb], axis=1)

    def _arm_segments(frame: int) -> np.ndarray:
        pts = np.zeros((n_j + 1, 2, 3), dtype=np.float32)
        for k in range(n_j + 1):
            pts[k] = [keypoints[frame, k], keypoints[frame, k + 1]]
        return pts

    arm_handle = server.scene.add_line_segments(
        "/ur5e_links", points=_arm_segments(0), colors=link_colors, line_width=6.0
    )
    laser_handle = server.scene.add_line_segments(
        "/laser",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.full((1, 2, 3), (255, 40, 40), dtype=np.uint8),
        line_width=3.0,
    )
    trace_handle = server.scene.add_point_cloud(
        "/laser_trace",
        points=trace[:1],
        colors=np.full((1, 3), (255, 40, 40), dtype=np.uint8),
        point_size=0.004,
    )

    def update(frame: int) -> None:
        arm_handle.points = _arm_segments(frame)
        p_ee = keypoints[frame, -1]
        hit = trace[frame] if np.isfinite(trace[frame]).all() else p_ee
        laser_handle.points = np.array([[p_ee, hit]], dtype=np.float32)
        drawn = trace[: frame + 1]
        drawn = drawn[np.isfinite(drawn).all(axis=1)]
        if len(drawn):
            trace_handle.points = drawn
            trace_handle.colors = np.full((len(drawn), 3), (255, 40, 40), dtype=np.uint8)

    add_animation_controls(server, t_vec, [update], loop=True)
    server.sleep_forever()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("UR5e laser tracing via frax dynamics")
    print("=" * 60)
    print(f"Nodes: {n}  |  Mission time: {total_time} s")
    print(f"Path: circle r={path_radius} m at {list(path_center)} on the surface z=0")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    q_traj = np.asarray(results.trajectory["q"])
    t_traj = np.asarray(results.trajectory["time"]).flatten()
    trace = laser_trace(q_traj)
    targets = np.array([np.asarray(target_at_time(t)) for t in t_traj])
    trace_err = np.linalg.norm(trace[:, :2] - targets[:, :2], axis=1)

    print("\nResults:")
    print(f"  Accumulated misalignment: {results.nodes['misalignment'][-1, 0]:.5f} s")
    print(f"  Laser dot error on surface: mean {np.nanmean(trace_err) * 1000:.1f} mm, "
          f"max {np.nanmax(trace_err) * 1000:.1f} mm")
    print(f"  CTCS violation: {results.ctcs_violation}")
    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results)
