"""UR5e pen tracing — drawing on a work surface with a tool-mounted pen via frax.

The contact companion of ``examples/frax/ur5e_laser_trace.py``: instead of
pointing a laser at the moving target, the arm holds a pen of length
``pen_length`` along the tool axis and draws the path directly — the pen tip
must stay on the work surface (the plane z = 0) while tracking the target.

Formulation
-----------
- ``FraxDynamics`` — full rigid-body joint-space dynamics from the UR5e URDF.
- The drawing path comes from an SVG file (a circle by default), resampled by
  arc length, eased in time, and baked into per-coordinate ``Cinterp`` cubic
  splines: inside the solver the moving target is a C2-smooth symbolic
  function of the time state.
- The pen is a rigid stick of length ``pen_length`` along the tool axis
  (EE local +z). ``pen_length`` is an ``ox.Parameter``: update its ``value``
  and re-solve to fit whatever pen ends up in the gripper — no rebuild.
- Contact: a CTCS band keeps the tip within ``contact_tol`` of the surface,
  and a tilt cone keeps the pen within ``tilt_max`` of the surface normal.
- Tracking: a ``tip_error`` state integrates the squared in-plane distance
  between the pen tip and the target (BYOF dynamics, since it needs frax FK).
  Its upper bound caps the average tracking error through CTCS, and its final
  value is minimized.
- Initial guess: sequential damped-least-squares IK through frax's own FK
  places the tip exactly on the moving target with the pen vertical, so the
  guess writes the figure perfectly and SCP only has to trade tracking
  against the dynamics.

Requires
--------
    pip install openscvx[frax] svgpathtools
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

try:
    import svgpathtools
except ImportError:
    print(
        "svgpathtools is not installed. Install with: pip install svgpathtools",
        file=sys.stderr,
    )
    sys.exit(1)

from scipy.interpolate import CubicSpline

import openscvx as ox
from openscvx.symbolic.lower import lower_to_jax

# =============================================================================
# Robot and dynamics
# =============================================================================

robot = frax.Manipulator(os.path.join(current_dir, "ur5e_assets", "ur5e.urdf"))
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls
n_j = robot.num_joints  # 6

# =============================================================================
# Pen
# =============================================================================
# Rigid pen of length ``pen_length`` held along the tool axis. The Parameter
# reaches the BYOF functions as ``params["pen_length"]``; assign a new
# ``pen_length.value`` and re-solve to change pens without rebuilding the
# problem (the IK guess below is built for the current value, so re-run it
# too if the length changes a lot).

pen_length = ox.Parameter("pen_length", shape=(1,), value=np.array([0.15]))
contact_tol = 0.002  # tip-to-surface band [m]
tilt_max = np.deg2rad(30.0)  # max pen tilt from the surface normal

# =============================================================================
# Discretization and time
# =============================================================================

n = 150
total_time = 45.0  # target completes the path exactly once (~3.3 m of drawing)

time = ox.Time(initial=0.0, final=total_time, min=0.0, max=total_time)

# =============================================================================
# Drawing path: SVG -> arc-length polyline -> eased time table -> Cinterp
# =============================================================================
# The work surface is the plane z = 0 (the robot is mounted on it). The SVG
# outline is scaled uniformly into a ``path_size`` box centred on
# ``path_center`` in front of the base.

# The OpenSCvx wordmark (single continuous outline, path 0 of the SVG the
# drone logo example uses; paths 1-2 are its frame). Laid across the table —
# long axis on y — via the 90 degree rotation. Swap in any other SVG
# (e.g. ur5e_assets/circle.svg with default indices/rotation) to trace it.
trace_svg = os.path.join(
    grandparent_dir, "examples", "drone", "logo_utils", "openscvx_logo_single.svg"
)
path_indices = [0]
rotation_deg = 90.0
path_center = np.array([0.45, 0.0])
path_size = 0.50  # bounding-box size of the drawing on the surface [m]


def svg_polyline(svg_file: str, n_points: int = 4000) -> np.ndarray:
    """Uniform-arc-length polyline through the SVG's paths, fitted to the drawing box.

    Concatenates the selected ``path_indices`` (all paths when None) in order,
    samples them densely, then resamples by cumulative arc length so equal
    index steps cover equal distances. The result is y-flipped (SVG y points
    down), rotated by ``rotation_deg``, scaled uniformly to fit ``path_size``,
    and centred on ``path_center``.
    """
    paths, _ = svgpathtools.svg2paths(svg_file)
    if path_indices is not None:
        paths = [paths[i] for i in path_indices]
    dense = []
    for path in paths:
        for seg in path:
            n_seg = max(2, int(4 * n_points * seg.length() / sum(p.length() for p in paths)))
            for t in np.linspace(0.0, 1.0, n_seg, endpoint=False):
                pt = seg.point(t)
                dense.append([pt.real, -pt.imag])
    dense = np.asarray(dense + [dense[0]] if _is_closed(paths) else dense)
    ang = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dense = dense @ rot.T

    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    s_uniform = np.linspace(0.0, arc[-1], n_points)
    pts = np.column_stack([np.interp(s_uniform, arc, dense[:, i]) for i in range(2)])

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pts = (pts - 0.5 * (lo + hi)) * (path_size / (hi - lo).max())
    return pts + path_center


def _is_closed(paths) -> bool:
    """True when the SVG traces a closed figure (end point returns to start)."""
    start, end = paths[0].start, paths[-1].end
    return abs(end - start) < 1e-6 * sum(p.length() for p in paths)


def path_progress(t_frac, ramp: float = 0.08):
    """Eased time progress: smoothstep speed ramps at both ends of the trace.

    The arm starts and ends at rest, so the target speed ramps from and to
    zero over the first and last ``ramp`` fraction of the mission — and stays
    near-constant in between, where curvature pacing governs the speed. (A
    full-trace sine ease would double the mid-trace speed instead.)
    """
    up = np.clip(t_frac / ramp, 0.0, 1.0)
    down = np.clip((1.0 - t_frac) / ramp, 0.0, 1.0)
    speed = (3 * up**2 - 2 * up**3) * (3 * down**2 - 2 * down**3)
    progress = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    return progress / progress[-1]


def curvature_pacing(polyline: np.ndarray, kappa_ref: float = 50.0, w_max: float = 8.0):
    """Map time fraction -> path progress, slowing the target through corners.

    Allocates trace time in proportion to a curvature weight ``1 + |kappa| /
    kappa_ref`` (clipped at ``w_max``), so sharp glyph corners get up to
    ``w_max`` times more time per unit arc length than straight strokes. With
    uniform pacing the tightest corners dominate the tracking error budget.

    Returns (time_fraction, progress) tables: monotone arrays mapping the
    fraction of trace time spent to the fraction of arc length covered.
    """
    xp, yp = np.gradient(polyline[:, 0]), np.gradient(polyline[:, 1])
    xpp, ypp = np.gradient(xp), np.gradient(yp)
    # Per-index derivatives suffice: on a uniform-arc-length polyline the
    # sample spacing cancels out of the curvature formula.
    kappa = np.abs(xp * ypp - yp * xpp) / np.maximum((xp**2 + yp**2) ** 1.5, 1e-12)
    window = np.ones(31) / 31.0  # ~8 mm smoothing at 0.5 m path size
    kappa = np.convolve(kappa, window, mode="same")
    weight = np.clip(1.0 + kappa / kappa_ref, 1.0, w_max)
    time_frac = np.concatenate([[0.0], np.cumsum(weight[:-1])])
    return time_frac / time_frac[-1], np.linspace(0.0, 1.0, len(polyline))


# Tabulate the target as a function of mission time: uniform time breakpoints,
# eased and curvature-paced progress, positions read off the arc-length
# polyline. Cubic splines through this table give the solver a C2-smooth
# moving target regardless of how the SVG itself is sampled.
_polyline = svg_polyline(trace_svg)
_time_frac, _s_polyline = curvature_pacing(_polyline)
_t_table = np.linspace(0.0, total_time, 1000)
_s_table = np.interp(path_progress(_t_table / total_time), _time_frac, _s_polyline)
_xy_table = np.column_stack(
    [np.interp(_s_table, _s_polyline, _polyline[:, i]) for i in range(2)]
)

# Numpy-side spline (initial guess, post-processing, plots) and its symbolic
# twin (Cinterp, same not-a-knot cubic through the same table) for the solver.
_target_spline = CubicSpline(_t_table, _xy_table)
_target_exprs = [ox.Cinterp(time[0], _t_table, _xy_table[:, i]) for i in range(2)]


def target_at_time(t) -> np.ndarray:
    """Moving pen target on the work surface at mission time t."""
    x_t, y_t = _target_spline(np.clip(t, 0.0, total_time))
    return np.array([x_t, y_t, 0.0])


# =============================================================================
# Initial guess: IK writing of the path with the pen vertical
# =============================================================================
# Desired EE poses put the pen tip exactly on the moving target with the tool
# axis pointing straight down, i.e. the EE hovers one pen length above the
# surface. Sequential damped-least-squares IK (each node seeded by the
# previous solution) through frax FK turns those poses into a smooth
# joint-space trajectory on a single IK branch.

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
    """Joint trajectory whose pen tip writes the path with the pen vertical."""
    ee_height = float(pen_length.value[0])
    q_traj = np.zeros((n_nodes, n_j))
    q_val = jnp.array(_q_seed)
    for k in range(n_nodes):
        t_k = total_time * k / (n_nodes - 1)
        p_des = jnp.array(target_at_time(t_k) + np.array([0.0, 0.0, ee_height]))
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
# Tip tracking metric state
# =============================================================================
# d(tip_error)/dt = || tip_xy - target_xy ||^2, so the state accumulates the
# squared in-plane tracking error of the pen tip over the mission [m^2 s].
# The upper bound caps it continuously (CTCS) and the final value is
# minimized. The squared distance is smooth at zero error, so the
# linearization stays well conditioned on the (exact) IK guess.

tip_error = ox.State("tip_error", shape=(1,))
tip_error.min = np.array([0.0])
# Loose guard only — Minimize provides the tracking pressure. A tight cap
# that the true integral cannot meet forces the solver into defect-cheating
# (it "erases" the metric through virtual control instead of steering).
tip_error.max = np.array([1e-2])
tip_error.initial = np.array([0.0])
tip_error.final = [ox.Minimize(0.0)]
tip_error.guess = np.zeros((n, 1))

# The Cinterp target lowers to JAX callables with the BYOF signature. Lowering
# resolves state slices, which exist only after Problem preprocessing — so
# lower lazily on the first call.
_lowered_target: list = []


def _target_fn(x, u, node, params):
    """Symbolic Cinterp target evaluated inside BYOF: (x, u, node, params) -> (2,)."""
    if not _lowered_target:
        _lowered_target.extend(lower_to_jax(_target_exprs))
    fx, fy = _lowered_target
    return jnp.array([fx(x, u, node, params), fy(x, u, node, params)])


def _tip_position(x, params):
    """Pen tip in world frame: EE position + pen_length along the tool axis."""
    T = jnp.asarray(robot.ee_transform(x[q.slice]))
    return T[:3, 3] + params["pen_length"][0] * T[:3, 2]


def _tip_error_dynamics(x, u, node, params):
    tip_xy = _tip_position(x, params)[:2]
    return jnp.array([jnp.sum((tip_xy - _target_fn(x, u, node, params)) ** 2)])


# --- Contact: pen tip stays on the surface -----------------------------------
# Enforced twice: hard nodal inequalities at every node (linear in tip z, so
# they linearize exactly), plus a CTCS penalty for the segments in between.
# The CTCS residual is scaled to millimetres: the raw square penalty of a
# metre-scale residual has a ~1e-3 gradient at mm violations — far too flat
# to push back against the proximal term.
_CONTACT_SCALE = 1e3  # residual in mm


def _tip_above_surface(x, u, node, params):
    return -_tip_position(x, params)[2] - contact_tol


def _tip_below_surface(x, u, node, params):
    return _tip_position(x, params)[2] - contact_tol


def _contact_ctcs(x, u, node, params):
    return _CONTACT_SCALE * (jnp.abs(_tip_position(x, params)[2]) - contact_tol)


_tilt_cos_max = float(np.cos(tilt_max))


def _tilt_ctcs(x, u, node, params):
    # Tool axis z-component is -1 when the pen points straight down; satisfied
    # (<= 0) while the pen tilts less than tilt_max from the surface normal.
    tool_z = jnp.asarray(robot.ee_transform(x[q.slice]))[2, 2]
    return tool_z + _tilt_cos_max


byof: ox.ByofSpec = {
    "parameters": [pen_length],
    "dynamics": {"tip_error": _tip_error_dynamics},
    "nodal_constraints": [
        {"constraint_fn": _tip_above_surface, "nodes": list(range(n))},
        {"constraint_fn": _tip_below_surface, "nodes": list(range(n))},
    ],
    "ctcs_constraints": [
        # The contact band gets its own augmented state (idx 1) with a tight
        # violation budget — sharing the box-constraint channel would let the
        # tip drift millimetres off the surface within the shared budget.
        {"constraint_fn": _contact_ctcs, "idx": 1, "bounds": (0.0, 1e-4)},
        {"constraint_fn": _tilt_ctcs},
    ],
}

# =============================================================================
# Constraints and Problem
# =============================================================================

states = [*dyn.states, tip_error]

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
        # The terminal tip_error is O(1e-5) — the default lam_cost of 1e-2
        # leaves it under-priced against the proximal term, and the solution
        # lags the target at peak path speed.
        "lam_cost": 1e1,
        "lam_vb": 1e0,
        # lam_vc must price virtual control well above the tracking cost:
        # otherwise the solver "erases" accumulated tip_error through defects
        # instead of steering the arm.
        "lam_vc": 1e4,
        "lam_prox": 1e1,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
    # tip_error is O(1e-6..1e-3), far below the default diffrax atol of 1e-3 —
    # tighten so the tracking metric is actually resolved between nodes.
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-8, "rtol": 1e-8}),
)

# =============================================================================
# Post-processing: pen tip path
# =============================================================================


def pen_tip_path(q_traj: np.ndarray) -> np.ndarray:
    """World-frame pen tip positions along a joint trajectory."""
    length = float(pen_length.value[0])
    tips = np.zeros((len(q_traj), 3))
    for k, q_val in enumerate(np.asarray(q_traj)):
        T = np.asarray(robot.ee_transform(q_val))
        tips[k] = T[:3, 3] + length * T[:3, 2]
    return tips


# =============================================================================
# Visualization
# =============================================================================


def visualize(results) -> None:
    """Animate in Viser: stick-model arm, pen, and the line it draws."""
    from openscvx.plotting.viser import add_animation_controls, create_server

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])
    prop = results.multishot_propagation()
    if prop is not None:
        q_traj, t_vec = prop.state("q")

    n_frames = len(q_traj)
    tips = pen_tip_path(q_traj)
    target_path = np.column_stack([_polyline, np.zeros(len(_polyline))])

    keypoints = np.zeros((n_frames, n_j + 2, 3))
    for k in range(n_frames):
        links = np.asarray(robot.link_to_world_transforms(q_traj[k]))
        keypoints[k, 1 : 1 + n_j] = links[:, :3, 3]
        keypoints[k, -1] = np.asarray(robot.ee_transform(q_traj[k]))[:3, 3]

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
    pen_handle = server.scene.add_line_segments(
        "/pen",
        points=np.array([[keypoints[0, -1], tips[0]]], dtype=np.float32),
        colors=np.full((1, 2, 3), (40, 40, 40), dtype=np.uint8),
        line_width=4.0,
    )
    ink_handle = server.scene.add_point_cloud(
        "/ink",
        points=tips[:1],
        colors=np.full((1, 3), (200, 30, 30), dtype=np.uint8),
        point_size=0.004,
    )

    def update(frame: int) -> None:
        arm_handle.points = _arm_segments(frame)
        pen_handle.points = np.array([[keypoints[frame, -1], tips[frame]]], dtype=np.float32)
        drawn = tips[: frame + 1]
        ink_handle.points = drawn
        ink_handle.colors = np.full((len(drawn), 3), (200, 30, 30), dtype=np.uint8)

    add_animation_controls(server, t_vec, [update], loop=True)
    server.sleep_forever()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("UR5e pen tracing via frax dynamics")
    print("=" * 60)
    print(f"Nodes: {n}  |  Mission time: {total_time} s")
    print(f"Pen length: {float(pen_length.value[0]) * 100:.0f} cm  |  "
          f"contact band: +/-{contact_tol * 1000:.0f} mm")
    print(f"Path: {os.path.basename(trace_svg)}, {path_size} m wide at {list(path_center)}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    # Measure on the multishot propagation: a single 10 s open-loop torque
    # replay of an arm diverges, so results.trajectory is not meaningful here.
    prop = results.multishot_propagation()
    q_traj, t_traj = prop.state("q")
    tips = pen_tip_path(q_traj)
    targets = np.array([target_at_time(t) for t in t_traj])
    tip_err = np.linalg.norm(tips[:, :2] - targets[:, :2], axis=1)

    print("\nResults:")
    print(f"  Tip tracking error: mean {tip_err.mean() * 1000:.1f} mm, "
          f"max {tip_err.max() * 1000:.1f} mm")
    print(f"  Tip height: min {tips[:, 2].min() * 1000:.1f} mm, "
          f"max {tips[:, 2].max() * 1000:.1f} mm (band +/-{contact_tol * 1000:.0f} mm)")
    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results)
