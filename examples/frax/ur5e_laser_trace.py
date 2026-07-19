"""UR5e laser tracing — boresight path following on a work surface via frax dynamics.

The arm counterpart of ``examples/drone/openscvx_logo.py``: instead of a
quadrotor aiming its boresight at a moving logo target in the air, a UR5e
points a tool-mounted laser at a target moving along a drawing path on the
work surface the robot is mounted on (the plane z = 0).

Formulation
-----------
- ``FraxDynamics`` — full rigid-body joint-space dynamics from the UR5e URDF.
- The drawing path comes from an SVG file (the OpenSCvx wordmark by default;
  swap in any SVG). The path is resampled by arc length, paced by curvature
  (corners get more time), eased in time so the target starts and ends at
  rest, and baked into per-coordinate ``Cinterp`` cubic splines, so inside
  the solver the moving target is a C2-smooth symbolic function of the time
  state.
- A ``misalignment`` state integrates ``1 - cos(theta)``, where ``theta`` is
  the angle between the tool axis (EE local +z, the laser) and the line of
  sight to the target (BYOF dynamics, since it needs frax FK). Its state
  upper bound caps the average pointing error through CTCS, and its final
  value is minimized.
- The initial guess is the "clever" part: sequential damped-least-squares IK
  through frax's own FK tracks the path at a standoff height with the tool
  axis normal to the surface, so the guess already points at the target and
  SCP only has to trade pointing accuracy against the dynamics. The same
  initialization works for any drawing path.

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
# Discretization and time
# =============================================================================

n = 300
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
    """Moving laser target on the work surface at mission time t."""
    x_t, y_t = _target_spline(np.clip(t, 0.0, total_time))
    return np.array([x_t, y_t, 0.0])


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
        p_des = jnp.array(target_at_time(t_k) + np.array([0.0, 0.0, standoff]))
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
# Loose guard only — Minimize provides the tracking pressure. A tight cap
# that the true integral cannot meet forces the solver into defect-cheating
# (it "erases" the metric through virtual control instead of steering).
misalignment.max = np.array([1.0])
misalignment.initial = np.array([0.0])
misalignment.final = [ox.Minimize(0.0)]
misalignment.guess = np.zeros((n, 1))

# The Cinterp target lowers to JAX callables with the BYOF signature. Lowering
# resolves state slices, which exist only after Problem preprocessing — so
# lower lazily on the first call.
_lowered_target: list = []


def _target_fn(x, u, node, params):
    """Symbolic Cinterp target evaluated inside BYOF: (x, u, node, params) -> (3,)."""
    if not _lowered_target:
        _lowered_target.extend(lower_to_jax(_target_exprs))
    fx, fy = _lowered_target
    return jnp.array([fx(x, u, node, params), fy(x, u, node, params), 0.0])


def _misalignment_dynamics(x, u, node, params):
    T = robot.ee_transform(x[q.slice])
    p_ee = jnp.asarray(T)[:3, 3]
    laser = jnp.asarray(T)[:3, 2]
    to_target = _target_fn(x, u, node, params) - p_ee
    return jnp.array([1.0 - jnp.dot(to_target, laser) / jnp.linalg.norm(to_target)])


def laser_angle_to_target(q_val, t_val):
    """Pointing angle in radians (for reporting; not used in the optimization)."""
    T = np.asarray(robot.ee_transform(q_val))
    to_target = target_at_time(t_val) - T[:3, 3]
    cos_angle = np.dot(to_target, T[:3, 2]) / np.linalg.norm(to_target)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


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


def trace_figure(traced: np.ndarray, t: np.ndarray):
    """Plotly figure comparing the drawn trace against the SVG target path.

    Same visual grammar as the race-car examples: the reference as a dashed
    black line, the executed path as markers coloured by speed.
    """
    import plotly.graph_objects as go

    ok = np.isfinite(traced).all(axis=1)
    traced, t = traced[ok], t[ok]
    speed = np.linalg.norm(np.gradient(traced[:, :2], t, axis=0), axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_polyline[:, 0], y=_polyline[:, 1],
            mode="lines", name="target path",
            line=dict(color="black", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=traced[:, 0], y=traced[:, 1],
            mode="markers", name="laser trace",
            marker=dict(
                color=speed, colorscale="Rainbow", size=4,
                colorbar=dict(title="dot speed [m/s]"), showscale=True,
            ),
        )
    )
    fig.update_layout(
        title=f"UR5e laser trace — {os.path.basename(trace_svg)}",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    return fig


# The menagerie MJCF mounts the UR5e base rotated 180 degrees about z
# relative to the URDF frame frax uses; every mesh pose is premultiplied by
# this to put the CAD model in the trajectory's world frame.
_RZ180 = np.diag([-1.0, -1.0, 1.0])


def _ur5e_mesh_scene(server, q_traj: np.ndarray):
    """Add the menagerie UR5e CAD meshes and return a per-frame pose updater.

    Everything is read from the MuJoCo model itself — visual mesh geoms,
    vertices/faces, and material colours — so no asset bookkeeping is needed.
    Returns None when the menagerie assets are unavailable (the caller falls
    back to the stick model); see openscvx.integrations.menagerie for how the
    asset directory is located.
    """
    try:
        import mujoco
        from openscvx.integrations.menagerie import get_model_dir

        xml = get_model_dir("universal_robots_ur5e") / "ur5e.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
    except Exception as exc:
        print(
            f"[viser] UR5e CAD meshes unavailable ({type(exc).__name__}: {exc}); "
            "falling back to line segments."
        )
        return None
    from scipy.spatial.transform import Rotation

    data = mujoco.MjData(model)
    geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

    n_frames = len(q_traj)
    pos = np.zeros((n_frames, len(geoms), 3))
    rot = np.zeros((n_frames, len(geoms), 3, 3))
    for k in range(n_frames):
        data.qpos[:6] = q_traj[k]
        mujoco.mj_kinematics(model, data)
        for j, g in enumerate(geoms):
            pos[k, j] = _RZ180 @ data.geom_xpos[g]
            rot[k, j] = _RZ180 @ data.geom_xmat[g].reshape(3, 3)
    quat_xyzw = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(n_frames, -1, 4)
    wxyz = quat_xyzw[..., [3, 0, 1, 2]]

    handles = []
    for j, g in enumerate(geoms):
        mesh_id = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        rgba = model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else [0.8] * 4
        handles.append(
            server.scene.add_mesh_simple(
                f"/ur5e/geom_{j}",
                vertices=model.mesh_vert[v0 : v0 + nv].astype(np.float32),
                faces=model.mesh_face[f0 : f0 + nf].astype(np.uint32),
                color=tuple(int(255 * c) for c in rgba[:3]),
                position=pos[0, j],
                wxyz=wxyz[0, j],
            )
        )
    print(f"[viser] Loaded {len(handles)} UR5e CAD mesh geoms from menagerie.")

    def update(frame: int) -> None:
        for j, handle in enumerate(handles):
            handle.position = pos[frame, j]
            handle.wxyz = wxyz[frame, j]

    return update


def visualize(results) -> None:
    """Animate in Viser: CAD-mesh arm (menagerie), laser ray, and the drawn trace."""
    from openscvx.plotting.viser import add_animation_controls, create_server

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])
    prop = results.multishot_propagation()
    if prop is not None:
        q_traj, t_vec = prop.state("q")

    n_frames = len(q_traj)
    trace = laser_trace(q_traj)
    target_path = np.column_stack([_polyline, np.zeros(len(_polyline))])

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

    update_robot = _ur5e_mesh_scene(server, q_traj)
    if update_robot is None:
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

        def update_robot(frame: int) -> None:
            arm_handle.points = _arm_segments(frame)

    laser_handle = server.scene.add_line_segments(
        "/laser",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.full((1, 2, 3), (255, 40, 40), dtype=np.uint8),
        line_width=3.0,
    )
    # The line drawn so far. Viser keeps line-segment points and colors as
    # separate fixed-shape buffers, so animate inside a full-size buffer with
    # the not-yet-drawn segments collapsed to a point (zero-length segments
    # are invisible) instead of resizing per frame. Segments where the laser
    # misses the surface are collapsed permanently.
    trace_segs = np.stack([trace[:-1], trace[1:]], axis=1).astype(np.float32)
    finite_hits = trace[np.isfinite(trace).all(axis=1)]
    fill = (finite_hits[0] if len(finite_hits) else np.zeros(3)).astype(np.float32)
    trace_segs[~np.isfinite(trace_segs).all(axis=(1, 2))] = fill
    trace_handle = server.scene.add_line_segments(
        "/laser_trace",
        points=np.broadcast_to(fill, trace_segs.shape).copy(),
        colors=np.full(trace_segs.shape, (255, 40, 40), dtype=np.uint8),
        line_width=3.0,
    )

    def update(frame: int) -> None:
        update_robot(frame)
        p_ee = keypoints[frame, -1]
        hit = trace[frame] if np.isfinite(trace[frame]).all() else p_ee
        laser_handle.points = np.array([[p_ee, hit]], dtype=np.float32)
        drawn = trace_segs.copy()
        drawn[frame:] = fill
        trace_handle.points = drawn

    add_animation_controls(server, t_vec, [update], loop=True)
    server.sleep_forever()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("UR5e laser tracing via frax dynamics")
    print("=" * 60)
    print(f"Nodes: {n}  |  Mission time: {total_time} s")
    print(f"Path: {os.path.basename(trace_svg)}, {path_size} m wide at {list(path_center)}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    # Measure on the multishot propagation: a single 10 s open-loop torque
    # replay of an arm diverges, so results.trajectory is not meaningful here.
    prop = results.multishot_propagation()
    q_traj, t_traj = prop.state("q")
    trace = laser_trace(q_traj)
    targets = np.array([target_at_time(t) for t in t_traj])
    trace_err = np.linalg.norm(trace[:, :2] - targets[:, :2], axis=1)
    angles = np.array([laser_angle_to_target(qk, tk) for qk, tk in zip(q_traj, t_traj)])

    print("\nResults:")
    print(f"  Pointing angle: mean {np.degrees(angles.mean()):.2f} deg, "
          f"max {np.degrees(angles.max()):.2f} deg")
    print(f"  Laser dot error on surface: mean {np.nanmean(trace_err) * 1000:.1f} mm, "
          f"max {np.nanmax(trace_err) * 1000:.1f} mm")
    trace_figure(trace, t_traj).show()

    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results)
