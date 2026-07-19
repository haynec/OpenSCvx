"""UR5e pen tracing — drawing on a work surface with a tool-mounted pen via frax.

The arm counterpart of ``examples/drone/openscvx_logo.py``: instead of a
quadrotor aiming its boresight at a logo target in the air, a UR5e holds a
pen of length ``pen_length`` along the tool axis and draws the figure
directly — the pen tip must stay on the work surface (the plane z = 0) while
tracking a target that sweeps the drawing path.

Formulation:
    - Joint-space rigid-body dynamics from the UR5e URDF via frax — the one
      raw JAX function in the problem; everything else is symbolic.
    - The EE pose is symbolic Product-of-Exponentials forward kinematics:
      screw axes extracted from frax at the zero configuration, chained with
      ``ox.lie.SE3Exp``. The pen tip is ``T[:3, 3] + pen_length * T[:3, 2]``
      — a symbolic expression the tracking metric and contact constraints
      are written with directly.
    - The drawing path comes from an SVG file (the OpenSCvx wordmark by
      default; swap in any SVG), resampled by arc length, paced by curvature
      (corners get more time), eased in time, and baked into per-coordinate
      ``Cinterp`` cubic splines: inside the solver the moving target is a
      C2-smooth symbolic function of the time state.
    - The pen is a rigid stick of length ``pen_length`` along the tool axis
      (EE local +z). ``pen_length`` is an ``ox.Parameter``: update its
      ``value`` and re-solve to fit whatever pen ends up in the gripper — no
      rebuild.
    - Pen-up handling: SVG paths may contain implicit pen-up moves between
      disconnected strokes. During those transits the target lifts off the
      surface on a smooth ``lift_height`` bump; the arm follows it up,
      across, and back down, and no ink is drawn.
    - Contact: the tip never goes below the surface, and a CTCS band keeps
      it within ``contact_tol`` while drawing — the band's ceiling follows
      the target's lift profile, so it releases exactly (and as smoothly) as
      the target leaves the surface during transits. A tilt cone keeps the
      pen within ``tilt_max`` of the surface normal.
    - Tracking: a ``tip_error`` state integrates the squared distance
      between the pen tip and the moving target — including the vertical
      component, so the tip presses toward the surface instead of wandering
      inside the contact band. Its upper bound caps the average tracking
      error through CTCS, and its final value is minimized.
    - Initial guess: sequential damped-least-squares IK through frax's own
      FK places the tip exactly on the moving target with the pen vertical,
      so the guess writes the figure perfectly and SCP only has to trade
      tracking against the dynamics.

Requires:
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
from openscvx.integrations import frax_dynamics

# =============================================================================
# Tuning knobs
# =============================================================================

# Drawing: which SVG to trace and where it sits on the work surface (z = 0).
trace_svg = os.path.join(
    grandparent_dir, "examples", "drone", "logo_utils", "openscvx_logo_single.svg"
)
path_indices = [0]  # SVG paths to trace (None = all; the wordmark outline is path 0)
rotation_deg = 90.0  # lay the figure across the table, long axis on y
path_center = np.array([0.45, 0.0])  # centre of the drawing on the surface [m]
path_size = 0.50  # larger bounding-box dimension of the drawing [m]

# Mission: trajectory nodes and the (fixed) duration of the single traversal.
n = 300
total_time = 45.0  # [s] — the default wordmark is ~3.3 m of drawing

# Pen. The length is an ox.Parameter: assign ``pen_length.value`` and
# re-solve to change pens without rebuilding the problem.
pen_length = ox.Parameter("pen_length", shape=(1,), value=np.array([0.15]))
contact_tol = 0.0005  # tip-to-surface contact band while drawing [m]
tilt_max = np.deg2rad(30.0)  # max pen tilt from the surface normal
lift_height = 0.03  # pen-up clearance over transits between strokes [m]

# =============================================================================
# Robot and dynamics
# =============================================================================

robot = frax.Manipulator(os.path.join(current_dir, "ur5e_assets", "ur5e.urdf"))
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls
n_j = robot.num_joints  # 6

# =============================================================================
# Symbolic forward kinematics (Product of Exponentials)
# =============================================================================
# Screw axes come from frax at the zero configuration
# (dT/dq_i |_{q=0} = hat(xi_i) T_home); the EE pose is the symbolic chain
# T = exp(xi_1 q_1) ... exp(xi_6 q_6) T_home.

_T_home = np.asarray(robot.ee_transform(jnp.zeros(n_j)))
_dT0 = np.asarray(jax.jacfwd(robot.ee_transform)(jnp.zeros(n_j)))
_screws = []
for _i in range(n_j):
    _xi_hat = _dT0[:, :, _i] @ np.linalg.inv(_T_home)
    _screws.append(
        np.array([*_xi_hat[:3, 3], _xi_hat[2, 1], _xi_hat[0, 2], _xi_hat[1, 0]])
    )

T_ee = ox.lie.SE3Exp(ox.Constant(_screws[0]) * q[0])
for _i in range(1, n_j):
    T_ee = T_ee @ ox.lie.SE3Exp(ox.Constant(_screws[_i]) * q[_i])
T_ee = T_ee @ ox.Constant(_T_home)

pen_tip = T_ee[:3, 3] + pen_length[0] * T_ee[:3, 2]  # tip of the pen, world frame

# =============================================================================
# Time
# =============================================================================

time = ox.Time(initial=0.0, final=total_time, min=0.0, max=total_time)

# =============================================================================
# Drawing path: SVG -> arc-length polyline -> eased time table -> Cinterp
# =============================================================================
# The work surface is the plane z = 0 (the robot is mounted on it). The SVG
# outline is scaled uniformly into a ``path_size`` box centred on
# ``path_center`` in front of the base.

def svg_polyline(svg_file: str, n_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-arc-length polyline through the SVG's strokes, fitted to the drawing box.

    Samples the selected ``path_indices`` in order and resamples by cumulative
    arc length; the straight bridges across implicit pen-up moves are flagged
    as transits. The result is y-flipped (SVG y points down), rotated by
    ``rotation_deg``, scaled to fit ``path_size``, and centred on
    ``path_center``.

    Returns:
        points: (n_points, 2) polyline in table coordinates.
        on_stroke: (n_points,) bool mask, False on pen-up transit samples.
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

    step = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    s_uniform = np.linspace(0.0, arc[-1], n_points)
    pts = np.column_stack([np.interp(s_uniform, arc, dense[:, i]) for i in range(2)])

    # Samples that fall inside an implicit pen-up move are transits.
    on_stroke = np.ones(n_points, dtype=bool)
    for i in np.nonzero(step > 10.0 * np.median(step))[0]:
        on_stroke &= ~((s_uniform > arc[i]) & (s_uniform < arc[i + 1]))

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pts = (pts - 0.5 * (lo + hi)) * (path_size / (hi - lo).max())
    return pts + path_center, on_stroke


def _is_closed(paths) -> bool:
    """True when the SVG traces a closed figure (end point returns to start)."""
    start, end = paths[0].start, paths[-1].end
    return abs(end - start) < 1e-6 * sum(p.length() for p in paths)


def path_progress(t_frac, ramp: float = 0.08):
    """Eased time progress: smoothstep speed ramps at both ends of the trace.

    The arm starts and ends at rest, so the target speed ramps from and to
    zero over the first and last ``ramp`` fraction of the mission and stays
    near-constant in between, where curvature pacing governs the speed.
    """
    up = np.clip(t_frac / ramp, 0.0, 1.0)
    down = np.clip((1.0 - t_frac) / ramp, 0.0, 1.0)
    speed = (3 * up**2 - 2 * up**3) * (3 * down**2 - 2 * down**3)
    progress = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    return progress / progress[-1]


def curvature_pacing(polyline: np.ndarray, kappa_ref: float = 50.0, w_max: float = 8.0):
    """Map time fraction -> path progress, slowing the target through corners.

    Allocates trace time in proportion to the curvature weight ``1 + |kappa|
    / kappa_ref`` (clipped at ``w_max``), so sharp corners get up to
    ``w_max`` times more time per unit arc length than straight strokes.

    Returns:
        (time_fraction, progress): monotone tables mapping the fraction of
        trace time spent to the fraction of arc length covered.
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


# Target as a function of mission time: eased, curvature-paced progress along
# the polyline, lifted on a smooth ``lift_height`` bump during pen-up
# transits. Cubic splines through the table give a C2-smooth moving target.
_polyline, _on_stroke = svg_polyline(trace_svg)
_time_frac, _s_polyline = curvature_pacing(_polyline)
_t_table = np.linspace(0.0, total_time, 1000)
_s_table = np.interp(path_progress(_t_table / total_time), _time_frac, _s_polyline)
_stroke_table = np.interp(_s_table, _s_polyline, _on_stroke.astype(float)) > 0.5

_z_table = np.zeros(len(_t_table))
_transit_edges = np.flatnonzero(np.diff(np.concatenate([[1.0], _stroke_table, [1.0]])))
for _i0, _i1 in _transit_edges.reshape(-1, 2):
    _z_table[_i0:_i1] = lift_height * np.sin(np.pi * np.linspace(0, 1, _i1 - _i0)) ** 2

_xyz_table = np.column_stack(
    [np.interp(_s_table, _s_polyline, _polyline[:, i]) for i in range(2)] + [_z_table]
)

# Numpy-side spline (initial guess, post-processing, plots) and its symbolic
# twin (Cinterp, same not-a-knot cubic through the same table) for the solver.
_target_spline = CubicSpline(_t_table, _xyz_table)
_target_exprs = [ox.Cinterp(time[0], _t_table, _xyz_table[:, i]) for i in range(3)]


def target_at_time(t) -> np.ndarray:
    """Moving pen target at mission time t (lifted during pen-up transits)."""
    return np.asarray(_target_spline(np.clip(t, 0.0, total_time)))


def target_engaged(t) -> np.ndarray:
    """True where the target is drawing (pen down), False on transits."""
    return np.interp(np.clip(t, 0.0, total_time), _t_table, _stroke_table.astype(float)) > 0.5


# =============================================================================
# Initial guess: IK writing of the path with the pen vertical
# =============================================================================
# Damped-least-squares IK through frax FK, solved sequentially (each node
# seeded by the previous solution) so the trajectory stays on one IK branch.

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
# Accumulates || tip - target ||^2 over the mission [m^2 s]. The vertical
# component is included so the tip presses toward the surface rather than
# wandering inside the contact band.

tip_error = ox.State("tip_error", shape=(1,))
tip_error.min = np.array([0.0])
# Loose guard — Minimize supplies the pressure; a cap the true integral
# cannot meet forces the solver into defect-cheating instead of steering.
tip_error.max = np.array([1e-2])
tip_error.initial = np.array([0.0])
tip_error.final = [ox.Minimize(0.0)]
tip_error.guess = np.zeros((n, 1))

target_x, target_y, target_z = _target_exprs

# =============================================================================
# Constraints and Problem
# =============================================================================

states = [*dyn.states, tip_error]

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

# Contact: "never below the surface" is always active; the ceiling follows
# the target's lift profile, releasing the band during pen-up transits. The
# pair shares augmented state idx 1, whose licq bound sets the allowed
# violation per segment (v ~= sqrt(licq_max / dt_seg), here ~26 um).
constraints.extend(
    [
        ox.ctcs(pen_tip[2] >= -contact_tol, idx=1, licq_max=1e-10, check_nodally=True),
        ox.ctcs(
            pen_tip[2] <= contact_tol + 2.0 * target_z,
            idx=1,
            licq_max=1e-10,
            check_nodally=True,
        ),
        # Tilt cone: tool-axis z is -1 with the pen straight down.
        ox.ctcs(T_ee[2, 2] <= -float(np.cos(tilt_max))),
    ]
)

# The frax forward dynamics is the one non-symbolic function in the problem.
dynamics = {
    "q": qd,
    "tip_error": (pen_tip[0] - target_x) ** 2
    + (pen_tip[1] - target_y) ** 2
    + (pen_tip[2] - target_z) ** 2,
}
byof: ox.ByofSpec = {"dynamics": {"qd": frax_dynamics(robot, q=q, qd=qd, tau=tau)}}

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    byof=byof,
    N=n,
    algorithm={
        "lam_cost": 1e1,  # the default 1e-2 under-prices the O(1e-5) terminal metric
        "lam_vb": 1e0,
        "lam_vc": 1e4,  # must price defects above the tracking reward
        "lam_prox": 1e1,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
    # The default atol of 1e-3 would swallow the O(1e-6..1e-3) tip_error integrand.
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-8, "rtol": 1e-8}),
)

# =============================================================================
# Post-processing: pen tip path
# =============================================================================


def pen_tip_path(q_traj: np.ndarray) -> np.ndarray:
    """World-frame pen tip positions along a joint trajectory (batched FK)."""
    T = np.asarray(jax.vmap(robot.ee_transform)(jnp.asarray(q_traj)))
    return T[:, :3, 3] + float(pen_length.value[0]) * T[:, :3, 2]


# =============================================================================
# Visualization
# =============================================================================


def trace_figure(tips: np.ndarray, t: np.ndarray):
    """Plotly figure: the SVG target path against the drawn line, coloured by speed."""
    import plotly.graph_objects as go

    speed = np.linalg.norm(np.gradient(tips[:, :2], t, axis=0), axis=1)
    engaged = target_engaged(t)
    tips, speed = tips[engaged], speed[engaged]

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
            x=tips[:, 0], y=tips[:, 1],
            mode="markers", name="pen trace",
            marker=dict(
                color=speed, colorscale="Rainbow", size=4,
                colorbar=dict(title="tip speed [m/s]"), showscale=True,
            ),
        )
    )
    fig.update_layout(
        title=f"UR5e pen trace — {os.path.basename(trace_svg)}",
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

    Mesh geoms, vertices, colours, and poses all come from the MuJoCo model.
    Returns None when the menagerie assets are unavailable (see
    openscvx.integrations.menagerie); the caller falls back to a stick model.
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
    """Animate in Viser: CAD-mesh arm (menagerie), pen, and the line it draws."""
    from openscvx.plotting.viser import add_animation_controls, create_server

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])
    prop = results.multishot_propagation()
    if prop is not None:
        q_traj, t_vec = prop.state("q")

    # Subsample to ~60 fps; full multishot resolution only slows the client.
    stride = max(1, len(q_traj) // 3000)
    q_traj = np.asarray(q_traj)[::stride]
    t_vec = np.asarray(t_vec).flatten()[::stride]

    n_frames = len(q_traj)
    tips = pen_tip_path(q_traj)
    target_path = np.column_stack([_polyline, np.zeros(len(_polyline))])

    links = np.asarray(jax.vmap(robot.link_to_world_transforms)(jnp.asarray(q_traj)))
    ee_T = np.asarray(jax.vmap(robot.ee_transform)(jnp.asarray(q_traj)))
    keypoints = np.zeros((n_frames, n_j + 2, 3))
    keypoints[:, 1 : 1 + n_j] = links[:, :, :3, 3]
    keypoints[:, -1] = ee_T[:, :3, 3]

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

    pen_handle = server.scene.add_line_segments(
        "/pen",
        points=np.array([[keypoints[0, -1], tips[0]]], dtype=np.float32),
        colors=np.full((1, 2, 3), (40, 40, 40), dtype=np.uint8),
        line_width=4.0,
    )
    # Animate the ink inside a fixed-shape buffer (viser line-segment buffers
    # must not be resized); undrawn and pen-up segments collapse to a point.
    ink_segs = np.stack([tips[:-1], tips[1:]], axis=1).astype(np.float32)
    engaged = target_engaged(t_vec)
    ink_segs[~(engaged[:-1] & engaged[1:])] = ink_segs[0, 0]
    ink_handle = server.scene.add_line_segments(
        "/ink",
        points=np.broadcast_to(ink_segs[0, 0], ink_segs.shape).copy(),
        colors=np.full(ink_segs.shape, (200, 30, 30), dtype=np.uint8),
        line_width=3.0,
    )

    def update(frame: int) -> None:
        update_robot(frame)
        pen_handle.points = np.array([[keypoints[frame, -1], tips[frame]]], dtype=np.float32)
        drawn = ink_segs.copy()
        drawn[frame:] = drawn[frame, 0] if frame < len(drawn) else drawn[-1, 1]
        ink_handle.points = drawn

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
          f"contact band: +/-{contact_tol * 1000:.1f} mm")
    print(f"Path: {os.path.basename(trace_svg)}, {path_size} m wide at {list(path_center)}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    # Measure on the multishot propagation — an open-loop torque replay of an
    # arm diverges, so results.trajectory is not meaningful here.
    prop = results.multishot_propagation()
    q_traj, t_traj = prop.state("q")
    tips = pen_tip_path(q_traj)
    targets = np.asarray(_target_spline(np.clip(t_traj, 0.0, total_time)))
    engaged = target_engaged(t_traj)
    tip_err = np.linalg.norm(tips[:, :2] - targets[:, :2], axis=1)[engaged]
    tips_down = tips[engaged]

    print("\nResults:")
    print(f"  Strokes: {len(_transit_edges) // 2 + 1}  |  pen-up transits: "
          f"{len(_transit_edges) // 2}")
    print(f"  Tip tracking error (pen down): mean {tip_err.mean() * 1000:.1f} mm, "
          f"max {tip_err.max() * 1000:.1f} mm")
    print(f"  Tip height (pen down): min {tips_down[:, 2].min() * 1000:.1f} mm, "
          f"max {tips_down[:, 2].max() * 1000:.1f} mm (band +/-{contact_tol * 1000:.1f} mm)")
    trace_figure(tips, t_traj).show()

    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results)
