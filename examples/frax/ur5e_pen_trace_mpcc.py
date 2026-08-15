"""MPCC UR5e pen tracing: receding-horizon contouring control of a drawing arm.

The MPCC counterpart of ``examples/frax/ur5e_pen_trace.py``, following the
same one-shot-to-MPCC move as ``examples/drone/boresight_trace_mpcc.py``:

- The one-shot version parametrizes the drawing target by *time*, which
  couples the node count to the path complexity (300 nodes for the wordmark)
  and requires hand-tuned curvature pacing so corners get enough time.
- Here the target is parametrized by *arc length*: a progress state advances
  along the path as fast as drawing accuracy allows, driven by a trace-rate
  control, and a short receding horizon retraces the figure chunk by chunk.
  Corner slowdown emerges from the contouring trade-off instead of being
  scheduled up front.

The reference tables map planar arc length to a 3D target point: the SVG
strokes lie on the work surface (z = 0) and the implicit pen-up transits
between strokes ride a smooth ``lift_height`` bump baked into the z table.
The contouring error — the symbolic pen tip (Product-of-Exponentials forward
kinematics through the pen) minus the ``Cinterp`` reference at the current
progress — is fully symbolic; the frax joint-space dynamics is the one
bring-your-own-function piece, exactly as in the one-shot example.

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

# Pen and contact.
pen_length = ox.Parameter("pen_length", shape=(1,), value=np.array([0.15]))
contact_tol = 0.001  # how far the tip may dip below the surface [m]
tilt_max = np.deg2rad(30.0)  # max pen tilt from the surface normal
lift_height = 0.03  # pen-up clearance over transits between strokes [m]

# MPC horizon. The executed step is the accuracy knob: the arm's joint
# dynamics need this resolution for the pen to settle onto the line — with
# coarser steps the tip error plateaus several millimetres high however the
# weights are tuned, the pen bouncing vertically through the surface.
n_mpc = 21  # Horizon nodes
horizon_duration = 1.0  # Horizon length [s]
dt_mpc = horizon_duration / (n_mpc - 1)  # Executed step [s]

Q_TRACE = 1e1  # Contouring error weight (squared tip-vs-path distance, x1e3 scaled)
Q_PROGRESS = 2e0  # Progress reward
Q_SMOOTH = 1e-1  # Joint-velocity damping: calms the arm between steps

# The raw squared tip error is O(1e-6) [m^2] at mm accuracy — below any
# integrator tolerance a receding-horizon loop can afford. The trace_error
# state integrates the error in mm^2-like units so the integrator sees it.
TRACE_SCALE = 1e3

trace_rate_max = 0.3  # Max speed of the traced point along the path [m/s]
trace_rate_guess = 0.1

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

_T_home = np.asarray(robot.ee_transform(jnp.zeros(n_j)))
_dT0 = np.asarray(jax.jacfwd(robot.ee_transform)(jnp.zeros(n_j)))
_screws = []
for _i in range(n_j):
    _xi_hat = _dT0[:, :, _i] @ np.linalg.inv(_T_home)
    _screws.append(np.array([*_xi_hat[:3, 3], _xi_hat[2, 1], _xi_hat[0, 2], _xi_hat[1, 0]]))

T_ee = ox.lie.SE3Exp(ox.Constant(_screws[0]) * q[0])
for _i in range(1, n_j):
    T_ee = T_ee @ ox.lie.SE3Exp(ox.Constant(_screws[_i]) * q[_i])
T_ee = T_ee @ ox.Constant(_T_home)

pen_tip = T_ee[:3, 3] + pen_length[0] * T_ee[:3, 2]  # tip of the pen, world frame

# =============================================================================
# Reference path: SVG -> arc-length-parametrized 3D target on the surface
# =============================================================================


def svg_polyline(svg_file: str, n_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-arc-length polyline through the SVG's strokes, fitted to the drawing box.

    Same construction as the one-shot example: samples the selected
    ``path_indices`` in order, resamples by cumulative arc length, flags the
    straight bridges across implicit pen-up moves as transits, then rotates,
    scales, and centres the figure on the work surface.

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
    start, end = paths[0].start, paths[-1].end
    closed = abs(end - start) < 1e-6 * sum(p.length() for p in paths)
    dense = np.asarray(dense + [dense[0]] if closed else dense)
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


_polyline, _on_stroke = svg_polyline(trace_svg)

# Arc length of the scaled polyline is the progress coordinate.
_step = np.linalg.norm(np.diff(_polyline, axis=0), axis=1)
s_path = np.concatenate([[0.0], np.cumsum(_step)])
total_arc_length = s_path[-1]

# Pen-up transits ride a smooth lift bump baked into the z table.
_z_path = np.zeros(len(s_path))
_transit_edges = np.flatnonzero(np.diff(np.concatenate([[1.0], _on_stroke, [1.0]])))
for _i0, _i1 in _transit_edges.reshape(-1, 2):
    _z_path[_i0:_i1] = lift_height * np.sin(np.pi * np.linspace(0, 1, _i1 - _i0)) ** 2

# Extend the reference past the true end by more than one horizon of pen
# travel, parked at the final point: the progress reward keeps pulling, the
# reference point stays put, and the pen decelerates into the endpoint
# instead of chasing a fictitious continuation (see the drone MPCC example).
pad_length = 2.0 * trace_rate_max * horizon_duration
_ds = s_path[-1] / (len(s_path) - 1)
_s_pad = s_path[-1] + np.arange(1, int(pad_length / _ds) + 1) * _ds
s_ref = np.concatenate([s_path, _s_pad])
xyz_path = np.column_stack([_polyline, _z_path])
ref_points = np.concatenate([xyz_path, np.tile(xyz_path[-1], (len(_s_pad), 1))])
padded_arc_length = s_ref[-1]

_ref_exprs = None  # built lazily below once `progress` exists


def ref_point(s):
    """Numeric reference lookup used for guesses and analysis (matches Cinterp data)."""
    s = np.clip(s, 0.0, padded_arc_length)
    return np.stack([np.interp(s, s_ref, ref_points[:, i]) for i in range(3)], axis=-1)


def ref_engaged(s) -> np.ndarray:
    """True where the reference is drawing (pen down), False on transits."""
    return np.interp(np.clip(s, 0.0, total_arc_length), s_path, _on_stroke.astype(float)) > 0.5


# =============================================================================
# MPCC states and controls
# =============================================================================

progress = ox.State("progress", shape=(1,))  # Arc length along the drawing
progress.min = np.array([0.0])
progress.max = np.array([padded_arc_length])
progress.initial = np.array([0.0])
progress.final = [ox.Maximize(0.0)]

trace_error = ox.State("trace_error", shape=(1,))  # Integrated squared tip error (scaled)
trace_error.min = np.array([0.0])
trace_error.max = np.array([1.0])
trace_error.initial = np.array([0.0])
trace_error.final = [ox.Minimize(0.0)]

smoothness = ox.State("smoothness", shape=(1,))  # Integrated squared joint velocity
smoothness.min = np.array([0.0])
smoothness.max = np.array([100.0])
smoothness.initial = np.array([0.0])
smoothness.final = [ox.Minimize(0.0)]

trace_rate = ox.Control("trace_rate", shape=(1,))  # d(progress)/dt
trace_rate.min = np.array([0.0])  # The pen only moves forward
trace_rate.max = np.array([trace_rate_max])

# Contouring error: symbolic pen tip vs the spline reference at the current
# progress. The z component presses the tip toward the surface while drawing
# and lifts it over the transit bumps.
ref_x = ox.Cinterp(progress[0], s_ref, ref_points[:, 0])
ref_y = ox.Cinterp(progress[0], s_ref, ref_points[:, 1])
ref_z = ox.Cinterp(progress[0], s_ref, ref_points[:, 2])
trace_cost = TRACE_SCALE * (
    (pen_tip[0] - ref_x) ** 2 + (pen_tip[1] - ref_y) ** 2 + (pen_tip[2] - ref_z) ** 2
)

# =============================================================================
# Constraints and Problem
# =============================================================================

states = [*dyn.states, progress, trace_error, smoothness]
controls = [*dyn.controls, trace_rate]

constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Contact: the pen never stabs below the surface. Unlike the one-shot
# example, no contact *band* is needed: the contouring cost includes the
# vertical error, so tracking itself presses the tip onto the surface while
# drawing and lifts it over the transits.
constraints.extend(
    [
        ox.ctcs(pen_tip[2] >= -contact_tol),
        # Tilt cone: tool-axis z is -1 with the pen straight down.
        ox.ctcs(T_ee[2, 2] <= -float(np.cos(tilt_max))),
    ]
)

dynamics = {
    "q": qd,
    "progress": trace_rate,
    "trace_error": trace_cost,
    "smoothness": ox.Sum(qd * qd),
}
byof: ox.ByofSpec = {"dynamics": {"qd": frax_dynamics(robot, q=q, qd=qd, tau=tau)}}

# Time: fixed horizon with a pinned first step (the executed segment).
t = ox.Time(
    initial=0.0,
    final=horizon_duration,
    min=0.0,
    max=horizon_duration,
    time_dilation_min=0.5 * horizon_duration,
    time_dilation_max=2.0 * horizon_duration,
)
constraints.append((t == dt_mpc).convex().at(1))

# =============================================================================
# Initial guess: IK placement along the first path chunk
# =============================================================================

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


def ik_at_progress(s_vals: np.ndarray, iters: int = 50) -> np.ndarray:
    """Joint configurations whose pen tip sits on the reference at each progress."""
    ee_height = float(pen_length.value[0])
    q_traj = np.zeros((len(s_vals), n_j))
    q_val = jnp.array(_q_seed)
    for k, s_k in enumerate(s_vals):
        p_des = jnp.array(ref_point(s_k) + np.array([0.0, 0.0, ee_height]))
        for _ in range(iters):
            q_val = _ik_step(q_val, p_des)
        q_traj[k] = np.asarray(q_val)
    return q_traj


def set_initial_guess():
    """Seed the first horizon: IK along the first path chunk at the guess pace."""
    arc_guess = np.linspace(0.0, trace_rate_guess * horizon_duration, n_mpc)
    q_guess = ik_at_progress(arc_guess)
    q.guess = q_guess
    q.initial = q_guess[0]
    q.final = [("free", 0.0)] * n_j
    qd.guess = np.gradient(q_guess, np.linspace(0.0, horizon_duration, n_mpc), axis=0)
    qd.initial = np.zeros(n_j)
    qd.final = [("free", 0.0)] * n_j
    tau.guess = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q_guess])

    progress.guess = arc_guess.reshape(-1, 1)
    trace_error.guess = np.zeros((n_mpc, 1))
    smoothness.guess = np.zeros((n_mpc, 1))
    trace_rate.guess = np.full((n_mpc, 1), trace_rate_guess)


def update_initial_conditions(nodes: dict):
    """Set initial conditions from node 1 of the previous solution (simulate one step)."""
    for state in [q, qd, progress]:
        state.initial = np.clip(nodes[state.name][1], state.min, state.max)
    # Cost integrators restart from zero each horizon
    trace_error.initial = np.array([0.0])
    smoothness.initial = np.array([0.0])


def shift_guess(nodes: dict):
    """Shift the previous solution by one node; extend the tip with a rest prior.

    The freshly revealed tip node holds the last joint configuration at rest
    with gravity-compensating torque, progress extended at the last trace
    rate. Repeating the last node's momentum would extrapolate drift into the
    new node; a rest prior lets the optimizer discover the motion the path
    actually needs (see the drone MPCC example).
    """
    ext_progress = min(
        nodes["progress"][-1, 0] + nodes["trace_rate"][-1, 0] * dt_mpc, padded_arc_length
    )
    q_tip = nodes["q"][-1]

    q.guess = np.vstack([nodes["q"][1:], [q_tip]])
    qd.guess = np.vstack([nodes["qd"][1:], np.zeros((1, n_j))])
    tau.guess = np.vstack([nodes["tau"][1:], [np.asarray(robot.gravity_vector(q_tip))]])
    progress.guess = np.vstack([nodes["progress"][1:], [[ext_progress]]])
    trace_rate.guess = np.vstack([nodes["trace_rate"][1:], nodes["trace_rate"][-1:]])

    for integrator, key in [(trace_error, "trace_error"), (smoothness, "smoothness")]:
        offset = nodes[key][1]
        integrator.guess = np.maximum(
            np.vstack([nodes[key][1:] - offset, nodes[key][-1:] - offset]), 0.0
        )

    # Time: shift and renormalize so the horizon starts at t = 0
    ext_time = nodes["time"][-1, 0] + nodes["_time_dilation"][-1, 0] * (1.0 / (n_mpc - 1))
    shifted_time = np.vstack([nodes["time"][1:], [[ext_time]]])
    t.guess = shifted_time - shifted_time[0]
    t._time_dilation_control.guess = np.vstack(
        [nodes["_time_dilation"][1:], nodes["_time_dilation"][-1:]]
    )


set_initial_guess()

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=t,
    constraints=constraints,
    byof=byof,
    N=n_mpc,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": {"trace_error": Q_TRACE, "progress": Q_PROGRESS, "smoothness": Q_SMOOTH},
        "lam_prox": 3e1,  # Strong anchoring: weak prox never converges here
        "lam_vc": 1e4,  # Keep dynamics honest: far above the largest cost weight
        "k_max": 30,  # Warm-started horizons converge in a handful of iterations
        # The latency/accuracy dial: the default 1e-4 polishes each horizon to
        # sub-mm tracking at ~10x the iterations per step, while 1e-3 accepts
        # the first iterate that lands within the pen's line width; the extra
        # polish below that mostly buys vertical (contact) accuracy.
        "ep_tr": 5e-4,
    },
    float_dtype="float64",
    # Integrator tolerance must stay below the O(1e-3) scaled trace-error
    # integrand or it silently swallows the cost signal (see TRACE_SCALE).
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-6, "rtol": 1e-6}),
    solver={"solver_args": {"abstol": 3e-5, "reltol": 3e-7, "enforce_dpp": True}},
)
problem.settings.dev.printing = False

# =============================================================================
# Analysis
# =============================================================================


def pen_tip_path(q_traj: np.ndarray) -> np.ndarray:
    """World-frame pen tip positions along a joint trajectory (batched FK)."""
    T = np.asarray(jax.vmap(robot.ee_transform)(jnp.asarray(q_traj)))
    return T[:, :3, 3] + float(pen_length.value[0]) * T[:, :3, 2]


def trace_figure(tips: np.ndarray, progress_dense: np.ndarray):
    """Plotly figure: the SVG target path against the drawn line, coloured by tip error."""
    import plotly.graph_objects as go

    engaged = ref_engaged(progress_dense)
    err = np.linalg.norm(tips - ref_point(progress_dense), axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_polyline[_on_stroke, 0],
            y=_polyline[_on_stroke, 1],
            mode="markers",
            name="target path",
            marker=dict(color="black", size=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tips[engaged, 0],
            y=tips[engaged, 1],
            mode="markers",
            name="pen trace",
            marker=dict(
                color=err[engaged] * 1000,
                colorscale="Rainbow",
                size=4,
                colorbar=dict(title="tip error [mm]"),
                showscale=True,
            ),
        )
    )
    fig.update_layout(
        title=f"UR5e MPCC pen trace — {os.path.basename(trace_svg)}",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    return fig


def deviation_figure(progress_dense: np.ndarray, tips: np.ndarray):
    """Plotly figure: tip deviation against arc length along the path."""
    import plotly.graph_objects as go

    err = np.linalg.norm(tips - ref_point(progress_dense), axis=1)
    engaged = ref_engaged(progress_dense)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=progress_dense[engaged],
            y=err[engaged] * 1000,
            mode="lines",
            name="tip error",
            line=dict(color="black", width=1),
        )
    )
    fig.update_layout(
        title=f"Trace deviation — {os.path.basename(trace_svg)}",
        xaxis=dict(title="path length [m]"),
        yaxis=dict(title="tip error [mm]"),
        height=400,
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


def visualize(q_traj: np.ndarray, t_vec: np.ndarray, progress_traj: np.ndarray) -> None:
    """Animate the executed MPC trajectory in Viser: arm, pen, and the ink it draws."""
    from openscvx.plotting.viser import add_animation_controls, create_server

    # Subsample to ~60 fps; full dense resolution only slows the client.
    stride = max(1, len(q_traj) // 3000)
    q_traj = np.asarray(q_traj)[::stride]
    t_vec = np.asarray(t_vec)[::stride]
    progress_traj = np.asarray(progress_traj)[::stride]

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
    engaged = ref_engaged(progress_traj)
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
# Main: receding-horizon loop
# =============================================================================

if __name__ == "__main__":
    show_viz = "--no-viz" not in sys.argv

    print("UR5e pen tracing via MPCC")
    print("=" * 60)
    print(
        f"Path: {os.path.basename(trace_svg)}, {total_arc_length:.2f} m of stroke "
        f"at {path_size:.2f} m wide  |  Horizon: {n_mpc} nodes / {horizon_duration} s"
    )
    print()

    problem.initialize()

    max_steps = 2000

    seg_q = []
    seg_progress = []
    seg_times = []
    solve_times = []

    for step in range(max_steps):
        problem.reset()
        results = problem.solve()
        results = problem.post_process()
        nodes = results.nodes
        solve_times.append(problem.timing_solve)

        # The executed segment is the dense trajectory from node 0 to node 1
        traj_time = results.trajectory["time"].flatten()
        seg_end = np.searchsorted(traj_time, dt_mpc, side="right")
        seg_q.append(results.trajectory["q"][:seg_end].copy())
        seg_progress.append(results.trajectory["progress"][:seg_end, 0].copy())
        seg_times.append(traj_time[:seg_end] + step * dt_mpc)

        cur_progress = nodes["progress"][0, 0]
        print(
            f"step {step:3d}: progress={cur_progress:7.3f} m "
            f"({100 * cur_progress / total_arc_length:5.1f}%), "
            f"trace_cost={nodes['trace_error'][-1, 0]:.6f}"
        )

        # The pen has crossed the true end of the drawing
        if nodes["progress"][1, 0] >= total_arc_length:
            break

        update_initial_conditions(nodes)
        shift_guess(nodes)

    # --- Trace accuracy report ---
    q_dense = np.concatenate(seg_q)
    progress_dense = np.concatenate(seg_progress)
    times = np.concatenate(seg_times)

    tips = pen_tip_path(q_dense)
    targets = ref_point(progress_dense)
    engaged = ref_engaged(progress_dense)
    tip_err = np.linalg.norm(tips[:, :2] - targets[:, :2], axis=1)[engaged]
    tips_down = tips[engaged]

    print(
        f"\nTraced {progress_dense[-1]:.2f} / {total_arc_length:.2f} m "
        f"in {times[-1]:.1f} s with {n_mpc}-node horizons"
    )
    print(
        f"Tip tracking error (pen down): mean {tip_err.mean() * 1000:.2f} mm, "
        f"max {tip_err.max() * 1000:.2f} mm"
    )
    print(
        f"Tip height (pen down): min {tips_down[:, 2].min() * 1000:.2f} mm, "
        f"max {tips_down[:, 2].max() * 1000:.2f} mm (floor -{contact_tol * 1000:.1f} mm)"
    )

    # First step pays the JIT warmup, so report it separately
    solve_ms = 1000 * np.array(solve_times[1:])
    print(
        f"Solve time per MPC step: mean {solve_ms.mean():.0f} ms, "
        f"p50 {np.percentile(solve_ms, 50):.0f} ms, "
        f"p99 {np.percentile(solve_ms, 99):.0f} ms, max {solve_ms.max():.0f} ms "
        f"(first step {1000 * solve_times[0]:.0f} ms; replanning budget {1000 * dt_mpc:.0f} ms)"
    )

    if show_viz:
        trace_figure(tips, progress_dense).show()
        deviation_figure(progress_dense, tips).show()

        print()
        print("Launching Viser visualization (Ctrl+C to exit)...")
        visualize(q_dense, times, progress_dense)
