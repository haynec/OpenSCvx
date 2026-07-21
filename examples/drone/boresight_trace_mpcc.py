"""MPCC SVG tracing: a quadrotor draws an SVG path with its camera boresight.

This example unifies two ideas from the examples suite:

- ``drone/openscvx_logo.py`` — a 6-DOF quadrotor whose boresight traces the
  OpenSCvx logo, with the target point parametrized by *time*
- ``mpc/double_integrator_drone_racing.py`` — model-predictive contouring
  control (MPCC, [Romero et al. 2022](https://arxiv.org/abs/2108.13205)),
  where progress along a reference path is a state the optimizer advances

Parametrizing the logo target by time couples the node count to the path
complexity (the one-shot version needs 500 nodes). Here the target is
parametrized by *arc length* instead: a cubic spline (``ox.Cinterp``) maps a
progress state to a point on the SVG path, and a progress-rate control
advances it as fast as drawing accuracy allows while a short receding horizon
retraces the path chunk by chunk. The "pen" is the point where the camera
boresight pierces the drawing plane, and the contouring cost — the pen's
squared distance to the path point at the current progress — is fully
symbolic, spline lookup included: no bring-your-own-function dynamics. Any
single-path SVG works unchanged (the OpenSCvx wordmark by default):

    python boresight_trace_mpcc.py [path/to/drawing.svg]

Requires:
    pip install svgpathtools
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.drone.logo_utils.svg_path_utils import extract_svg_arc_length_path
from openscvx import Problem

###############################################################################
# Reference path: SVG -> arc-length-parametrized curve in a vertical plane
###############################################################################

_DEFAULT_SVG = os.path.join(current_dir, "logo_utils", "openscvx_logo_single.svg")
svg_file = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SVG

path_width = 10.0  # Larger dimension of the drawing [m]
path_center = np.array([1.0, 0.0, 5.0])  # Center of the drawing plane
hover_pos = np.array([-5.0, 0.0, 5.0])  # Drone hovers here, looking +x at the plane

# Path 0 is the drawing in both bundled SVGs (later paths are background rects)
s_path, planar_points = extract_svg_arc_length_path(svg_file, path_indices=[0], width=path_width)
total_arc_length = s_path[-1]

# Lift the planar curve into the world: the drawing spans the y-z plane at
# x = path_center[0], with the SVG's y-axis pointing up (world z).
path_points = path_center + np.column_stack(
    [np.zeros(len(s_path)), planar_points[:, 0], planar_points[:, 1]]
)

# Spline tables extend the reference past the true end by more than one horizon
# of pen travel, holding it parked at the final path point. The progress reward
# keeps pulling the progress state forward — without the extension it saturates
# as the horizon runs out of table and the endgame dissolves into a hovering
# scribble — but the reference *point* stays put, so the pen decelerates into
# the endpoint and settles there instead of chasing a fictitious continuation
# off the wordmark. The loop stops as the pen crosses the true finish, so the
# parked tail collapses onto the endpoint rather than trailing ink past it.
pad_length = 10.0  # [m]
_ds = s_path[1] - s_path[0]
_s_pad = np.arange(1, int(pad_length / _ds)) * _ds
s_ref = np.concatenate([s_path, total_arc_length + _s_pad])
ref_points = np.concatenate([path_points, np.tile(path_points[-1], (len(_s_pad), 1))])
padded_arc_length = s_ref[-1]


def ref_point(s):
    """Numeric reference lookup used for guesses and analysis (matches Cinterp data)."""
    return np.stack([np.interp(s, s_ref, ref_points[:, i]) for i in range(3)], axis=-1)


boresight_body = np.array([1.0, 0.0, 0.0])  # Boresight is the body x-axis


def look_at(from_pos, target):
    """Quaternion [w, x, y, z] rotating the body boresight onto ``target - from_pos``."""
    a = target - from_pos
    a = a / np.linalg.norm(a)
    q = np.concatenate([[1.0 + boresight_body @ a], np.cross(boresight_body, a)])
    return q / np.linalg.norm(q)


###############################################################################
# MPCC problem: 6-DOF quadrotor + progress along the path
###############################################################################

n_mpc = 13  # Horizon nodes
horizon_duration = 2.4  # Horizon length [s]
dt_mpc = horizon_duration / (n_mpc - 1)  # Executed step [s]

Q_TRACE = 4e1  # Contouring error weight (squared trace-vs-path distance)
Q_PROGRESS = 1e-1  # Progress reward
Q_SMOOTH = 2e-1  # Acceleration (thrust tilt) penalty: calms the airframe

trace_rate_max = 1.5  # Max speed of the traced point along the path [m/s]
trace_rate_guess = 1.0

m = 1.0  # Mass of the drone
g_const = -9.81
J_b = np.array([0.01, 0.01, 0.02])  # Body inertia of a 1 kg racing quad [kg m^2]

# --- States ---
# The x band holds a healthy standoff from the canvas: too close and the
# pointing geometry demands wild attitude swings, too far and every milliradian
# of attitude wobble smears centimeters of ink. The y-z room generously covers
# the drawing so the drone can track the pen's height and never gets cornered.
position = ox.State("position", shape=(3,))
position.max = np.array([-4.0, 7.0, 11.0])
position.min = np.array([-7.5, -7.0, 0.5])
position.initial = hover_pos
position.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([10.0, 10.0, 10.0])
velocity.min = np.array([-10.0, -10.0, -10.0])
velocity.initial = np.zeros(3)
velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

attitude = ox.State("attitude", shape=(4,))  # Quaternion [w, x, y, z]
attitude.max = np.ones(4)
attitude.min = -np.ones(4)
attitude.initial = look_at(hover_pos, ref_point(0.0))
attitude.final = [ox.Free(0.0)] * 4

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([6.0, 6.0, 6.0])
angular_velocity.min = np.array([-6.0, -6.0, -6.0])
angular_velocity.initial = np.zeros(3)
angular_velocity.final = [ox.Free(0.0)] * 3

progress = ox.State("progress", shape=(1,))  # Arc length along the SVG path
progress.min = np.array([0.0])
progress.max = np.array([padded_arc_length])
progress.initial = np.array([0.0])
progress.final = [ox.Maximize(0.0)]

trace_error = ox.State("trace_error", shape=(1,))  # Integrated squared trace error
trace_error.min = np.array([0.0])
trace_error.max = np.array([2.0])
trace_error.initial = np.array([0.0])
trace_error.final = [ox.Minimize(0.0)]

smoothness = ox.State("smoothness", shape=(1,))  # Integrated squared acceleration
smoothness.min = np.array([0.0])
smoothness.max = np.array([100.0])
smoothness.initial = np.array([0.0])
smoothness.final = [ox.Minimize(0.0)]

# --- Controls ---
thrust_force = ox.Control("thrust_force", shape=(3,))  # Body-frame thrust (z only)
thrust_force.max = np.array([0.0, 0.0, 4.179446268 * 9.81])
thrust_force.min = np.array([0.0, 0.0, 0.0])

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([1.0, 1.0, 0.3])
torque.min = np.array([-1.0, -1.0, -0.3])

trace_rate = ox.Control("trace_rate", shape=(1,))  # d(progress)/dt
trace_rate.min = np.array([0.0])  # The pen only moves forward
trace_rate.max = np.array([trace_rate_max])

# --- Contouring error (fully symbolic via spline lookup of the path) ---
# The "pen" is where the boresight ray pierces the drawing plane; the error is
# its distance to the path point at the current progress. Penalizing the pen
# (rather than a pointing angle) makes the optimizer orchestrate position and
# attitude together — a pitched quadrotor necessarily accelerates sideways, so
# drawing accurately while hovering is a genuinely coupled 6-DOF problem.
p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_ref, ref_points[:, 0]),
    ox.Cinterp(progress[0], s_ref, ref_points[:, 1]),
    ox.Cinterp(progress[0], s_ref, ref_points[:, 2]),
)

# Guarded denominators: SCP iterates can wander through states the constraints
# only push back on softly (q near 0, boresight grazing the plane), and the
# linearization must stay finite there. Both guards are inactive at any
# solution the constraints accept.
attitude_normalized = attitude / ox.Sqrt(ox.Sum(attitude * attitude) + 1e-8)
boresight = ox.spatial.QDCM(attitude_normalized) @ ox.Constant(boresight_body)

pen = position + boresight * ((path_center[0] - position[0]) / ox.Max(boresight[0], 0.1))
pen_error = pen - p_ref
trace_cost = ox.Sum(pen_error * pen_error)

# --- Dynamics ---
# The smoothness integrand prices acceleration (tilt-induced lurching that
# drags the pen off the path) and velocity (damping that keeps the
# receding-horizon loop from building oscillations step over step). Velocity is
# weighted 3x: a becalmed airframe holds the pen on the line far more than it
# slows the pen's pace, so the heavier damping cuts trace error without
# stretching the flight. Together with the progress reward this sets the cruise
# pace of the pen.
J_b_diag = ox.linalg.Diag(J_b)
acceleration = (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force + np.array(
    [0.0, 0.0, g_const]
)
dynamics = {
    "position": velocity,
    "velocity": acceleration,
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(1.0 / J_b)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    "progress": trace_rate,
    "trace_error": trace_cost,
    "smoothness": ox.Sum(acceleration * acceleration) + 3.0 * ox.Sum(velocity * velocity),
}

# --- Constraints ---
states = [position, velocity, attitude, angular_velocity, progress, trace_error, smoothness]
controls = [thrust_force, torque, trace_rate]

constraints = []
for state in [position, velocity, angular_velocity]:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Keep the camera facing the canvas (also keeps the pen intersection well-posed)
constraints.append(ox.ctcs(0.2 <= boresight[0]))

# --- Time: fixed horizon with a pinned first step (the executed segment) ---
# Dilation bounds keep the node spacing near-uniform: freely stretched
# segments integrate saturated dynamics over long intervals and blow up
t = ox.Time(
    initial=0.0,
    final=horizon_duration,
    min=0.0,
    max=horizon_duration,
    time_dilation_min=0.5 * horizon_duration,
    time_dilation_max=2.0 * horizon_duration,
)
constraints.append((t == dt_mpc).convex().at(1))


###############################################################################
# Initial guess and warm-starting
###############################################################################


def set_initial_guess():
    """Hover at the start location, boresight swept along the first path chunk."""
    arc_guess = np.linspace(0.0, trace_rate_guess * horizon_duration, n_mpc)
    position.guess = np.tile(hover_pos, (n_mpc, 1))
    velocity.guess = np.zeros((n_mpc, 3))
    attitude.guess = np.array([look_at(hover_pos, ref_point(s)) for s in arc_guess])
    attitude.initial = attitude.guess[0]
    angular_velocity.guess = np.zeros((n_mpc, 3))
    progress.guess = arc_guess.reshape(-1, 1)
    trace_error.guess = np.zeros((n_mpc, 1))
    smoothness.guess = np.zeros((n_mpc, 1))

    thrust_force.guess = np.tile([0.0, 0.0, -m * g_const], (n_mpc, 1))
    torque.guess = np.zeros((n_mpc, 3))
    trace_rate.guess = np.full((n_mpc, 1), trace_rate_guess)


def update_initial_conditions(nodes: dict):
    """Set initial conditions from node 1 of the previous solution (simulate one step).

    Solutions can sit exactly on a state bound, so clip away the solver's
    float roundoff before handing the values back as fixed initial conditions.
    """
    for state in [position, velocity, attitude, angular_velocity, progress]:
        state.initial = np.clip(nodes[state.name][1], state.min, state.max)
    # Cost integrators restart from zero each horizon
    trace_error.initial = np.array([0.0])
    smoothness.initial = np.array([0.0])


def shift_guess(nodes: dict):
    """Shift the previous solution by one node; extend the tip with a hover prior.

    The freshly revealed tip node points its attitude along the path at the
    extended progress but is otherwise seeded as a calm hover — zero velocity,
    hover thrust, zero torque, position held from the last node. Repeating the
    last node's momentum instead extrapolates any drift into the new node,
    which the receding-horizon loop can amplify step over step; a hover prior
    lets the optimizer discover the motion the path actually needs.
    """
    ext_progress = min(
        nodes["progress"][-1, 0] + nodes["trace_rate"][-1, 0] * dt_mpc, padded_arc_length
    )
    ext_attitude = look_at(nodes["position"][-1], ref_point(ext_progress))

    position.guess = np.vstack([nodes["position"][1:], nodes["position"][-1:]])
    velocity.guess = np.vstack([nodes["velocity"][1:], np.zeros((1, 3))])
    attitude.guess = np.vstack([nodes["attitude"][1:], [ext_attitude]])
    angular_velocity.guess = np.vstack([nodes["angular_velocity"][1:], np.zeros((1, 3))])
    progress.guess = np.vstack([nodes["progress"][1:], [[ext_progress]]])

    for integrator, key in [(trace_error, "trace_error"), (smoothness, "smoothness")]:
        offset = nodes[key][1]
        integrator.guess = np.maximum(
            np.vstack([nodes[key][1:] - offset, nodes[key][-1:] - offset]), 0.0
        )

    thrust_force.guess = np.vstack([nodes["thrust_force"][1:], [[0.0, 0.0, -m * g_const]]])
    torque.guess = np.vstack([nodes["torque"][1:], np.zeros((1, 3))])
    trace_rate.guess = np.vstack([nodes["trace_rate"][1:], nodes["trace_rate"][-1:]])

    # Time: shift and renormalize so the horizon starts at t = 0
    ext_time = nodes["time"][-1, 0] + nodes["_time_dilation"][-1, 0] * (1.0 / (n_mpc - 1))
    shifted_time = np.vstack([nodes["time"][1:], [[ext_time]]])
    t.guess = shifted_time - shifted_time[0]
    t._time_dilation_control.guess = np.vstack(
        [nodes["_time_dilation"][1:], nodes["_time_dilation"][-1:]]
    )


set_initial_guess()

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=t,
    constraints=constraints,
    N=n_mpc,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": {"trace_error": Q_TRACE, "progress": Q_PROGRESS, "smoothness": Q_SMOOTH},
        "lam_prox": 3e1,  # Strong anchoring: weak prox never converges here
        "lam_vc": 1e4,  # Keep dynamics honest: far above the largest cost weight
        "k_max": 30,  # Warm-started horizons converge in a handful of iterations
    },
    float_dtype="float64",
    # Integrator tolerance must stay well below the O(1e-4) trace-error
    # integrand or it silently swallows the cost signal; 1e-6 is the fastest
    # setting that preserves tracking accuracy
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-6, "rtol": 1e-6}),
    # Relaxed subproblem tolerances: corner subproblems otherwise grind toward
    # QOCO's defaults and dominate the worst-case step time
    solver={"solver_args": {"abstol": 3e-5, "reltol": 3e-7, "enforce_dpp": True}},
)
problem.settings.dev.printing = False


###############################################################################
# Analysis: where does the boresight actually draw on the plane?
###############################################################################


def qdcm(q):
    """Quaternion [w, x, y, z] to DCM (numeric twin of ox.spatial.QDCM)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def boresight_trace(positions, attitudes):
    """Intersect the boresight ray with the drawing plane x = path_center[0]."""
    points = []
    for pos, att in zip(positions, attitudes):
        direction = qdcm(att) @ boresight_body
        points.append(pos + direction * (path_center[0] - pos[0]) / direction[0])
    return np.array(points)


def trace_figure(traced, times):
    """Plotly figure: the SVG reference path against the drawn trace, colored by pen speed."""
    import plotly.graph_objects as go

    speed = np.linalg.norm(np.gradient(traced, times, axis=0), axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=path_points[:, 1],
            y=path_points[:, 2],
            mode="lines",
            name="reference path",
            line=dict(color="black", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=traced[:, 1],
            y=traced[:, 2],
            mode="markers",
            name="pen trace",
            marker=dict(
                color=speed,
                colorscale="Rainbow",
                size=4,
                colorbar=dict(title="pen speed [m/s]"),
                showscale=True,
            ),
        )
    )
    fig.update_layout(
        title=f"Boresight pen trace — {os.path.basename(svg_file)}",
        xaxis=dict(title="y [m]", scaleanchor="y"),
        yaxis=dict(title="z [m]"),
        legend=dict(x=0.01, y=0.99),
        height=600,
    )
    return fig


def deviation_figure(progress_dense, trace_err):
    """Plotly figure: trace deviation against arc length along the path."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=progress_dense,
            y=trace_err * 100,
            mode="lines",
            name="trace error",
            line=dict(color="black", width=1),
        )
    )
    fig.update_layout(
        title=f"Trace deviation — {os.path.basename(svg_file)}",
        xaxis=dict(title="path length [m]"),
        yaxis=dict(title="trace error [cm]"),
        height=400,
    )
    return fig


###############################################################################
# Main: receding-horizon loop
###############################################################################

if __name__ == "__main__":
    print("Quadrotor boresight tracing via MPCC")
    print("=" * 60)
    print(
        f"Path: {os.path.basename(svg_file)}, {total_arc_length:.1f} m of stroke "
        f"at {path_width:.0f} m wide  |  Horizon: {n_mpc} nodes / {horizon_duration} s"
    )
    print()

    problem.initialize()

    max_steps = 600

    seg_positions = []
    seg_attitudes = []
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
        seg_positions.append(results.trajectory["position"][:seg_end].copy())
        seg_attitudes.append(results.trajectory["attitude"][:seg_end].copy())
        seg_progress.append(results.trajectory["progress"][:seg_end, 0].copy())
        seg_times.append(traj_time[:seg_end] + step * dt_mpc)

        cur_progress = nodes["progress"][0, 0]
        print(
            f"step {step:3d}: progress={cur_progress:7.2f} m "
            f"({100 * cur_progress / total_arc_length:5.1f}%), "
            f"trace_cost={nodes['trace_error'][-1, 0]:.5f}"
        )

        # The pen has crossed the true end of the drawing (the horizon may
        # already be planning into the reference extension, which is fine)
        if nodes["progress"][1, 0] >= total_arc_length:
            break

        update_initial_conditions(nodes)
        shift_guess(nodes)

    # --- Trace accuracy report ---
    positions = np.concatenate(seg_positions)
    attitudes = np.concatenate(seg_attitudes)
    progress_dense = np.concatenate(seg_progress)
    times = np.concatenate(seg_times)

    traced = boresight_trace(positions, attitudes)
    trace_err = np.linalg.norm(traced - ref_point(progress_dense), axis=1)
    print(
        f"\nTraced {progress_dense[-1]:.1f} / {total_arc_length:.1f} m "
        f"in {times[-1]:.1f} s with {n_mpc}-node horizons"
    )
    print(
        f"Trace error vs reference: mean {trace_err.mean() * 100:.2f} cm, "
        f"max {trace_err.max() * 100:.2f} cm"
    )

    # First step pays the JIT warmup, so report it separately
    solve_ms = 1000 * np.array(solve_times[1:])
    print(
        f"Solve time per MPC step: mean {solve_ms.mean():.0f} ms, "
        f"p50 {np.percentile(solve_ms, 50):.0f} ms, "
        f"p99 {np.percentile(solve_ms, 99):.0f} ms, max {solve_ms.max():.0f} ms "
        f"(first step {1000 * solve_times[0]:.0f} ms; replanning budget {1000 * dt_mpc:.0f} ms)"
    )

    trace_figure(traced, times).show()
    deviation_figure(progress_dense, trace_err).show()

    # --- Visualization ---
    from examples.drone.logo_utils.quadrotor_mesh import make_quadrotor_mesh
    from openscvx.plotting.viser import add_animation_controls, create_server

    server = create_server(positions)

    # The canvas: reference path as a continuous grey line, gold ink stroked on
    # top of it as the animation plays (all trails are line segments — point
    # clouds fail to render on some browser/GPU combinations)
    server.scene.add_line_segments(
        "/reference_path",
        points=np.stack([path_points[:-1], path_points[1:]], axis=1).astype(np.float32),
        colors=(140, 140, 145),
        line_width=2.0,
    )

    def add_line_trail(name, trail, color, line_width):
        """Growing line-strip trail; returns an update callback for the animation.

        The segment buffer keeps a constant size — not-yet-drawn segments are
        collapsed to zero length instead of sliced away, because some clients
        drop buffer updates that change size.
        """
        segments = np.stack([trail[:-1], trail[1:]], axis=1).astype(np.float32)
        handle = server.scene.add_line_segments(
            name, points=segments, colors=color, line_width=line_width
        )

        def update(frame_idx: int) -> None:
            shown = segments.copy()
            shown[frame_idx:] = trail[min(frame_idx, len(trail) - 1)]
            handle.points = shown

        return update

    # Trails load fully drawn; pressing Play rewinds and animates the drawing
    update_ink = add_line_trail("/ink", traced, (255, 200, 60), line_width=5.0)
    update_flight = add_line_trail("/flight_path", positions, (120, 170, 255), line_width=2.0)

    # The artist: posed quadrotor mesh with its boresight beam to the pen
    mesh_vertices, mesh_faces = make_quadrotor_mesh()
    drone_handle = server.scene.add_mesh_simple(
        "/drone",
        vertices=np.asarray(mesh_vertices, dtype=np.float32),
        faces=np.asarray(mesh_faces, dtype=np.uint32),
        color=(150, 155, 165),
        position=tuple(positions[0]),
        wxyz=tuple(attitudes[0]),
    )
    beam_handle = server.scene.add_line_segments(
        "/boresight_beam",
        points=np.array([[positions[0], traced[0]]]),
        colors=(255, 90, 90),
        line_width=2.0,
    )

    def update_drone(frame_idx: int) -> None:
        q = attitudes[frame_idx] / np.linalg.norm(attitudes[frame_idx])
        drone_handle.position = tuple(float(x) for x in positions[frame_idx])
        drone_handle.wxyz = tuple(float(x) for x in q)

    def update_beam(frame_idx: int) -> None:
        beam_handle.points = np.array([[positions[frame_idx], traced[frame_idx]]])

    add_animation_controls(server, times, [update_ink, update_flight, update_drone, update_beam])
    server.sleep_forever()
