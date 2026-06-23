"""Three-agent heterogeneous navigation (OpenSCvx port of lqrax iLQGames).

Port of the multi-agent iLQGames example from lqrax
(``examples/ilqgames_example.ipynb``), based on Fridovich-Keil et al., ICRA 2020.

Three agents cross a shared workspace while avoiding each other:

* **Diff-drive** — unicycle ``(v, omega)`` controls, state ``(x, y, theta)``
* **Point mass** — double integrator ``(a_x, a_y)``, state ``(x, y, v_x, v_y)``
* **Bicycle** — kinematic bicycle ``(v, delta)``, state ``(x, y, theta)``

The lqrax notebook solves a *general-sum differential game* via iterative
linearization (each agent optimizes its own cost given the others' trajectories).
OpenSCvx does not implement iLQGames directly; this script formulates a
**centralized cooperative** trajectory optimization that mirrors the same
scenario, dynamics, initial conditions, and goals.

Running cost (per agent, summed into one integrated ``stage_cost``):

* Quadratic tracking to the goal position
* Weighted control effort

Pairwise CTCS constraints enforce a minimum planar separation between agents.
Terminal constraints fix each agent's ``(x, y)`` at its goal; heading and
velocity are free at the final time.

See ``ilqgames_three_agent_lqr.py`` for the original lqrax Gaussian collision
penalty in the running cost instead of hard separation.
"""

from __future__ import annotations

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem

# ── Horizon (notebook: dt=0.05, 100 steps → T=5 s) ───────────────────────────
N = 51
TF = 5.0

# ── Agent goals (endpoints of the notebook reference straight lines) ─────────
DIFFDRIVE_GOAL = np.array([2.0, 0.0])
POINT_GOAL = np.array([-2.0, 0.0])
BICYCLE_GOAL = np.array([0.0, 2.0])

# ── Cost weights (match notebook structure) ──────────────────────────────────
CTRL_WEIGHT = 0.1
SAFE_DISTANCE = 0.5

BICYCLE_WHEELBASE = 0.03


def _sq_dist2(a0, a1, b0, b1) -> ox.Expr:
    return (a0 - b0) ** 2 + (a1 - b1) ** 2


def build_problem() -> Problem:
    # ── Diff-drive agent ─────────────────────────────────────────────────────
    dd_pos = ox.State("diffdrive_pos", shape=(2,))
    dd_pos.min = np.array([-3.0, -3.0])
    dd_pos.max = np.array([3.0, 3.0])
    dd_pos.initial = np.array([-2.0, -0.1])
    dd_pos.final = DIFFDRIVE_GOAL.tolist()

    dd_theta = ox.State("diffdrive_theta", shape=(1,))
    dd_theta.min = np.array([-2.0 * np.pi])
    dd_theta.max = np.array([2.0 * np.pi])
    dd_theta.initial = np.array([0.0])
    dd_theta.final = [ox.Free(0.0)]

    dd_v = ox.Control("diffdrive_v", shape=(1,), parameterization="ZOH")
    dd_v.min = np.array([0.0])
    dd_v.max = np.array([2.0])

    dd_omega = ox.Control("diffdrive_omega", shape=(1,), parameterization="ZOH")
    dd_omega.min = np.array([-3.0])
    dd_omega.max = np.array([3.0])

    # ── Point-mass agent ─────────────────────────────────────────────────────
    pt_pos = ox.State("point_pos", shape=(2,))
    pt_pos.min = np.array([-3.0, -3.0])
    pt_pos.max = np.array([3.0, 3.0])
    pt_pos.initial = np.array([2.0, 0.1])
    pt_pos.final = POINT_GOAL.tolist()

    pt_vel = ox.State("point_vel", shape=(2,))
    pt_vel.min = np.array([-3.0, -3.0])
    pt_vel.max = np.array([3.0, 3.0])
    pt_vel.initial = np.array([-0.8, 0.0])
    pt_vel.final = [ox.Free(0.0), ox.Free(0.0)]

    pt_ax = ox.Control("point_ax", shape=(1,), parameterization="ZOH")
    pt_ax.min = np.array([-2.0])
    pt_ax.max = np.array([2.0])

    pt_ay = ox.Control("point_ay", shape=(1,), parameterization="ZOH")
    pt_ay.min = np.array([-2.0])
    pt_ay.max = np.array([2.0])

    # ── Bicycle agent ────────────────────────────────────────────────────────
    bc_pos = ox.State("bicycle_pos", shape=(2,))
    bc_pos.min = np.array([-3.0, -3.0])
    bc_pos.max = np.array([3.0, 3.0])
    bc_pos.initial = np.array([-0.2, -2.0])
    bc_pos.final = BICYCLE_GOAL.tolist()

    bc_theta = ox.State("bicycle_theta", shape=(1,))
    bc_theta.min = np.array([-2.0 * np.pi])
    bc_theta.max = np.array([2.0 * np.pi])
    bc_theta.initial = np.array([np.pi / 2.0])
    bc_theta.final = [ox.Free(0.0)]

    bc_v = ox.Control("bicycle_v", shape=(1,), parameterization="ZOH")
    bc_v.min = np.array([0.0])
    bc_v.max = np.array([1.5])

    bc_delta = ox.Control("bicycle_delta", shape=(1,), parameterization="ZOH")
    bc_delta.min = np.array([-np.pi / 4.0])
    bc_delta.max = np.array([np.pi / 4.0])

    # ── Integrated running cost ──────────────────────────────────────────────
    stage_cost = ox.State("stage_cost", shape=(1,))
    stage_cost.min = np.array([0.0])
    stage_cost.max = np.array([1e4])
    stage_cost.initial = np.array([0.0])
    stage_cost.final = [ox.Minimize(0.0)]

    states = [
        dd_pos,
        dd_theta,
        pt_pos,
        pt_vel,
        bc_pos,
        bc_theta,
        stage_cost,
    ]
    controls = [dd_v, dd_omega, pt_ax, pt_ay, bc_v, bc_delta]

    # ── Dynamics ─────────────────────────────────────────────────────────────
    dd_nav = _sq_dist2(dd_pos[0], dd_pos[1], DIFFDRIVE_GOAL[0], DIFFDRIVE_GOAL[1])
    pt_nav = _sq_dist2(pt_pos[0], pt_pos[1], POINT_GOAL[0], POINT_GOAL[1])
    bc_nav = _sq_dist2(bc_pos[0], bc_pos[1], BICYCLE_GOAL[0], BICYCLE_GOAL[1])

    dd_ctrl = ox.Constant(CTRL_WEIGHT) * (
        dd_v[0] ** 2 + ox.Constant(0.01) * dd_omega[0] ** 2
    )
    pt_ctrl = ox.Constant(CTRL_WEIGHT) * (
        pt_ax[0] ** 2 + ox.Constant(0.5) * pt_ay[0] ** 2
    )
    bc_ctrl = ox.Constant(CTRL_WEIGHT) * (
        bc_v[0] ** 2 + ox.Constant(0.01) * bc_delta[0] ** 2
    )

    dynamics = {
        "diffdrive_pos": ox.Concat(
            dd_v[0] * ox.Cos(dd_theta[0]),
            dd_v[0] * ox.Sin(dd_theta[0]),
        ),
        "diffdrive_theta": dd_omega[0],
        "point_pos": pt_vel,
        "point_vel": ox.Concat(pt_ax[0], pt_ay[0]),
        "bicycle_pos": ox.Concat(
            bc_v[0] * ox.Cos(bc_theta[0]),
            bc_v[0] * ox.Sin(bc_theta[0]),
        ),
        "bicycle_theta": bc_v[0]
        * ox.Tan(bc_delta[0])
        / ox.Constant(BICYCLE_WHEELBASE),
        "stage_cost": dd_nav + pt_nav + bc_nav + dd_ctrl + pt_ctrl + bc_ctrl,
    }

    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    for control in controls:
        constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

    min_sep = ox.Constant(SAFE_DISTANCE)
    constraints.extend(
        [
            ox.ctcs(min_sep <= ox.linalg.Norm(dd_pos - pt_pos)),
            ox.ctcs(min_sep <= ox.linalg.Norm(dd_pos - bc_pos)),
            ox.ctcs(min_sep <= ox.linalg.Norm(pt_pos - bc_pos)),
        ]
    )

    # ── Initial guess (notebook warm start) ────────────────────────────────
    t_vals = np.linspace(0.0, 1.0, N)

    dd_pos.guess = (
        (1.0 - t_vals)[:, None] * dd_pos.initial + t_vals[:, None] * DIFFDRIVE_GOAL
    )
    dd_theta.guess = np.zeros((N, 1))
    dd_v.guess = np.full((N, 1), 0.8)
    dd_omega.guess = np.zeros((N, 1))

    pt_pos.guess = (1.0 - t_vals)[:, None] * pt_pos.initial + t_vals[:, None] * POINT_GOAL
    pt_vel.guess = np.tile(pt_vel.initial, (N, 1))
    pt_ax.guess = np.zeros((N, 1))
    pt_ay.guess = np.zeros((N, 1))

    bc_pos.guess = (1.0 - t_vals)[:, None] * bc_pos.initial + t_vals[:, None] * BICYCLE_GOAL
    bc_theta.guess = np.full((N, 1), np.pi / 2.0)
    bc_v.guess = np.full((N, 1), 0.5)
    bc_delta.guess = np.zeros((N, 1))

    stage_cost.guess = np.zeros((N, 1))

    time = ox.Time(
        initial=0.0,
        final=TF,
        min=0.0,
        max=TF,
        uniform_time_grid=True,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=N,
        algorithm={
            "lam_prox": 2e0,
            "lam_vc": 1e1,
            "lam_cost": 2e0,
            "autotuner": ox.ConstantProximalWeight(),
        },
        # discretizer=ox.LinearizeDiscretizeSparse(dis_type="ZOH"),
    )
    problem.settings.prp.dt = 0.05
    problem.settings.dev.printing = False
    return problem


def _trajectory_arrays(results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return diff-drive, point-mass, and bicycle state trajectories."""
    traj = results.trajectory
    source = traj if traj else results.nodes

    diffdrive = np.column_stack(
        [
            np.asarray(source["diffdrive_pos"], dtype=np.float64),
            np.asarray(source["diffdrive_theta"], dtype=np.float64).reshape(-1, 1),
        ]
    )
    point = np.column_stack(
        [
            np.asarray(source["point_pos"], dtype=np.float64),
            np.asarray(source["point_vel"], dtype=np.float64),
        ]
    )
    bicycle = np.column_stack(
        [
            np.asarray(source["bicycle_pos"], dtype=np.float64),
            np.asarray(source["bicycle_theta"], dtype=np.float64).reshape(-1, 1),
        ]
    )
    n = min(len(diffdrive), len(point), len(bicycle))
    return diffdrive[:n], point[:n], bicycle[:n]


def animate_ilqgames_three_agent(
    diffdrive: np.ndarray,
    point: np.ndarray,
    bicycle: np.ndarray,
    *,
    save_path: str | None = None,
    show: bool = True,
    interval_ms: int = 50,
) -> None:
    """Matplotlib animation matching lqrax ``ilqgames_example.ipynb``."""
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt

    colors = ("C0", "C1", "C2")
    goals = (DIFFDRIVE_GOAL, POINT_GOAL, BICYCLE_GOAL)
    n_frames = len(diffdrive)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120, tight_layout=True)

    def update(t: int):
        ax.cla()
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis("off")

        for goal, color in zip(goals, colors):
            ax.plot(
                goal[0],
                goal[1],
                linestyle="",
                marker="X",
                markersize=20,
                color=color,
                alpha=0.5,
            )

        # Diff-drive: diamond body + heading tick
        dd_xt = diffdrive[t]
        dd_theta = dd_xt[2]
        dd_angle = np.rad2deg(dd_theta)
        ax.plot(
            diffdrive[: t + 1, 0],
            diffdrive[: t + 1, 1],
            linestyle="-",
            linewidth=5,
            color="C0",
            alpha=0.5,
        )
        ax.plot(
            dd_xt[0],
            dd_xt[1],
            linestyle="",
            marker=(4, 0, dd_angle + 45),
            markersize=30,
            color="C0",
        )
        ax.plot(
            dd_xt[0] + np.cos(dd_theta) * 0.32,
            dd_xt[1] + np.sin(dd_theta) * 0.32,
            linestyle="",
            marker=(3, 0, dd_angle + 30),
            markersize=15,
            color="C0",
        )

        # Point mass: circle body + velocity heading
        pt_xt = point[t]
        pt_theta = np.arctan2(pt_xt[3], pt_xt[2])
        pt_angle = np.rad2deg(pt_theta)
        ax.plot(
            point[: t + 1, 0],
            point[: t + 1, 1],
            linestyle="-",
            linewidth=5,
            color="C1",
            alpha=0.5,
        )
        ax.plot(
            pt_xt[0],
            pt_xt[1],
            linestyle="",
            marker="o",
            markersize=25,
            color="C1",
        )
        ax.plot(
            pt_xt[0] + np.cos(pt_theta) * 0.36,
            pt_xt[1] + np.sin(pt_theta) * 0.36,
            linestyle="",
            marker=(3, 0, pt_angle + 30),
            markersize=15,
            color="C1",
        )

        # Bicycle: diamond body + heading tick
        bc_xt = bicycle[t]
        bc_theta = bc_xt[2]
        bc_angle = np.rad2deg(bc_theta)
        ax.plot(
            bicycle[: t + 1, 0],
            bicycle[: t + 1, 1],
            linestyle="-",
            linewidth=5,
            color="C2",
            alpha=0.5,
        )
        ax.plot(
            bc_xt[0],
            bc_xt[1],
            linestyle="",
            marker=(4, 0, bc_angle + 45),
            markersize=30,
            color="C2",
        )
        ax.plot(
            bc_xt[0] + np.cos(bc_theta) * 0.33,
            bc_xt[1] + np.sin(bc_theta) * 0.33,
            linestyle="",
            marker=(3, 0, bc_angle + 30),
            markersize=15,
            color="C2",
        )
        return []

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
    )

    if save_path is not None:
        if save_path.endswith(".gif"):
            writer = mpl_animation.PillowWriter()
            ani.save(save_path, writer=writer)
        else:
            ani.save(save_path, writer="ffmpeg")
        print(f"Saved animation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ilqgames_three_agent(results, *, width: int = 720, height: int = 720):
    """Plot planar trajectories for all three agents."""
    import plotly.graph_objects as go

    from openscvx.plotting.publication import LM_PLOTLY_FONT as _LM_PLOTLY_FONT
    from openscvx.plotting.publication import LM_PLOTLY_TICK_FONT as _LM_PLOTLY_TICK_FONT

    traj = results.trajectory
    dd = np.asarray(traj["diffdrive_pos"], dtype=np.float64)
    pt = np.asarray(traj["point_pos"], dtype=np.float64)
    bc = np.asarray(traj["bicycle_pos"], dtype=np.float64)

    goals = [
        (DIFFDRIVE_GOAL, "diff-drive", "#1f77b4"),
        (POINT_GOAL, "point mass", "#ff7f0e"),
        (BICYCLE_GOAL, "bicycle", "#2ca02c"),
    ]
    paths = [dd, pt, bc]

    fig = go.Figure()
    for path, (goal, label, color) in zip(paths, goals):
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines",
                name=label,
                line={"color": color, "width": 2.0},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[path[0, 0]],
                y=[path[0, 1]],
                mode="markers",
                showlegend=False,
                marker={"color": color, "size": 10, "symbol": "circle"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[goal[0]],
                y=[goal[1]],
                mode="markers",
                name=f"{label} goal",
                marker={"color": color, "size": 14, "symbol": "x"},
            )
        )

    fig.update_layout(
        title="Three-agent iLQGames scenario (OpenSCvx cooperative port)",
        xaxis_title=r"$x$",
        yaxis_title=r"$y$",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_LM_PLOTLY_FONT,
        autosize=False,
        width=width,
        height=height,
        margin={"l": 60, "r": 40, "t": 48, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(title_font=_LM_PLOTLY_FONT, tickfont=_LM_PLOTLY_TICK_FONT)
    fig.update_yaxes(title_font=_LM_PLOTLY_FONT, tickfont=_LM_PLOTLY_TICK_FONT)
    return fig


def _show_plot(fig):
    try:
        fig.show()
    except PermissionError as exc:
        print(f"Skipping plot display: {exc}")


if __name__ == "__main__":
    plot_solution = os.environ.get("OPENSCVX_NO_PLOT") is None
    animation_path = os.environ.get(
        "OPENSCVX_ILQGAMES_ANIM",
        os.path.join(current_dir, "ilqgames_three_agent.mp4"),
    )

    problem = build_problem()
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print(f"converged: {results.converged}")
    print(f"integrated cost: {nodes['stage_cost'][-1, 0]:.4f}")
    print(f"diff-drive final pos: {nodes['diffdrive_pos'][-1]}  (goal {DIFFDRIVE_GOAL})")
    print(f"point mass final pos: {nodes['point_pos'][-1]}  (goal {POINT_GOAL})")
    print(f"bicycle final pos:    {nodes['bicycle_pos'][-1]}  (goal {BICYCLE_GOAL})")

    if plot_solution:
        _show_plot(plot_ilqgames_three_agent(results))
        diffdrive, point, bicycle = _trajectory_arrays(results)
        try:
            animate_ilqgames_three_agent(
                diffdrive,
                point,
                bicycle,
                save_path=animation_path,
                show=False,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            gif_path = os.path.splitext(animation_path)[0] + ".gif"
            print(f"Could not save MP4 ({exc}); trying GIF at {gif_path}")
            animate_ilqgames_three_agent(
                diffdrive,
                point,
                bicycle,
                save_path=gif_path,
                show=False,
            )
