"""Three-agent iLQGames with original lqrax soft collision cost.

Same scenario as ``ilqgames_three_agent.py``, but collision avoidance uses the
lqrax notebook's Gaussian repulsion in the running cost instead of CTCS
separation constraints:

    10 * exp(-5 * ||p_i - p_j||^2)

per unordered agent pair. Navigation and control penalties match the notebook.

Compare with ``ilqgames_three_agent.py`` for the CTCS hard-separation variant.

Use :func:`evaluate_lqrax_reference_costs` to score an OpenSCvx solution with the
notebook's per-agent ``dt * sum(runtime_loss)`` metrics (nav / collision / control).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem
from openscvx.algorithms.optimization_results import OptimizationResults

from examples.multi_agent.ilqgames_three_agent import (
    BICYCLE_GOAL,
    BICYCLE_WHEELBASE,
    DIFFDRIVE_GOAL,
    N,
    POINT_GOAL,
    TF,
    _show_plot,
    _trajectory_arrays,
    animate_ilqgames_three_agent,
    plot_ilqgames_three_agent,
)

# ── Cost weights (match lqrax ilqgames_example.ipynb) ────────────────────────
COLLISION_WEIGHT = 10.0
COLLISION_GAIN = 5.0
CTRL_WEIGHT = 0.1

# Notebook discretization (dt=0.05, 100 control steps → T=5 s)
LQRAX_DT = 0.05
LQRAX_STEPS = 100

# Reported notebook costs at iteration 200 (for quick comparison)
LQRAX_NOTEBOOK_BASELINE = {
    "diffdrive": 1.06,
    "point": 1.41,
    "bicycle": 1.45,
}


def _sq_dist2(a0, a1, b0, b1) -> ox.Expr:
    return (a0 - b0) ** 2 + (a1 - b1) ** 2


def _collision_penalty(ax0, ax1, bx0, bx1) -> ox.Expr:
    return ox.Constant(COLLISION_WEIGHT) * ox.Exp(
        -ox.Constant(COLLISION_GAIN) * _sq_dist2(ax0, ax1, bx0, bx1)
    )


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
    dd_theta.final = [0.0]

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
    bc_theta.initial = [np.pi / 2.0]
    bc_theta.final = [np.pi / 2.0]

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

    collision = (
        _collision_penalty(dd_pos[0], dd_pos[1], pt_pos[0], pt_pos[1])
        + _collision_penalty(dd_pos[0], dd_pos[1], bc_pos[0], bc_pos[1])
        + _collision_penalty(pt_pos[0], pt_pos[1], bc_pos[0], bc_pos[1])
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
        "stage_cost": dd_nav + pt_nav + bc_nav + collision + dd_ctrl + pt_ctrl + bc_ctrl,
    }

    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    for control in controls:
        constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

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
        # uniform_time_grid=True,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=N,
        algorithm={
            # "lam_prox": 2e0,
            # "lam_vc": 1e1,
            "lam_cost": 4e-1,
            "k_max": 1000,
            "ep_tr": 1e-6,
            # "autotuner": ox.ConstantProximalWeight(),
        },
    )
    problem.settings.prp.dt = 0.05
    # problem.settings.dev.printing = False
    return problem


@dataclass(frozen=True)
class AgentCostBreakdown:
    """Per-agent lqrax notebook cost J_i = dt * sum_k runtime_loss_i(k)."""

    nav: float
    collision: float
    control: float

    @property
    def total(self) -> float:
        return self.nav + self.collision + self.control


@dataclass(frozen=True)
class LqraxReferenceCosts:
    """Notebook-style costs for all three agents."""

    diffdrive: AgentCostBreakdown
    point: AgentCostBreakdown
    bicycle: AgentCostBreakdown
    dt: float
    n_steps: int
    openscvx_stage_cost: float | None = None

    @property
    def sum_per_agent(self) -> float:
        return self.diffdrive.total + self.point.total + self.bicycle.total


def _lqrax_reference_paths(n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight-line xy references from ``ilqgames_example.ipynb`` (``[1:]`` slice)."""
    dd_ref = np.linspace(np.array([-2.0, 0.0]), np.array([2.0, 0.0]), n_steps + 1)[1:]
    pt_ref = np.linspace(np.array([2.0, 0.0]), np.array([-2.0, 0.0]), n_steps + 1)[1:]
    bc_ref = np.linspace(np.array([0.0, -2.0]), np.array([0.0, 2.0]), n_steps + 1)[1:]
    return dd_ref, pt_ref, bc_ref


def _resample_trajectory(t_src: np.ndarray, values: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Linearly resample ``values`` (n_src, dim) onto ``t_eval``."""
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    out = np.empty((len(t_eval), values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(t_eval, t_src, values[:, j])
    return out


def _trajectory_time_and_series(
    results: OptimizationResults, name: str
) -> tuple[np.ndarray, np.ndarray]:
    if results.t_full is not None:
        t = np.asarray(results.t_full, dtype=np.float64).reshape(-1)
    elif results.trajectory and "time" in results.trajectory:
        t = np.asarray(results.trajectory["time"], dtype=np.float64).reshape(-1)
    else:
        n = len(results.nodes[name])
        t = np.linspace(0.0, TF, n)

    if results.trajectory and name in results.trajectory:
        values = np.asarray(results.trajectory[name], dtype=np.float64)
    else:
        values = np.asarray(results.nodes[name], dtype=np.float64)
    return t, values


def evaluate_lqrax_reference_costs(
    results: OptimizationResults,
    *,
    dt: float = LQRAX_DT,
    n_steps: int = LQRAX_STEPS,
    openscvx_stage_cost: float | None = None,
) -> LqraxReferenceCosts:
    """Evaluate lqrax ``ilqgames_example.ipynb`` per-agent costs on a trajectory.

    Resamples ``results.trajectory`` (or node values) onto the notebook grid
    ``t = dt, 2*dt, ..., n_steps*dt`` and applies the same runtime losses as the
    reference notebook (time-varying xy reference, Gaussian collision, weighted
    control).

    Args:
        results: Post-processed optimization results.
        dt: Notebook step size (default ``0.05``).
        n_steps: Number of control intervals (default ``100``).
        openscvx_stage_cost: Optional ``stage_cost(T)`` from OpenSCvx for reporting
            alongside the notebook metrics (not directly comparable).

    Returns:
        :class:`LqraxReferenceCosts` with per-agent nav / collision / control totals.
    """
    t_eval = np.arange(1, n_steps + 1, dtype=np.float64) * dt
    dd_ref, pt_ref, bc_ref = _lqrax_reference_paths(n_steps)

    t_dd, dd_pos = _trajectory_time_and_series(results, "diffdrive_pos")
    t_pt, pt_pos = _trajectory_time_and_series(results, "point_pos")
    t_bc, bc_pos = _trajectory_time_and_series(results, "bicycle_pos")

    t_dd_u, dd_v = _trajectory_time_and_series(results, "diffdrive_v")
    _, dd_omega = _trajectory_time_and_series(results, "diffdrive_omega")
    t_pt_u, pt_ax = _trajectory_time_and_series(results, "point_ax")
    _, pt_ay = _trajectory_time_and_series(results, "point_ay")
    t_bc_u, bc_v = _trajectory_time_and_series(results, "bicycle_v")
    _, bc_delta = _trajectory_time_and_series(results, "bicycle_delta")

    dd_xy = _resample_trajectory(t_dd, dd_pos, t_eval)
    pt_xy = _resample_trajectory(t_pt, pt_pos, t_eval)
    bc_xy = _resample_trajectory(t_bc, bc_pos, t_eval)

    dd_u = _resample_trajectory(
        t_dd_u,
        np.column_stack([dd_v.reshape(-1, 1), dd_omega.reshape(-1, 1)]),
        t_eval,
    )
    pt_u = _resample_trajectory(
        t_pt_u,
        np.column_stack([pt_ax.reshape(-1, 1), pt_ay.reshape(-1, 1)]),
        t_eval,
    )
    bc_u = _resample_trajectory(
        t_bc_u,
        np.column_stack([bc_v.reshape(-1, 1), bc_delta.reshape(-1, 1)]),
        t_eval,
    )

    dd_nav = np.sum(np.sum((dd_xy - dd_ref) ** 2, axis=1))
    pt_nav = np.sum(np.sum((pt_xy - pt_ref) ** 2, axis=1))
    bc_nav = np.sum(np.sum((bc_xy - bc_ref) ** 2, axis=1))

    dd_col = np.sum(
        COLLISION_WEIGHT * np.exp(-COLLISION_GAIN * np.sum((dd_xy - pt_xy) ** 2, axis=1))
        + COLLISION_WEIGHT * np.exp(-COLLISION_GAIN * np.sum((dd_xy - bc_xy) ** 2, axis=1))
    )
    pt_col = np.sum(
        COLLISION_WEIGHT * np.exp(-COLLISION_GAIN * np.sum((pt_xy - dd_xy) ** 2, axis=1))
        + COLLISION_WEIGHT * np.exp(-COLLISION_GAIN * np.sum((pt_xy - bc_xy) ** 2, axis=1))
    )
    bc_col = np.sum(
        COLLISION_WEIGHT * np.exp(-COLLISION_GAIN * np.sum((bc_xy - dd_xy) ** 2, axis=1))
        + COLLISION_WEIGHT * np.exp(-COLLISION_GAIN * np.sum((bc_xy - pt_xy) ** 2, axis=1))
    )

    dd_ctrl = np.sum(
        CTRL_WEIGHT * np.sum((dd_u * np.array([1.0, 0.01])) ** 2, axis=1)
    )
    pt_ctrl = np.sum(
        CTRL_WEIGHT * np.sum((pt_u * np.array([1.0, 0.5])) ** 2, axis=1)
    )
    bc_ctrl = np.sum(
        CTRL_WEIGHT * np.sum((bc_u * np.array([1.0, 0.01])) ** 2, axis=1)
    )

    scale = dt
    return LqraxReferenceCosts(
        diffdrive=AgentCostBreakdown(dd_nav * scale, dd_col * scale, dd_ctrl * scale),
        point=AgentCostBreakdown(pt_nav * scale, pt_col * scale, pt_ctrl * scale),
        bicycle=AgentCostBreakdown(bc_nav * scale, bc_col * scale, bc_ctrl * scale),
        dt=dt,
        n_steps=n_steps,
        openscvx_stage_cost=openscvx_stage_cost,
    )


def print_lqrax_reference_costs(costs: LqraxReferenceCosts) -> None:
    """Pretty-print notebook costs with per-term breakdown and notebook baseline."""
    agents = (
        ("diff-drive", "diffdrive", costs.diffdrive),
        ("point mass", "point", costs.point),
        ("bicycle", "bicycle", costs.bicycle),
    )
    print("\n── lqrax reference costs (ilqgames_example.ipynb) ──")
    print(f"grid: dt={costs.dt:g} s, {costs.n_steps} steps")
    for label, key, br in agents:
        baseline = LQRAX_NOTEBOOK_BASELINE[key]
        print(
            f"  {label:11s}  total={br.total:6.2f}  "
            f"(nav={br.nav:6.2f}, coll={br.collision:6.2f}, ctrl={br.control:6.2f})"
            f"  | notebook iter-200: {baseline:.2f}"
        )
    print(f"  sum (3 agents): {costs.sum_per_agent:.2f}")
    if costs.openscvx_stage_cost is not None:
        print(
            f"  OpenSCvx stage_cost(T): {costs.openscvx_stage_cost:.2f} "
            "(centralized objective; not directly comparable)"
        )


if __name__ == "__main__":
    plot_solution = os.environ.get("OPENSCVX_NO_PLOT") is None
    animation_path = os.environ.get(
        "OPENSCVX_ILQGAMES_ANIM",
        os.path.join(current_dir, "ilqgames_three_agent_lqr.mp4"),
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

    ref_costs = evaluate_lqrax_reference_costs(
        results,
        openscvx_stage_cost=float(nodes["stage_cost"][-1, 0]),
    )
    print_lqrax_reference_costs(ref_costs)

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
