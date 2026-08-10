"""Use an RL-collected neural dynamics model inside an OpenSCvx problem.

Pipeline
--------
1. A PPO policy explores a *true* nonlinear plant (quadratic drag + bias force)
   that OpenSCvx never sees symbolically (see ``_learned_dynamics.py``).
2. Transitions logged during exploration train an MLP acceleration model
   ``a_θ(x, u) ≈ v̇``.
3. OpenSCvx optimizes a constrained transfer with **hybrid dynamics**:
   known kinematics ``ṗ = v`` (symbolic) + learned ``v̇ = a_θ(x, u)`` (BYOF).
4. A CTCS keep-out constraint is enforced on the *learned* model; the script
   also re-simulates the OpenSCvx controls on the true plant for comparison.

This is the model-based-RL complementary pattern to ``rl_warmstart_obstacle.py``
(policy warm-start). Here the learned artifact is the **dynamics**, not the guess.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import openscvx as ox
from openscvx import Problem

from _learned_dynamics import (  # noqa: E402
    A_MAX,
    DT,
    GOAL,
    HORIZON,
    X0,
    PPOConfig,
    dynamics_net_apply,
    explore_with_ppo,
    fit_dynamics,
    load_dynamics,
    save_dynamics,
    true_step,
)

N = HORIZON
TF = DT * (N - 1)
OBSTACLE_CENTER = np.array([0.0, 0.0])
OBSTACLE_RADIUS = 0.85
DYN_PATH = Path(current_dir) / "assets" / "learned_accel.npz"


def _ensure_dynamics(path: Path = DYN_PATH, *, retrain: bool = False, updates: int = 300):
    if path.is_file() and not retrain:
        return load_dynamics(path)
    print(f"Exploring plant + fitting dynamics ({updates} PPO updates) → {path}")
    _, batch = explore_with_ppo(seed=0, ppo_cfg=PPOConfig(num_updates=updates))
    params = fit_dynamics(batch, seed=0, num_epochs=80)
    save_dynamics(params, path)
    return params


def _build_problem(dyn_params) -> Problem:
    position = ox.State("position", shape=(2,))
    position.min = np.array([-3.5, -3.5])
    position.max = np.array([3.5, 3.5])
    position.initial = X0[:2].astype(np.float64)
    position.final = GOAL.astype(np.float64)

    velocity = ox.State("velocity", shape=(2,))
    velocity.min = np.array([-4.0, -4.0])
    velocity.max = np.array([4.0, 4.0])
    velocity.initial = X0[2:].astype(np.float64)
    velocity.final = [ox.Free(0.0), ox.Free(0.0)]

    force = ox.Control("force", shape=(2,), parameterization="ZOH")
    force.min = np.array([-A_MAX, -A_MAX])
    force.max = np.array([A_MAX, A_MAX])

    # Known kinematics stay symbolic; acceleration is provided only via BYOF
    # (do not also put ``velocity`` in this dict — builder injects a placeholder).
    dynamics = {
        "position": velocity,
    }

    def learned_accel(x, u, node, params):
        del node, params
        # Use State/Control slices so layout stays robust to CTCS augmentation.
        xu = jnp.concatenate([x[position.slice], x[velocity.slice]])
        uu = u[force.slice]
        return dynamics_net_apply(dyn_params, xu, uu)

    byof = {"dynamics": {"velocity": learned_accel}}

    states = [position, velocity]
    controls = [force]
    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    constraints.extend([ox.ctcs(force <= force.max), ox.ctcs(force.min <= force)])
    constraints.append(
        ox.ctcs(OBSTACLE_RADIUS <= ox.linalg.Norm(position - ox.Constant(OBSTACLE_CENTER)))
    )

    # Straight-line guess (the interesting artifact here is the dynamics, not RL guess).
    tau = np.linspace(0.0, 1.0, N)
    position.guess = (1.0 - tau)[:, None] * X0[:2] + tau[:, None] * GOAL
    velocity.guess = np.zeros((N, 2))
    force.guess = np.zeros((N, 2))

    time = ox.Time(initial=0.0, final=TF, min=0.0, max=TF, uniform_time_grid=True)

    return Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=N,
        byof=byof,
        float_dtype="float64",
        algorithm={
            "lam_prox": 1e-1,
            "lam_vc": 1e2,
            "lam_vb": 1e2,
            "ep_tr": 1e-3,
            "ep_vb": 1e-4,
            "ep_vc": 1e-4,
        },
    )


def _simulate_true(u_traj: np.ndarray, x0: np.ndarray = X0) -> np.ndarray:
    """Roll OpenSCvx controls on the true plant (semi-implicit Euler)."""
    x = jnp.asarray(x0, dtype=jnp.float32)
    xs = [np.asarray(x)]
    for k in range(len(u_traj) - 1):
        x = true_step(x, jnp.asarray(u_traj[k], dtype=jnp.float32), DT)
        xs.append(np.asarray(x))
    return np.stack(xs, axis=0)


def _obstacle_penetration(pos_traj: np.ndarray) -> float:
    dist = np.linalg.norm(pos_traj - OBSTACLE_CENTER, axis=1)
    return float(np.maximum(0.0, OBSTACLE_RADIUS - dist).max())


_dyn_params = _ensure_dynamics(DYN_PATH, retrain=False)
problem = _build_problem(_dyn_params)


def plot_learned_dynamics_solution(
    results,
    true_rollout: np.ndarray | None = None,
    *,
    width: int = 640,
    height: int = 640,
):
    """Plot SCvx trajectory on the learned model (+ optional true-plant replay)."""
    import plotly.graph_objects as go

    pos = np.asarray(results.nodes["position"])
    theta = np.linspace(0.0, 2.0 * np.pi, 128)
    circle_x = OBSTACLE_CENTER[0] + OBSTACLE_RADIUS * np.cos(theta)
    circle_y = OBSTACLE_CENTER[1] + OBSTACLE_RADIUS * np.sin(theta)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            fill="toself",
            fillcolor="rgba(200, 80, 80, 0.25)",
            line={"color": "rgba(180, 60, 60, 0.8)", "width": 1},
            name="keep-out",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="lines+markers",
            name="OpenSCvx (learned dynamics)",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 5},
        )
    )
    if true_rollout is not None:
        fig.add_trace(
            go.Scatter(
                x=true_rollout[:, 0],
                y=true_rollout[:, 1],
                mode="lines+markers",
                name="true-plant replay of U*",
                line={"color": "#ff7f0e", "dash": "dash"},
                marker={"size": 4},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[X0[0], GOAL[0]],
            y=[X0[1], GOAL[1]],
            mode="markers+text",
            text=["start", "goal"],
            textposition="top center",
            marker={"size": 10, "color": ["#2ca02c", "#d62728"]},
            name="endpoints",
        )
    )
    fig.update_layout(
        width=width,
        height=height,
        title="OpenSCvx planning on RL-learned dynamics",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.retrain:
        dyn_params = _ensure_dynamics(DYN_PATH, retrain=True, updates=args.updates)
        problem_local = _build_problem(dyn_params)
    else:
        problem_local = problem

    problem_local.initialize()
    results = problem_local.solve()
    results = problem_local.post_process()

    pos = np.asarray(results.nodes["position"])
    u_star = np.asarray(results.nodes["force"])
    true_traj = _simulate_true(u_star)

    print(f"SCvx converged={results['converged']}")
    print(f"SCvx final pos={pos[-1]}  penetration={_obstacle_penetration(pos):.4f}")
    print(
        f"True-plant replay final pos={true_traj[-1, :2]}  "
        f"penetration={_obstacle_penetration(true_traj[:, :2]):.4f}  "
        f"goal error={np.linalg.norm(true_traj[-1, :2] - GOAL):.4f}"
    )

    if not args.no_show:
        plot_learned_dynamics_solution(results, true_rollout=true_traj).show()
