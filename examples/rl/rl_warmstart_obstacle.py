"""RL warm-start → OpenSCvx constraint refinement (planar double integrator).

Pipeline
--------
1. Load (or train) a PureJaxRL-style PPO policy that drives a point mass from
   start → goal. The policy never sees the obstacle — only a soft goal reward.
2. Roll the policy out to obtain a state/control trajectory guess.
3. Hand that guess to OpenSCvx, which re-solves the same transfer with a *hard*
   continuous-time circular keep-out constraint via CTCS.

This is the simplest “learning + SCP” pattern: RL proposes, SCvx projects onto
the feasible set and (re)optimizes. See ``examples/rl/README.md`` for JAX RL
package recommendations and other integration patterns.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem

# Local PPO helper (same directory); keep import robust for script + pytest discovery.
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from _ppo_double_integrator import (  # noqa: E402
    A_MAX,
    DT,
    GOAL,
    HORIZON,
    X0,
    PPOConfig,
    load_params,
    rollout_policy,
    save_params,
    train_ppo,
)

# ── Geometry ─────────────────────────────────────────────────────────────────
N = HORIZON
TF = DT * (N - 1)

OBSTACLE_CENTER = np.array([0.0, 0.0])
OBSTACLE_RADIUS = 0.85  # Blocks the naive diagonal; RL (unaware) clips it.

POLICY_PATH = Path(current_dir) / "assets" / "ppo_di_policy.npz"


def _ensure_policy(path: Path = POLICY_PATH, *, retrain: bool = False, updates: int = 120):
    """Load a pretrained PPO policy, training one if missing (or requested)."""
    if path.is_file() and not retrain:
        return load_params(path)
    print(f"Training PPO policy ({updates} updates) → {path}")
    params = train_ppo(seed=0, ppo_cfg=PPOConfig(num_updates=updates))
    save_params(params, path)
    return params


def _build_problem(x_guess: np.ndarray, u_guess: np.ndarray) -> Problem:
    position = ox.State("position", shape=(2,))
    position.min = np.array([-3.5, -3.5])
    position.max = np.array([3.5, 3.5])
    position.initial = X0[:2].astype(np.float64)
    position.final = GOAL.astype(np.float64)
    position.guess = x_guess[:, :2]

    velocity = ox.State("velocity", shape=(2,))
    velocity.min = np.array([-4.0, -4.0])
    velocity.max = np.array([4.0, 4.0])
    velocity.initial = X0[2:].astype(np.float64)
    velocity.final = [ox.Free(0.0), ox.Free(0.0)]
    velocity.guess = x_guess[:, 2:]

    force = ox.Control("force", shape=(2,), parameterization="ZOH")
    force.min = np.array([-A_MAX, -A_MAX])
    force.max = np.array([A_MAX, A_MAX])
    force.guess = u_guess

    states = [position, velocity]
    controls = [force]

    dynamics = {
        "position": velocity,
        "velocity": force,
    }

    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    constraints.extend([ox.ctcs(force <= force.max), ox.ctcs(force.min <= force)])

    # Hard keep-out that the RL policy was never trained against.
    constraints.append(
        ox.ctcs(OBSTACLE_RADIUS <= ox.linalg.Norm(position - ox.Constant(OBSTACLE_CENTER)))
    )

    time = ox.Time(initial=0.0, final=TF, min=0.0, max=TF, uniform_time_grid=True)

    return Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=N,
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


def _rl_obstacle_penetration(x_traj: np.ndarray) -> float:
    """Max radial penetration of the keep-out (0 ⇒ feasible)."""
    dist = np.linalg.norm(x_traj[:, :2] - OBSTACLE_CENTER, axis=1)
    return float(np.maximum(0.0, OBSTACLE_RADIUS - dist).max())


# Module-level ``problem`` for the examples sweep / ``python -m`` import path.
_params = _ensure_policy(POLICY_PATH, retrain=False)
_X_rl, _U_rl = rollout_policy(_params, horizon=N, dt=DT, a_max=A_MAX)
problem = _build_problem(_X_rl, _U_rl)


def plot_rl_vs_scvx(results, x_rl: np.ndarray = _X_rl, *, width: int = 640, height: int = 640):
    """Overlay RL rollout, SCvx solution, and the keep-out disk."""
    import plotly.graph_objects as go

    nodes = results.nodes
    pos = np.asarray(nodes["position"])

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
            x=x_rl[:, 0],
            y=x_rl[:, 1],
            mode="lines+markers",
            name="RL rollout (warm start)",
            line={"color": "#888", "dash": "dash"},
            marker={"size": 4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="lines+markers",
            name="OpenSCvx refined",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 5},
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
        title="RL warm-start → OpenSCvx CTCS refinement",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore the shipped checkpoint and retrain PPO before solving.",
    )
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--no-show", action="store_true", help="Skip Plotly display.")
    args = parser.parse_args()

    if args.retrain:
        params = _ensure_policy(POLICY_PATH, retrain=True, updates=args.updates)
        X_rl, U_rl = rollout_policy(params, horizon=N, dt=DT, a_max=A_MAX)
        problem_local = _build_problem(X_rl, U_rl)
    else:
        X_rl, U_rl = _X_rl, _U_rl
        problem_local = problem

    print(
        f"RL rollout: final pos={X_rl[-1, :2]}, "
        f"obstacle penetration={_rl_obstacle_penetration(X_rl):.4f}"
    )

    problem_local.initialize()
    results = problem_local.solve()
    results = problem_local.post_process()

    pos = np.asarray(results.nodes["position"])
    print(f"SCvx converged={results['converged']}")
    print(
        f"SCvx final pos={pos[-1]}, "
        f"obstacle penetration={_rl_obstacle_penetration(np.hstack([pos, np.zeros((N, 2))])):.4f}"
    )

    if not args.no_show:
        plot_rl_vs_scvx(results, x_rl=X_rl).show()
