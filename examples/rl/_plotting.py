"""Static matplotlib plots for the RL examples (mobile / CI friendly PNGs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _draw_keepout(ax, center, radius: float) -> None:
    disk = plt.Circle(
        center,
        radius,
        facecolor=(0.78, 0.31, 0.31, 0.28),
        edgecolor=(0.70, 0.24, 0.24, 0.85),
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(disk)


def plot_warmstart_png(
    x_rl: np.ndarray,
    x_scvx: np.ndarray,
    *,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    out: Path,
    title: str = "RL warm-start → OpenSCvx CTCS refinement",
) -> Path:
    """Save RL rollout vs SCvx refinement overlay."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=160)
    _draw_keepout(ax, obstacle_center, obstacle_radius)
    ax.plot(
        x_rl[:, 0],
        x_rl[:, 1],
        color="#777777",
        linestyle="--",
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="RL rollout (warm start)",
        zorder=2,
    )
    ax.plot(
        x_scvx[:, 0],
        x_scvx[:, 1],
        color="#1f77b4",
        marker="o",
        markersize=4.0,
        linewidth=2.4,
        label="OpenSCvx refined",
        zorder=3,
    )
    ax.scatter([start[0]], [start[1]], s=70, c="#2ca02c", zorder=4, label="start")
    ax.scatter([goal[0]], [goal[1]], s=70, c="#d62728", zorder=4, label="goal")
    ax.annotate("start", start + np.array([-0.15, 0.18]), fontsize=9, color="#2ca02c")
    ax.annotate("goal", goal + np.array([-0.1, 0.18]), fontsize=9, color="#d62728")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_learned_dynamics_png(
    x_scvx: np.ndarray,
    x_true_replay: np.ndarray,
    *,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    out: Path,
    title: str = "OpenSCvx on RL-learned dynamics",
) -> Path:
    """Save SCvx (learned model) vs true-plant replay overlay."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=160)
    _draw_keepout(ax, obstacle_center, obstacle_radius)
    ax.plot(
        x_scvx[:, 0],
        x_scvx[:, 1],
        color="#1f77b4",
        marker="o",
        markersize=4.0,
        linewidth=2.4,
        label="OpenSCvx (learned dynamics)",
        zorder=3,
    )
    ax.plot(
        x_true_replay[:, 0],
        x_true_replay[:, 1],
        color="#ff7f0e",
        linestyle="--",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="true-plant replay of U*",
        zorder=2,
    )
    ax.scatter([start[0]], [start[1]], s=70, c="#2ca02c", zorder=4, label="start")
    ax.scatter([goal[0]], [goal[1]], s=70, c="#d62728", zorder=4, label="goal")
    ax.annotate("start", start + np.array([-0.15, 0.18]), fontsize=9, color="#2ca02c")
    ax.annotate("goal", goal + np.array([-0.1, 0.18]), fontsize=9, color="#d62728")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_controls_png(
    t: np.ndarray,
    u: np.ndarray,
    *,
    out: Path,
    title: str = "Optimized controls",
    labels: tuple[str, str] = ("u_x", "u_y"),
) -> Path:
    """Save control time histories."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    ax.step(t, u[:, 0], where="post", label=labels[0], color="#1f77b4")
    ax.step(t, u[:, 1], where="post", label=labels[1], color="#ff7f0e")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("acceleration")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
