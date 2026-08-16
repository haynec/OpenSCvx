"""Scaling studies for the OpenSCvx tutorial paper, based on ``drone/dr_vp``.

Three experiments on the 6-DoF drone-racing + viewpoint (LoS) problem:

1. **Nodes** — increase ``nodes_per_gate`` (nodes inserted between the fixed
   gate sequence), measuring init / solve / mean SCP step vs ``N``
2. **LoS vectorization** — increase the number of viewpoint targets, comparing
   ``ox.Vmap`` vs an explicit Python loop of scalar constraints (both CTCS)
3. **Nodal vs CTCS** — same LoS geometry, compare nodal enforcement vs
   continuous-time (CTCS) enforcement while sweeping ``N``

Usage::

    python examples/scaling/scaling_study.py
    python examples/scaling/scaling_study.py --plot-only
    python examples/scaling/scaling_study.py --study nodes
    python examples/scaling/scaling_study.py --quick

Outputs land in ``figures/scaling/`` (CSV + PDF + PNG).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import openscvx as ox
from openscvx import Problem
from openscvx.utils import rot

OUT_DIR = _ROOT / "figures" / "scaling"

C_BLUE = "#4477AA"
C_RED = "#EE6677"
C_GREEN = "#228833"
C_ORANGE = "#EE7733"
C_PURPLE = "#AA3377"

# Fixed race-course geometry (from examples/drone/dr_vp.py)
_GATE_CENTERS_RAW = [
    np.array([59.436, 0.0000, 20.0000]),
    np.array([92.964, -23.750, 25.5240]),
    np.array([92.964, -29.274, 20.0000]),
    np.array([92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([92.964, -75.080, 20.0000]),
    np.array([92.964, -68.556, 20.0000]),
    np.array([59.436, -81.358, 20.0000]),
    np.array([22.250, -42.672, 20.0000]),
]
N_GATES = len(_GATE_CENTERS_RAW)


# Dedicated CTCS group index for the LoS / viewpoint family.
# Box-bound CTCS stay on the auto-assigned group (idx=0); LoS uses idx=1 so
# a single Vmap residual feeds one constraint-integrator state.
_LOS_CTCS_IDX = 1


@dataclass
class TimingRow:
    study: str
    x: float
    x_label: str
    series: str
    build_s: float
    init_s: float
    solve_s: float
    n_iters: int
    mean_step_s: float
    n_states: int
    n_controls: int
    n_nodes: int
    n_ctcs: int
    n_ctcs_aug: int  # number of CTCS integrator states (aug dim)
    n_targets: int
    nodes_per_gate: int
    converged: bool
    # LoS violation on single-shot nonlinear propagation (post_process)
    los_viol_max: float = float("nan")
    los_viol_mean_pos: float = float("nan")
    los_viol_frac: float = float("nan")
    los_viol_l1: float = float("nan")
    init_s_std: float = 0.0
    solve_s_std: float = 0.0
    mean_step_s_std: float = 0.0
    n_trials: int = 1


# ---------------------------------------------------------------------------
# Problem builder (parameterized dr_vp)
# ---------------------------------------------------------------------------


def _gate_centers() -> list[np.ndarray]:
    """Gate centers with the same +2.5 offset as ``dr_vp.py`` (copied, not mutated)."""
    centers = []
    for raw in _GATE_CENTERS_RAW:
        c = raw.astype(float).copy()
        c[0] += 2.5
        c[2] += 2.5
        centers.append(c)
    return centers


def _target_poses(n_targets: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            [
                100.0 + rng.random() * 20.0,
                -60.0 + rng.random() * 20.0,
                20.0,
            ]
            for _ in range(n_targets)
        ],
        dtype=np.float64,
    )


def _sensor_params():
    alpha_x = 6.0
    alpha_y = 6.0
    A_cone = np.diag(
        [
            1 / np.tan(np.pi / alpha_x),
            1 / np.tan(np.pi / alpha_y),
            0,
        ]
    )
    c = jnp.array([0.0, 0.0, 1.0])
    R_sb = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return A_cone, c, R_sb, 2  # norm_type


def build_dr_vp(
    *,
    nodes_per_gate: int = 3,
    n_targets: int = 10,
    los_encoding: Literal["vmap", "loop"] = "vmap",
    los_enforcement: Literal["ctcs", "nodal"] = "ctcs",
    k_max: int = 10,
    total_time: float = 40.0,
    seed: int = 0,
) -> tuple[Problem, float, dict]:
    """Build a parameterized ``dr_vp`` instance.

    ``N = nodes_per_gate * (n_gates + 1)`` so gate nodes land at
    ``nodes_per_gate, 2*nodes_per_gate, ..., n_gates*nodes_per_gate``.

    Returns ``(problem, build_wall_time_s, meta)``.
    """
    t0 = time.perf_counter()

    n = int(nodes_per_gate * (N_GATES + 1))
    gate_centers = _gate_centers()
    gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
    assert len(gate_nodes) == N_GATES, (gate_nodes, n, nodes_per_gate)

    radii = np.array([2.5, 1e-4, 2.5])
    A_gate = rot @ np.diag(1 / radii) @ rot.T
    A_gate_cen = [A_gate @ center for center in gate_centers]
    init_poses = _target_poses(n_targets, seed=seed)
    A_cone, c_bore, R_sb, norm_type = _sensor_params()

    position = ox.State("position", shape=(3,))
    position.max = np.array([200.0, 100.0, 50.0])
    position.min = np.array([-200.0, -100.0, 15.0])
    position.initial = np.array([10.0, 0.0, 20.0])
    position.final = [10.0, 0.0, 20.0]

    velocity = ox.State("velocity", shape=(3,))
    velocity.max = np.array([100.0, 100.0, 100.0])
    velocity.min = np.array([-100.0, -100.0, -100.0])
    velocity.initial = np.array([0.0, 0.0, 0.0])
    velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

    attitude = ox.State("attitude", shape=(4,))
    attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
    attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
    attitude.initial = [("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]
    attitude.final = [("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]

    angular_velocity = ox.State("angular_velocity", shape=(3,))
    angular_velocity.max = np.array([10.0, 10.0, 10.0])
    angular_velocity.min = np.array([-10.0, -10.0, -10.0])
    angular_velocity.initial = [("free", 0.0), ("free", 0.0), ("free", 0.0)]
    angular_velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

    thrust_force = ox.Control("thrust_force", shape=(3,))
    thrust_force.max = np.array([0.0, 0.0, 4.179446268 * 9.81])
    thrust_force.min = np.array([0.0, 0.0, 0.0])
    thrust_force.guess = np.repeat(np.array([[0.0, 0.0, 10.0]]), n, axis=0)

    torque = ox.Control("torque", shape=(3,))
    torque.max = np.array([18.665, 18.665, 0.55562])
    torque.min = np.array([-18.665, -18.665, -0.55562])
    torque.guess = np.zeros((n, 3))

    states = [position, velocity, attitude, angular_velocity]
    controls = [thrust_force, torque]

    def g_vp(p_s_I, x_pos, x_quat):
        p_s_s = R_sb @ ox.spatial.QDCM(x_quat).T @ (p_s_I - x_pos)
        return ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c_bore.T @ p_s_s)

    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    # Line-of-sight / viewpoint constraints --------------------------------
    # CTCS LoS always uses idx=_LOS_CTCS_IDX so the (vectorized) residual maps
    # to a *single* constraint-integrator state, separate from box-bound CTCS.
    if los_encoding == "vmap":
        los_expr = (
            ox.Vmap(
                lambda pose: g_vp(pose, position, attitude),
                batch=init_poses,
            )
            <= 0.0
        )
        if los_enforcement == "ctcs":
            constraints.append(ox.ctcs(los_expr, idx=_LOS_CTCS_IDX))
        else:
            constraints.append(los_expr)  # nodal over all nodes
    else:
        for pose in init_poses:
            pose_c = np.asarray(pose, dtype=np.float64)
            scalar = g_vp(pose_c, position, attitude) <= 0.0
            if los_enforcement == "ctcs":
                # Same integrator idx as Vmap, but M separate symbolic CTCS
                constraints.append(ox.ctcs(scalar, idx=_LOS_CTCS_IDX))
            else:
                constraints.append(scalar)

    # Gate passage (convex nodal) -----------------------------------------
    for node, cen in zip(gate_nodes, A_gate_cen):
        gate_constraint = (
            (ox.linalg.Norm(A_gate @ position - cen, ord="inf") <= 1.0).convex().at([int(node)])
        )
        constraints.append(gate_constraint)

    # Dynamics ------------------------------------------------------------
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])
    q_norm = ox.linalg.Norm(attitude)
    attitude_normalized = attitude / q_norm
    J_b_inv = 1.0 / J_b
    J_b_diag = ox.linalg.Diag(J_b)
    dynamics = {
        "position": velocity,
        "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
        + np.array([0.0, 0.0, g_const], dtype=np.float64),
        "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
        "angular_velocity": ox.linalg.Diag(J_b_inv)
        @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    }

    # Guesses -------------------------------------------------------------
    position_bar = ox.init.linspace(
        keyframes=[np.asarray(position.initial)] + gate_centers + [np.array([10.0, 0.0, 20.0])],
        nodes=[0] + list(map(int, gate_nodes)) + [n - 1],
    )
    b = np.asarray(R_sb @ np.array([0.0, 1.0, 0.0]))
    mean_target = np.mean(init_poses, axis=0)
    attitude_bar = np.zeros((n, 4))
    for k in range(n):
        a = mean_target - position_bar[k]
        q_xyz = np.cross(b, a)
        q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
        q_no_norm = np.hstack((q_w, q_xyz))
        attitude_bar[k] = q_no_norm / la.norm(q_no_norm)
    position.guess = position_bar
    attitude.guess = attitude_bar

    time_var = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time_var,
        constraints=constraints,
        N=n,
        float_dtype="float64",
        algorithm={
            "k_max": k_max,
            "autotuner": "ConstantProximalWeight",
            "lam_prox": 1e-1,
            "lam_cost": 1e-1,
            "lam_vc": 1e0,
            "lam_vb": 4e-2,
        },
    )
    build_s = time.perf_counter() - t0
    meta = {
        "n": n,
        "nodes_per_gate": nodes_per_gate,
        "n_targets": n_targets,
        "los_encoding": los_encoding,
        "los_enforcement": los_enforcement,
        "gate_nodes": gate_nodes,
        "init_poses": init_poses,
        "A_cone": A_cone,
        "c_bore": np.asarray(c_bore),
        "R_sb": np.asarray(R_sb),
        "norm_type": norm_type,
    }
    return problem, build_s, meta


# ---------------------------------------------------------------------------
# LoS violation on single-shot propagated trajectory
# ---------------------------------------------------------------------------


def g_vp_numpy(
    targets: np.ndarray,
    positions: np.ndarray,
    attitudes: np.ndarray,
    *,
    A_cone: np.ndarray,
    c_bore: np.ndarray,
    R_sb: np.ndarray,
    norm_type: int = 2,
) -> np.ndarray:
    """Evaluate ``g_fov`` on a dense trajectory.

    Args:
        targets: ``(M, 3)`` inertial target positions.
        positions: ``(T, 3)`` drone positions.
        attitudes: ``(T, 4)`` quaternions ``[qw, qx, qy, qz]``.

    Returns:
        ``(T, M)`` residual array; positive ⇒ target outside the sensor cone.
    """
    targets = np.asarray(targets, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    attitudes = np.asarray(attitudes, dtype=np.float64)
    A_cone = np.asarray(A_cone, dtype=np.float64)
    c_bore = np.asarray(c_bore, dtype=np.float64).reshape(3)
    R_sb = np.asarray(R_sb, dtype=np.float64)

    # Match openscvx.symbolic.lowerers.jax.spatial.QDCM: scalar-first [w,x,y,z],
    # R = QDCM(q) maps body→inertial, so R.T maps inertial→body (as in dr_vp).
    q = attitudes / np.linalg.norm(attitudes, axis=1, keepdims=True)
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R_ib = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R_ib[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R_ib[:, 0, 1] = 2 * (qx * qy - qz * w)
    R_ib[:, 0, 2] = 2 * (qx * qz + qy * w)
    R_ib[:, 1, 0] = 2 * (qx * qy + qz * w)
    R_ib[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R_ib[:, 1, 2] = 2 * (qy * qz - qx * w)
    R_ib[:, 2, 0] = 2 * (qx * qz - qy * w)
    R_ib[:, 2, 1] = 2 * (qy * qz + qx * w)
    R_ib[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    R_bi = np.transpose(R_ib, (0, 2, 1))

    T = positions.shape[0]
    M = targets.shape[0]
    g = np.empty((T, M), dtype=np.float64)
    for i in range(M):
        rel_I = targets[i] - positions  # (T, 3)
        rel_B = np.einsum("tij,tj->ti", R_bi, rel_I)
        rel_S = (R_sb @ rel_B.T).T  # (T, 3)
        Ax = (A_cone @ rel_S.T).T
        if norm_type == 2:
            lhs = np.linalg.norm(Ax, axis=1)
        else:
            lhs = np.linalg.norm(Ax, ord=norm_type, axis=1)
        rhs = rel_S @ c_bore
        g[:, i] = lhs - rhs
    return g


def evaluate_los_violation(results, meta: dict) -> dict[str, float]:
    """LoS metrics on the single-shot nonlinear propagation."""
    traj = results.trajectory
    if traj is None or "position" not in traj or "attitude" not in traj:
        return {
            "los_viol_max": float("nan"),
            "los_viol_mean_pos": float("nan"),
            "los_viol_frac": float("nan"),
            "los_viol_l1": float("nan"),
        }

    g = g_vp_numpy(
        meta["init_poses"],
        np.asarray(traj["position"]),
        np.asarray(traj["attitude"]),
        A_cone=meta["A_cone"],
        c_bore=meta["c_bore"],
        R_sb=meta["R_sb"],
        norm_type=meta["norm_type"],
    )
    # Worst residual over time and targets (≤0 means satisfied)
    viol = np.maximum(g, 0.0)
    return {
        "los_viol_max": float(np.max(g)),
        "los_viol_mean_pos": float(np.mean(viol)),
        "los_viol_frac": float(np.mean(g > 0.0)),
        # Sum of positive residuals over all propagated samples and targets
        "los_viol_l1": float(np.sum(viol)),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mute_stdio():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _n_ctcs(problem: Problem) -> int:
    try:
        return len(problem._lowered.jax_constraints.ctcs)
    except Exception:
        return -1


def _n_ctcs_aug(problem: Problem) -> int:
    """Number of CTCS integrator states (augmented dimension)."""
    try:
        sl = problem.settings.sim.ctcs_slice
        if sl is None:
            return 0
        return int(sl.stop - sl.start)
    except Exception:
        return -1


def _assert_ctcs_los_single_integrator(problem: Problem, meta: dict) -> None:
    """Ensure CTCS+Vmap LoS uses a dedicated single integrator state."""
    if meta["los_enforcement"] != "ctcs" or meta["los_encoding"] != "vmap":
        return
    n_aug = _n_ctcs_aug(problem)
    # Box-bound CTCS (idx=0) + LoS Vmap CTCS (idx=1) ⇒ exactly 2 integrators
    if n_aug < 2:
        raise RuntimeError(
            f"Expected ≥2 CTCS integrator states for CTCS+Vmap LoS "
            f"(box group + dedicated LoS), got n_ctcs_aug={n_aug}"
        )


def time_problem(
    problem: Problem,
    *,
    build_s: float,
    study: str,
    x: float,
    x_label: str,
    series: str,
    n_targets: int,
    nodes_per_gate: int,
    meta: dict | None = None,
    post_process: bool = False,
) -> TimingRow:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _mute_stdio():
            problem.initialize()
            if meta is not None:
                _assert_ctcs_los_single_integrator(problem, meta)
            results = problem.solve()
            if post_process:
                results = problem.post_process()

    n_iters = len(results.X) if results.X is not None else 0
    solve_s = float(problem.timing_solve or 0.0)
    init_s = float(problem.timing_init or 0.0)
    mean_step = solve_s / n_iters if n_iters > 0 else float("nan")
    conv = bool(np.asarray(results.converged).reshape(-1)[0])

    los_metrics = {
        "los_viol_max": float("nan"),
        "los_viol_mean_pos": float("nan"),
        "los_viol_frac": float("nan"),
        "los_viol_l1": float("nan"),
    }
    if post_process and meta is not None:
        los_metrics = evaluate_los_violation(results, meta)

    return TimingRow(
        study=study,
        x=float(x),
        x_label=x_label,
        series=series,
        build_s=float(build_s),
        init_s=init_s,
        solve_s=solve_s,
        n_iters=int(n_iters),
        mean_step_s=float(mean_step),
        n_states=int(problem.settings.sim.n_states),
        n_controls=int(problem.settings.sim.n_controls),
        n_nodes=int(problem.settings.sim.n),
        n_ctcs=_n_ctcs(problem),
        n_ctcs_aug=_n_ctcs_aug(problem),
        n_targets=int(n_targets),
        nodes_per_gate=int(nodes_per_gate),
        converged=conv,
        **los_metrics,
    )


def _aggregate_trials(trials: list[TimingRow]) -> TimingRow:
    base = trials[0]
    init = np.array([t.init_s for t in trials], dtype=float)
    solve = np.array([t.solve_s for t in trials], dtype=float)
    step = np.array([t.mean_step_s for t in trials], dtype=float)
    build = np.array([t.build_s for t in trials], dtype=float)

    def _nanmean(vals):
        a = np.array(vals, dtype=float)
        return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")

    return TimingRow(
        study=base.study,
        x=base.x,
        x_label=base.x_label,
        series=base.series,
        build_s=float(build.mean()),
        init_s=float(init.mean()),
        solve_s=float(solve.mean()),
        n_iters=int(np.median([t.n_iters for t in trials])),
        mean_step_s=float(step.mean()),
        n_states=base.n_states,
        n_controls=base.n_controls,
        n_nodes=base.n_nodes,
        n_ctcs=base.n_ctcs,
        n_ctcs_aug=base.n_ctcs_aug,
        n_targets=base.n_targets,
        nodes_per_gate=base.nodes_per_gate,
        converged=all(t.converged for t in trials),
        los_viol_max=_nanmean([t.los_viol_max for t in trials]),
        los_viol_mean_pos=_nanmean([t.los_viol_mean_pos for t in trials]),
        los_viol_frac=_nanmean([t.los_viol_frac for t in trials]),
        los_viol_l1=_nanmean([t.los_viol_l1 for t in trials]),
        init_s_std=float(init.std(ddof=1)) if len(trials) > 1 else 0.0,
        solve_s_std=float(solve.std(ddof=1)) if len(trials) > 1 else 0.0,
        mean_step_s_std=float(step.std(ddof=1)) if len(trials) > 1 else 0.0,
        n_trials=len(trials),
    )


def _run_repeated(
    builder,
    *,
    study,
    x,
    x_label,
    series,
    n_targets,
    nodes_per_gate,
    n_trials,
    post_process: bool = False,
):
    trials = []
    for trial in range(n_trials):
        problem, build_s, meta = builder()
        trials.append(
            time_problem(
                problem,
                build_s=build_s,
                study=study,
                x=x,
                x_label=x_label,
                series=series,
                n_targets=n_targets,
                nodes_per_gate=nodes_per_gate,
                meta=meta,
                post_process=post_process,
            )
        )
        if n_trials > 1:
            print(f"    trial {trial + 1}/{n_trials}: init={trials[-1].init_s:.2f}s", flush=True)
    return _aggregate_trials(trials)


def run_nodes_study(
    nodes_per_gate_list: Iterable[int],
    *,
    n_targets: int = 10,
    k_max: int = 10,
    n_trials: int = 1,
) -> list[TimingRow]:
    """Sweep nodes between gates (LoS via CTCS + Vmap, fixed target count)."""
    rows: list[TimingRow] = []
    for npg in nodes_per_gate_list:
        N = npg * (N_GATES + 1)
        print(f"[nodes] nodes_per_gate={npg} → N={N} ...", flush=True)
        row = _run_repeated(
            lambda npg=npg: build_dr_vp(
                nodes_per_gate=npg,
                n_targets=n_targets,
                los_encoding="vmap",
                los_enforcement="ctcs",
                k_max=k_max,
            ),
            study="nodes",
            x=N,
            x_label="N",
            series="ctcs_vmap",
            n_targets=n_targets,
            nodes_per_gate=npg,
            n_trials=n_trials,
        )
        rows.append(row)
        print(
            f"  init={row.init_s:.2f}s  solve={row.solve_s:.3f}s  "
            f"mean_step={1e3 * row.mean_step_s:.1f}ms  "
            f"n_ctcs={row.n_ctcs} n_ctcs_aug={row.n_ctcs_aug}",
            flush=True,
        )
    return rows


def run_los_vectorization_study(
    target_counts: Iterable[int],
    *,
    nodes_per_gate: int = 3,
    k_max: int = 5,
    n_trials: int = 1,
) -> list[TimingRow]:
    """Sweep #LoS targets: Vmap vs explicit loop (both CTCS)."""
    rows: list[TimingRow] = []
    for n_t in target_counts:
        for encoding in ("vmap", "loop"):
            print(
                f"[los_vectorization] n_targets={n_t} encoding={encoding} ...",
                flush=True,
            )
            row = _run_repeated(
                lambda n_t=n_t, encoding=encoding: build_dr_vp(
                    nodes_per_gate=nodes_per_gate,
                    n_targets=n_t,
                    los_encoding=encoding,
                    los_enforcement="ctcs",
                    k_max=k_max,
                ),
                study="los_vectorization",
                x=n_t,
                x_label="n_targets",
                series=encoding,
                n_targets=n_t,
                nodes_per_gate=nodes_per_gate,
                n_trials=n_trials,
            )
            rows.append(row)
            print(
                f"  build={row.build_s:.2f}s  init={row.init_s:.2f}s  "
                f"solve={row.solve_s:.3f}s  n_ctcs={row.n_ctcs} "
                f"n_ctcs_aug={row.n_ctcs_aug}",
                flush=True,
            )
    return rows


def run_nodal_vs_ctcs_study(
    nodes_per_gate_list: Iterable[int],
    *,
    n_targets: int = 10,
    k_max: int = 30,
    n_trials: int = 1,
) -> list[TimingRow]:
    """Sweep N with LoS enforced nodally vs continuously (both Vmap).

    CTCS LoS is a single ``ox.Vmap`` residual on a dedicated integrator state
    (``idx=1``). After each solve we single-shot propagate and report continuous
    LoS violation on the dense trajectory.
    """
    rows: list[TimingRow] = []
    for npg in nodes_per_gate_list:
        N = npg * (N_GATES + 1)
        for enforcement in ("ctcs", "nodal"):
            print(
                f"[nodal_vs_ctcs] N={N} enforcement={enforcement} ...",
                flush=True,
            )
            row = _run_repeated(
                lambda npg=npg, enforcement=enforcement: build_dr_vp(
                    nodes_per_gate=npg,
                    n_targets=n_targets,
                    los_encoding="vmap",  # required: one Vmap → one integrator
                    los_enforcement=enforcement,
                    k_max=k_max,
                ),
                study="nodal_vs_ctcs",
                x=N,
                x_label="N",
                series=enforcement,
                n_targets=n_targets,
                nodes_per_gate=npg,
                n_trials=n_trials,
                post_process=True,
            )
            rows.append(row)
            print(
                f"  init={row.init_s:.2f}s  solve={row.solve_s:.3f}s  "
                f"mean_step={1e3 * row.mean_step_s:.1f}ms  "
                f"n_ctcs_aug={row.n_ctcs_aug}  "
                f"los_max={row.los_viol_max:+.3e}  "
                f"los_sum={row.los_viol_l1:.3e}  conv={row.converged}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def save_rows(rows: list[TimingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"[csv] wrote {path}")


def load_rows(path: Path) -> list[TimingRow]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:

            def _f(key, default="nan"):
                return (
                    float(raw[key]) if key in raw and raw[key] not in ("", None) else float(default)
                )

            rows.append(
                TimingRow(
                    study=raw["study"],
                    x=float(raw["x"]),
                    x_label=raw["x_label"],
                    series=raw["series"],
                    build_s=float(raw["build_s"]),
                    init_s=float(raw["init_s"]),
                    solve_s=float(raw["solve_s"]),
                    n_iters=int(raw["n_iters"]),
                    mean_step_s=float(raw["mean_step_s"]),
                    n_states=int(raw["n_states"]),
                    n_controls=int(raw["n_controls"]),
                    n_nodes=int(raw["n_nodes"]),
                    n_ctcs=int(raw["n_ctcs"]),
                    n_ctcs_aug=int(raw.get("n_ctcs_aug") or -1),
                    n_targets=int(raw.get("n_targets") or 0),
                    nodes_per_gate=int(raw.get("nodes_per_gate") or 0),
                    converged=raw["converged"].lower() in ("1", "true", "yes"),
                    los_viol_max=_f("los_viol_max"),
                    los_viol_mean_pos=_f("los_viol_mean_pos"),
                    los_viol_frac=_f("los_viol_frac"),
                    los_viol_l1=_f("los_viol_l1"),
                    init_s_std=float(raw.get("init_s_std") or 0.0),
                    solve_s_std=float(raw.get("solve_s_std") or 0.0),
                    mean_step_s_std=float(raw.get("mean_step_s_std") or 0.0),
                    n_trials=int(raw.get("n_trials") or 1),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Publication plots
# ---------------------------------------------------------------------------


def _lm_font():
    from openscvx.plotting.publication import latin_modern_fontproperties

    return latin_modern_fontproperties()


def _style_axis(ax, lm_fp) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="0.85", linewidth=0.6, linestyle="-")
    ax.grid(True, which="minor", color="0.92", linewidth=0.4, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="0.15")
    if lm_fp is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(lm_fp)


def _legend(ax, lm_fp, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=9, prop=lm_fp, **kwargs)
    if lm_fp is not None and leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(lm_fp)


def _save_fig(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, format=ext, bbox_inches="tight", facecolor="white", dpi=300)
        print(f"[plot] saved {out}")


def _rows_for(rows: list[TimingRow], study: str, series: str | None = None) -> list[TimingRow]:
    out = [r for r in rows if r.study == study]
    if series is not None:
        out = [r for r in out if r.series == series]
    return sorted(out, key=lambda r: r.x)


def plot_nodes(rows: list[TimingRow], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    lm_fp = _lm_font()
    data = _rows_for(rows, "nodes")
    if not data:
        print("[plot] no nodes data; skip")
        return

    xs = [r.x for r in data]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=120)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.errorbar(
        xs,
        [r.init_s for r in data],
        yerr=[r.init_s_std for r in data],
        fmt="o-",
        color=C_BLUE,
        lw=1.6,
        ms=5.5,
        capsize=2.5,
        label="Initialize",
    )
    ax.errorbar(
        xs,
        [r.solve_s for r in data],
        yerr=[r.solve_s_std for r in data],
        fmt="s--",
        color=C_ORANGE,
        lw=1.6,
        ms=5.5,
        capsize=2.5,
        label="Solve",
    )
    ax.set_xlabel(r"Number of nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(r"Wall-clock time (s)", fontproperties=lm_fp)
    ax.set_title("(a) Init & solve vs. nodes", fontproperties=lm_fp, fontsize=11)
    _legend(ax, lm_fp, loc="upper left")
    _style_axis(ax, lm_fp)

    ax = axes[1]
    ax.errorbar(
        xs,
        [1e3 * r.mean_step_s for r in data],
        yerr=[1e3 * r.mean_step_s_std for r in data],
        fmt="D-",
        color=C_GREEN,
        lw=1.6,
        ms=5.5,
        capsize=2.5,
        label="Mean SCP step",
    )
    # Secondary annotation: nodes_per_gate
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(r.nodes_per_gate) for r in data])
    ax2.set_xlabel(r"Nodes per gate", fontproperties=lm_fp)
    if lm_fp is not None:
        ax2.xaxis.label.set_fontproperties(lm_fp)
        for lbl in ax2.get_xticklabels():
            lbl.set_fontproperties(lm_fp)
    ax.set_xlabel(r"Number of nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(r"Mean SCP iteration (ms)", fontproperties=lm_fp)
    ax.set_title("(b) Per-iteration cost", fontproperties=lm_fp, fontsize=11)
    _legend(ax, lm_fp, loc="upper left")
    _style_axis(ax, lm_fp)

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, out_dir / "scaling_nodes")
    plt.close(fig)


def plot_los_vectorization(rows: list[TimingRow], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    lm_fp = _lm_font()
    vmap = _rows_for(rows, "los_vectorization", "vmap")
    loop = _rows_for(rows, "los_vectorization", "loop")
    if not vmap or not loop:
        print("[plot] no los_vectorization data; skip")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=120)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.errorbar(
        [r.x for r in vmap],
        [r.init_s for r in vmap],
        yerr=[r.init_s_std for r in vmap],
        fmt="o-",
        color=C_BLUE,
        lw=1.8,
        ms=5.5,
        capsize=2.5,
        label="ox.Vmap",
    )
    ax.errorbar(
        [r.x for r in loop],
        [r.init_s for r in loop],
        yerr=[r.init_s_std for r in loop],
        fmt="s--",
        color=C_RED,
        lw=1.8,
        ms=5.5,
        capsize=2.5,
        label="Python loop",
    )
    ax.set_xlabel(r"Number of LoS targets", fontproperties=lm_fp)
    ax.set_ylabel(r"Initialize time (s)", fontproperties=lm_fp)
    ax.set_title("(a) Compile / init scaling", fontproperties=lm_fp, fontsize=11)
    ax.set_yscale("log")
    _legend(ax, lm_fp, loc="upper left")
    _style_axis(ax, lm_fp)

    ax = axes[1]
    vmap_by_x = {r.x: r for r in vmap}
    xs_speed, speedups = [], []
    for r in loop:
        if r.x in vmap_by_x and vmap_by_x[r.x].init_s > 0:
            xs_speed.append(r.x)
            speedups.append(r.init_s / vmap_by_x[r.x].init_s)
    ax.plot(xs_speed, speedups, "D-", color=C_GREEN, lw=1.8, ms=5.5)
    ax.axhline(1.0, color="0.55", lw=0.9, ls=":")
    ax.set_xlabel(r"Number of LoS targets", fontproperties=lm_fp)
    ax.set_ylabel(r"Init speedup (loop / Vmap)", fontproperties=lm_fp)
    ax.set_title("(b) Vectorization speedup", fontproperties=lm_fp, fontsize=11)
    _style_axis(ax, lm_fp)

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, out_dir / "scaling_los_vectorization")
    plt.close(fig)


def plot_nodal_vs_ctcs(rows: list[TimingRow], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    lm_fp = _lm_font()
    ctcs = _rows_for(rows, "nodal_vs_ctcs", "ctcs")
    nodal = _rows_for(rows, "nodal_vs_ctcs", "nodal")
    if not ctcs or not nodal:
        print("[plot] no nodal_vs_ctcs data; skip")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.85), dpi=120)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.errorbar(
        [r.x for r in ctcs],
        [1e3 * r.mean_step_s for r in ctcs],
        yerr=[1e3 * r.mean_step_s_std for r in ctcs],
        fmt="o-",
        color=C_BLUE,
        lw=1.8,
        ms=5.5,
        capsize=2.5,
        label="CTCS (Vmap)",
    )
    ax.errorbar(
        [r.x for r in nodal],
        [1e3 * r.mean_step_s for r in nodal],
        yerr=[1e3 * r.mean_step_s_std for r in nodal],
        fmt="s--",
        color=C_PURPLE,
        lw=1.8,
        ms=5.5,
        capsize=2.5,
        label="Nodal (Vmap)",
    )
    ax.set_xlabel(r"Number of nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(r"Mean SCP iteration (ms)", fontproperties=lm_fp)
    ax.set_title("(a) Per-iteration cost", fontproperties=lm_fp, fontsize=11)
    _legend(ax, lm_fp, loc="upper left")
    _style_axis(ax, lm_fp)

    ax = axes[1]
    ax.plot(
        [r.x for r in ctcs],
        [r.los_viol_max for r in ctcs],
        "o-",
        color=C_BLUE,
        lw=1.8,
        ms=5.5,
        label="CTCS (Vmap)",
    )
    ax.plot(
        [r.x for r in nodal],
        [r.los_viol_max for r in nodal],
        "s--",
        color=C_PURPLE,
        lw=1.8,
        ms=5.5,
        label="Nodal (Vmap)",
    )
    ax.axhline(0.0, color="0.55", lw=0.9, ls=":")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel(r"Number of nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(r"Max $g_{\mathrm{fov}}$ on singleshot", fontproperties=lm_fp)
    ax.set_title("(b) Peak LoS residual (propagated)", fontproperties=lm_fp, fontsize=11)
    _legend(ax, lm_fp, loc="best")
    _style_axis(ax, lm_fp)

    ax = axes[2]
    ax.plot(
        [r.x for r in ctcs],
        [r.los_viol_l1 for r in ctcs],
        "o-",
        color=C_BLUE,
        lw=1.8,
        ms=5.5,
        label="CTCS (Vmap)",
    )
    ax.plot(
        [r.x for r in nodal],
        [r.los_viol_l1 for r in nodal],
        "s--",
        color=C_PURPLE,
        lw=1.8,
        ms=5.5,
        label="Nodal (Vmap)",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(
        r"$\sum \max(g_{\mathrm{fov}},0)$ on singleshot",
        fontproperties=lm_fp,
    )
    ax.set_title("(c) Total LoS violation", fontproperties=lm_fp, fontsize=11)
    _legend(ax, lm_fp, loc="best")
    _style_axis(ax, lm_fp)

    fig.tight_layout(w_pad=1.6)
    _save_fig(fig, out_dir / "scaling_nodal_vs_ctcs")
    plt.close(fig)


def plot_combined(rows: list[TimingRow], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    lm_fp = _lm_font()
    nodes = _rows_for(rows, "nodes")
    vmap = _rows_for(rows, "los_vectorization", "vmap")
    loop = _rows_for(rows, "los_vectorization", "loop")
    ctcs = _rows_for(rows, "nodal_vs_ctcs", "ctcs")
    nodal = _rows_for(rows, "nodal_vs_ctcs", "nodal")
    if not (nodes and vmap and loop and ctcs and nodal):
        print("[plot] incomplete data for combined figure; skip")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.85), dpi=120)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(
        [r.x for r in nodes],
        [1e3 * r.mean_step_s for r in nodes],
        "o-",
        color=C_GREEN,
        lw=1.7,
        ms=5,
    )
    ax.set_xlabel(r"Nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(r"Mean SCP iter (ms)", fontproperties=lm_fp)
    ax.set_title("(a) Nodes between gates", fontproperties=lm_fp, fontsize=11)
    _style_axis(ax, lm_fp)

    ax = axes[1]
    ax.plot(
        [r.x for r in vmap],
        [r.init_s for r in vmap],
        "o-",
        color=C_BLUE,
        lw=1.7,
        ms=5,
        label="ox.Vmap",
    )
    ax.plot(
        [r.x for r in loop],
        [r.init_s for r in loop],
        "s--",
        color=C_RED,
        lw=1.7,
        ms=5,
        label="Python loop",
    )
    ax.set_xlabel(r"LoS targets", fontproperties=lm_fp)
    ax.set_ylabel(r"Initialize time (s)", fontproperties=lm_fp)
    ax.set_title("(b) LoS vectorization", fontproperties=lm_fp, fontsize=11)
    ax.set_yscale("log")
    _legend(ax, lm_fp, loc="upper left")
    _style_axis(ax, lm_fp)

    ax = axes[2]
    ax.plot(
        [r.x for r in ctcs],
        [r.los_viol_max for r in ctcs],
        "o-",
        color=C_BLUE,
        lw=1.7,
        ms=5,
        label="CTCS (Vmap)",
    )
    ax.plot(
        [r.x for r in nodal],
        [r.los_viol_max for r in nodal],
        "s--",
        color=C_PURPLE,
        lw=1.7,
        ms=5,
        label="Nodal (Vmap)",
    )
    ax.axhline(0.0, color="0.55", lw=0.9, ls=":")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel(r"Nodes $N$", fontproperties=lm_fp)
    ax.set_ylabel(r"Max $g_{\mathrm{fov}}$ (singleshot)", fontproperties=lm_fp)
    ax.set_title("(c) CTCS vs nodal LoS residual", fontproperties=lm_fp, fontsize=11)
    _legend(ax, lm_fp, loc="best")
    _style_axis(ax, lm_fp)

    fig.tight_layout(w_pad=1.6)
    _save_fig(fig, out_dir / "scaling_combined")
    plt.close(fig)


def plot_all(rows: list[TimingRow], out_dir: Path = OUT_DIR) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
        }
    )
    plot_nodes(rows, out_dir)
    plot_los_vectorization(rows, out_dir)
    plot_nodal_vs_ctcs(rows, out_dir)
    plot_combined(rows, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--study",
        choices=("all", "nodes", "los_vectorization", "nodal_vs_ctcs"),
        default="all",
    )
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--quick", action="store_true", help="Shorter sweeps for a smoke test.")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scaling_results.csv"

    if args.plot_only:
        if not csv_path.is_file():
            raise SystemExit(f"No results at {csv_path}; run without --plot-only first.")
        rows = load_rows(csv_path)
        plot_all(rows, out_dir)
        return

    if args.quick:
        npg_list = [2, 3, 4]
        target_counts = [5, 10, 20]
        npg_nodal = [3, 4]  # N=22 is too coarse for meaningful LoS comparison
        k_nodes = 5
        k_los = 3
        k_nodal = 15  # enough iters for meaningful singleshot LoS metrics
    else:
        # N = npg * 11 → 22, 33, 44, 55, 66, 88
        npg_list = [2, 3, 4, 5, 6, 8]
        target_counts = [5, 10, 20, 40, 80]
        # Start at npg=3 (N=33): npg=2 is too coarse for this FOV cone
        npg_nodal = [3, 4, 5, 6, 8]
        k_nodes = 10
        k_los = 5
        k_nodal = 40  # longer budget so LoS residual comparison is meaningful

    n_trials = max(1, int(args.trials))
    rows: list[TimingRow] = []
    if args.study in ("all", "nodes"):
        rows.extend(run_nodes_study(npg_list, k_max=k_nodes, n_trials=n_trials))
    if args.study in ("all", "los_vectorization"):
        rows.extend(run_los_vectorization_study(target_counts, k_max=k_los, n_trials=n_trials))
    if args.study in ("all", "nodal_vs_ctcs"):
        rows.extend(run_nodal_vs_ctcs_study(npg_nodal, k_max=k_nodal, n_trials=n_trials))

    if args.study != "all" and csv_path.is_file():
        existing = [r for r in load_rows(csv_path) if r.study != args.study]
        # Drop legacy double-integrator studies if present
        legacy = {"state_dim", "constraints"}
        existing = [r for r in existing if r.study not in legacy]
        rows = existing + rows

    save_rows(rows, csv_path)
    plot_all(rows, out_dir)
    print(f"\nDone. Results and figures in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
