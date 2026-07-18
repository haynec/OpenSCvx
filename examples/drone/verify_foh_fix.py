"""Verifies the FOH-interpolation off-by-one fix (beta = tau*N -> tau*(N-1))
on the 6-DOF quadrotor obstacle-avoidance problem.

Run with no arguments from anywhere inside a checkout of this branch:

    python examples/drone/verify_foh_fix.py

It compares two codebases:
  - "after"  = THIS checkout (wherever this script currently lives)
  - "before" = the exact pre-fix commit (BEFORE_COMMIT below -- the parent of
               the interpolation-fix commit, i.e. the last commit before it
               was applied), checked out into a throwaway git worktree at
               runtime and removed when done.

BEFORE_COMMIT is a fixed commit SHA, not a branch pointer, deliberately --
"origin/main" would stop being useful the moment this branch merges into
main (main would then equal "after", making the comparison vacuous). Pinning
to the exact pre-fix commit keeps this comparison meaningful forever,
regardless of where main goes afterward. That commit is an ancestor of this
branch's own history, so it's always present locally -- no network fetch
needed.

No files are copied by hand: the "before" codebase is obtained by asking git
for it directly at run time. Each codebase is solved and verified in its own
fresh subprocess (import isolation -- JAX/OpenSCvx global state must not leak
between the two solves), using this same script re-invoked with --worker.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

THIS_FILE = Path(__file__).resolve()

# Last commit before the FOH interpolation fix (parent of the fix commit).
# Pinned to an exact SHA -- see module docstring for why.
BEFORE_COMMIT = "56a68269b050d6df6d2fa9faff4807eb6ce711ae"

# --- parameters (validated in the earlier investigation) ---
LICQ_MAX = 1e-6
EP_VC = 1e-10
EP_TR = 1e-6
LAM_VC = 1e1  # default 1e0 does not converge at this licq_max within k_max
K_MAX = 2000
SOLVER_ARGS = {"abstol": 1e-9, "reltol": 1e-12}
DISCRETIZER_TOL = {"rtol": 1e-9, "atol": 1e-9}


def run_worker(openscvx_root: str) -> dict:
    sys.path.insert(0, openscvx_root)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.integrate import solve_ivp

    import openscvx as ox
    from openscvx import Problem
    from openscvx.utils import generate_orthogonal_unit_vectors

    n = 6
    total_time = 4.0

    position = ox.State("position", shape=(3,))
    position.max = np.array([200.0, 10, 20])
    position.min = np.array([-200.0, -100, 0])
    position.initial = np.array([10.0, 0, 2])
    position.final = [-10.0, 0, 2]

    velocity = ox.State("velocity", shape=(3,))
    velocity.max = np.array([100, 100, 100])
    velocity.min = np.array([-100, -100, -100])
    velocity.initial = np.array([0, 0, 0])
    velocity.final = [("free", 0), ("free", 0), ("free", 0)]

    attitude = ox.State("attitude", shape=(4,))
    attitude.max = np.array([1, 1, 1, 1])
    attitude.min = np.array([-1, -1, -1, -1])
    attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
    attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]

    angular_velocity = ox.State("angular_velocity", shape=(3,))
    angular_velocity.max = np.array([10, 10, 10])
    angular_velocity.min = np.array([-10, -10, -10])
    angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
    angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]

    thrust_force = ox.Control("thrust_force", shape=(3,))
    thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
    thrust_force.min = np.array([0, 0, 0])
    initial_control = np.array([0.0, 0.0, thrust_force.max[2]])
    thrust_force.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)

    torque = ox.Control("torque", shape=(3,))
    torque.max = np.array([18.665, 18.665, 0.55562])
    torque.min = np.array([-18.665, -18.665, -0.55562])
    torque.guess = np.zeros((n, 3))

    states = [position, velocity, attitude, angular_velocity]
    controls = [thrust_force, torque]

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
        + np.array([0, 0, g_const], dtype=np.float64),
        "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
        "angular_velocity": ox.linalg.Diag(J_b_inv)
        @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    }

    obstacle_center_positions = [
        np.array([-5.1, 0.1, 2]),
        np.array([0.1, 0.1, 2]),
        np.array([5.1, 0.1, 2]),
    ]
    obstacle_centers = [
        ox.Parameter(f"obstacle_center_{i + 1}", shape=(3,), value=c)
        for i, c in enumerate(obstacle_center_positions)
    ]

    np.random.seed(0)
    A_obs = []
    for _ in obstacle_center_positions:
        ax = generate_orthogonal_unit_vectors()
        generate_orthogonal_unit_vectors()  # matches upstream example's (unused) 2nd draw
        rad = np.random.rand(3) + 0.1 * np.ones(3)
        A_obs.append(ax @ np.diag(rad**2) @ ax.T)

    constraints = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    for center, A in zip(obstacle_centers, A_obs):
        diff = position - center
        constraints.append(ox.ctcs(1.0 <= diff.T @ A @ diff))

    time = ox.Time(initial=0.0, final=("minimize", total_time), min=0.0, max=total_time)

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=n,
        licq_max=LICQ_MAX,
    )
    problem.algorithm.ep_vc = EP_VC
    problem.algorithm.ep_tr = EP_TR
    problem.algorithm.k_max = K_MAX
    problem.algorithm.lam_vc = LAM_VC
    problem.solver.solver_args = SOLVER_ARGS
    problem.discretizer.diffrax_kwargs = DISCRETIZER_TOL

    problem.initialize()
    results = problem.solve()

    state = problem._state
    x, u = np.asarray(state.x), np.asarray(state.u)
    N = x.shape[0]
    params = problem._parameters
    f_jit = jax.jit(problem._lowered.dynamics.f, static_argnums=(2,))
    foh_mask = np.nan_to_num(np.asarray(problem.settings.sim.u.foh_mask), nan=1.0)
    dtau = 1.0 / (N - 1)

    def rhs(T, xx):
        k = min(int(np.floor(T / dtau)), N - 2)
        beta = ((T - k * dtau) * (N - 1)) * foh_mask
        return np.asarray(f_jit(xx, u[k] + beta * (u[k + 1] - u[k]), k, params))

    sol = solve_ivp(
        rhs,
        (0.0, 1.0),
        x[0].copy(),
        method="RK45",
        rtol=1e-10,
        atol=1e-10,
        dense_output=True,
        max_step=0.0005,
    )

    def violation(pos):
        return max(
            1.0 - (pos - c) @ A @ (pos - c) for c, A in zip(obstacle_center_positions, A_obs)
        )

    s = np.linspace(0, 1.0, 2000)
    y = sol.sol(s)
    viol = np.array([violation(y[0:3, i]) for i in range(y.shape[1])])
    final_err = float(np.linalg.norm(sol.y[0:3, -1] - np.array([-10.0, 0.0, 2.0])))

    return {
        "converged": bool(results.converged),
        "iters": len(results.acceptance_ratio_history),
        "J_vc": float(state.J_vc),
        "J_tr": float(state.J_tr),
        "true_max_violation": float(viol.max()),
        "final_pos_err": final_err,
        "flown_position": y[0:3, :].T.tolist(),
        "certified_nodes": x[:, 0:3].tolist(),
        "obstacle_centers": [c.tolist() for c in obstacle_center_positions],
        "obstacle_A": [A.tolist() for A in A_obs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--openscvx-root", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        result = run_worker(args.openscvx_root)
        print("WORKER_RESULT " + json.dumps(result))
        return

    # Orchestrator: "after" = this checkout; "before" = the pinned pre-fix
    # commit (BEFORE_COMMIT), checked out into a throwaway worktree and
    # removed when done. See module docstring for why this is a fixed SHA
    # rather than origin/main.
    repo_root = subprocess.run(
        ["git", "-C", str(THIS_FILE.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    verify = subprocess.run(
        ["git", "-C", repo_root, "cat-file", "-e", BEFORE_COMMIT],
        capture_output=True,
    )
    if verify.returncode != 0:
        # not present locally (e.g. a shallow clone) -- fetch it directly, no branch needed
        subprocess.run(["git", "-C", repo_root, "fetch", "origin", BEFORE_COMMIT], check=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="foh_verify_before_"))
    tmp_dir.rmdir()  # git worktree add requires the path not exist yet
    try:
        subprocess.run(
            ["git", "-C", repo_root, "worktree", "add", "--detach", str(tmp_dir), BEFORE_COMMIT],
            check=True,
        )
        before_commit = subprocess.run(
            ["git", "-C", str(tmp_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        after_commit = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        def run_and_parse(openscvx_root: str) -> dict:
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = subprocess.run(
                [sys.executable, str(THIS_FILE), "--worker", "--openscvx-root", openscvx_root],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("WORKER_RESULT "):
                    return json.loads(line[len("WORKER_RESULT ") :])
            raise RuntimeError(
                f"worker failed (root={openscvx_root}):\n{proc.stdout}\n{proc.stderr}"
            )

        print(f"after  = {repo_root}  @ {after_commit}")
        after = run_and_parse(repo_root)
        print(f"before = {tmp_dir}  @ {before_commit}")
        before = run_and_parse(str(tmp_dir))
    finally:
        subprocess.run(
            ["git", "-C", repo_root, "worktree", "remove", "--force", str(tmp_dir)], check=False
        )

    print()
    print(f"{'':22}{'before (main)':>20}{'after (fix)':>20}")
    for key in ("converged", "iters", "J_vc", "J_tr", "true_max_violation", "final_pos_err"):
        b, a = before[key], after[key]
        b_str = f"{b:.4e}" if isinstance(b, float) else str(b)
        a_str = f"{a:.4e}" if isinstance(a, float) else str(a)
        print(f"{key:22}{b_str:>20}{a_str:>20}")

    make_plots(before, after, THIS_FILE.parent)


def make_plots(before: dict, after: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    centers = before["obstacle_centers"]
    A_list = [np.array(A) for A in before["obstacle_A"]]

    def violation(pos):
        return max(
            1.0 - (np.array(pos) - np.array(c)) @ A @ (np.array(pos) - np.array(c))
            for c, A in zip(centers, A_list)
        )

    fb = np.array(before["flown_position"])
    fa = np.array(after["flown_position"])
    cb = np.array(before["certified_nodes"])
    ca = np.array(after["certified_nodes"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, title in zip(axes, ["XY (top view)", "XZ (side view)"]):
        sel = (0, 1) if "XY" in title else (0, 2)
        for center, A in zip(centers, A_list):
            Asub = A[np.ix_(list(sel), list(sel))]
            w, v = np.linalg.eigh(Asub)
            r = 1 / np.sqrt(w)
            theta = np.linspace(0, 2 * np.pi, 100)
            pts = v @ np.diag(r) @ np.array([np.cos(theta), np.sin(theta)])
            ax.fill(
                pts[0] + center[sel[0]], pts[1] + center[sel[1]], color="gray", alpha=0.4, zorder=0
            )
        ax.plot(fb[:, sel[0]], fb[:, sel[1]], "r-", lw=2, label="flown (buggy, before)")
        ax.plot(fa[:, sel[0]], fa[:, sel[1]], "b-", lw=2, label="flown (fixed, after)")
        ax.plot(cb[:, sel[0]], cb[:, sel[1]], "rx", ms=8, label="certified nodes (before)")
        ax.plot(ca[:, sel[0]], ca[:, sel[1]], "b+", ms=10, mew=2, label="certified nodes (after)")
        ax.set_title(title)
        ax.set_xlabel("xyz"[sel[0]])
        ax.set_ylabel("xyz"[sel[1]])
        ax.axis("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_comparison.png", dpi=130)
    plt.close(fig)

    vb = np.array([violation(p) for p in fb])
    worst_k = int(np.argmax(vb))
    worst_idx = int(
        np.argmax(
            [
                1.0 - (fb[worst_k] - np.array(c)) @ A @ (fb[worst_k] - np.array(c))
                for c, A in zip(centers, A_list)
            ]
        )
    )
    c = np.array(centers[worst_idx])
    A = A_list[worst_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    Asub = A[np.ix_([0, 2], [0, 2])]
    w, v = np.linalg.eigh(Asub)
    r = 1 / np.sqrt(w)
    theta = np.linspace(0, 2 * np.pi, 200)
    pts = v @ np.diag(r) @ np.array([np.cos(theta), np.sin(theta)])
    ax.fill(
        pts[0] + c[0],
        pts[1] + c[2],
        color="gray",
        alpha=0.5,
        label="obstacle boundary (XZ slice)",
        zorder=0,
    )
    margin = 3.5
    mask_b = np.abs(fb[:, 0] - c[0]) < margin
    mask_a = np.abs(fa[:, 0] - c[0]) < margin
    ax.plot(
        fb[mask_b, 0], fb[mask_b, 2], "r-", lw=2.5, label=f"flown (buggy) max viol={vb.max():.3f}"
    )
    ax.plot(fa[mask_a, 0], fa[mask_a, 2], "b-", lw=2.5, label="flown (fixed)")
    ax.plot([fb[worst_k, 0]], [fb[worst_k, 2]], "ko", ms=8, label="worst point (buggy)")
    ax.set_xlim(c[0] - margin, c[0] + margin)
    ax.set_ylim(c[2] - margin, c[2] + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Zoom on obstacle {worst_idx + 1}, XZ slice")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_zoom_xz.png", dpi=140)
    plt.close(fig)

    print(f"\nwrote {out_dir / 'trajectory_comparison.png'}")
    print(f"wrote {out_dir / 'trajectory_zoom_xz.png'}")


if __name__ == "__main__":
    main()
