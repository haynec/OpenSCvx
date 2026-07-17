"""Verifies the FOH-interpolation off-by-one fix (beta = tau*N -> tau*(N-1))
on the 6-DOF quadrotor obstacle-avoidance problem.

Run with no arguments from anywhere inside a checkout of this branch:

    python examples/drone/verify_foh_fix.py

It compares two codebases:
  - "after"  = THIS checkout (wherever this script currently lives -- the
               fix, since this script ships on fix/fohinterpolation_offbyone_correction)
  - "before" = origin/main's current tip, fetched and checked out into a
               throwaway git worktree at runtime, removed when done

No files are copied by hand: the "before" codebase is obtained by asking git
for it directly at run time. Each codebase is solved and verified in its own
fresh subprocess (import isolation -- JAX/OpenSCvx global state must not leak
between the two solves), using this same script re-invoked with --worker.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_FILE = Path(__file__).resolve()

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

    sol = solve_ivp(rhs, (0.0, 1.0), x[0].copy(), method="RK45",
                     rtol=1e-10, atol=1e-10, dense_output=True, max_step=0.0005)

    def violation(pos):
        return max(1.0 - (pos - c) @ A @ (pos - c) for c, A in zip(obstacle_center_positions, A_obs))

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

    # Orchestrator: "after" = this checkout; "before" = origin/main, fetched
    # into a throwaway worktree and removed when done.
    repo_root = subprocess.run(
        ["git", "-C", str(THIS_FILE.parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    subprocess.run(["git", "-C", repo_root, "fetch", "origin", "main"], check=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="foh_verify_before_"))
    tmp_dir.rmdir()  # git worktree add requires the path not exist yet
    try:
        subprocess.run(
            ["git", "-C", repo_root, "worktree", "add", "--detach", str(tmp_dir), "origin/main"],
            check=True,
        )
        before_commit = subprocess.run(
            ["git", "-C", str(tmp_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        after_commit = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()

        def run_and_parse(openscvx_root: str) -> dict:
            proc = subprocess.run(
                [sys.executable, str(THIS_FILE), "--worker", "--openscvx-root", openscvx_root],
                capture_output=True, text=True,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("WORKER_RESULT "):
                    return json.loads(line[len("WORKER_RESULT "):])
            raise RuntimeError(f"worker failed (root={openscvx_root}):\n{proc.stdout}\n{proc.stderr}")

        print(f"after  = {repo_root}  @ {after_commit}")
        after = run_and_parse(repo_root)
        print(f"before = {tmp_dir}  @ {before_commit}")
        before = run_and_parse(str(tmp_dir))
    finally:
        subprocess.run(["git", "-C", repo_root, "worktree", "remove", "--force", str(tmp_dir)],
                        check=False)

    print()
    print(f"{'':22}{'before (main)':>20}{'after (fix)':>20}")
    for key in ("converged", "iters", "J_vc", "J_tr", "true_max_violation", "final_pos_err"):
        b, a = before[key], after[key]
        b_str = f"{b:.4e}" if isinstance(b, float) else str(b)
        a_str = f"{a:.4e}" if isinstance(a, float) else str(a)
        print(f"{key:22}{b_str:>20}{a_str:>20}")


if __name__ == "__main__":
    main()
