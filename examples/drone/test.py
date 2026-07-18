"""cd into before/examples/drone or after/examples/drone and run: python test.py"""

import os
import sys
import time

import jax
import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- parameters ---
LICQ_MAX = 1e-6
EP_VC = 1e-10
EP_TR = 1e-6
K_MAX = 2000
LAM_VC = 1e1  # default 1e0 was too weak to converge at tight licq_max
SOLVER_ARGS = {"abstol": 1e-9, "reltol": 1e-12}
DISCRETIZER_TOL = {"rtol": 1e-9, "atol": 1e-9}

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "obstacle_avoidance.py")
src = open(SRC, encoding="utf-8").read().split("if __name__ ==")[0]
old = (
    "problem = Problem(\n"
    "    dynamics=dynamics,\n"
    "    states=states,\n"
    "    controls=controls,\n"
    "    time=time,\n"
    "    constraints=constraints,\n"
    "    N=n,\n"
    ")"
)
src = src.replace(
    old, old.replace("N=n,\n)", f"N=n,\n    licq_max={LICQ_MAX!r},\n)")
)

ns = {"__name__": "test", "__file__": SRC}
exec(compile(src, SRC, "exec"), ns)
problem = ns["problem"]
A_obs = ns["A_obs"]
obstacle_centers = ns["obstacle_center_positions"]

problem.algorithm.ep_vc = EP_VC
problem.algorithm.ep_tr = EP_TR
problem.algorithm.k_max = K_MAX
problem.algorithm.lam_vc = LAM_VC
problem.solver.solver_args = SOLVER_ARGS
problem.discretizer.diffrax_kwargs = DISCRETIZER_TOL

problem.initialize()
t0 = time.time()
results = problem.solve()
print(
    f"licq_max={LICQ_MAX:.1e} converged={results.converged} "
    f"iters={len(results.acceptance_ratio_history)} solve_time={time.time() - t0:.2f}s"
)

state = problem._state
x, u = np.asarray(state.x), np.asarray(state.u)
N = x.shape[0]
params = problem._parameters
f_jit = jax.jit(problem._lowered.dynamics.f, static_argnums=(2,))
foh_mask = np.nan_to_num(np.asarray(problem.settings.sim.u.foh_mask), nan=1.0)
dtau = 1.0 / (N - 1)
print(f"J_vc={float(state.J_vc):.3e} J_tr={float(state.J_tr):.3e}")


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
    return max(1.0 - (pos - c) @ A @ (pos - c) for c, A in zip(obstacle_centers, A_obs))


s = np.linspace(0, 1.0, 2000)
y = sol.sol(s)
viol = np.array([violation(y[0:3, i]) for i in range(y.shape[1])])
final_err = np.linalg.norm(sol.y[0:3, -1] - np.array([-10.0, 0.0, 2.0]))

print(
    f"RESULT licq_max={LICQ_MAX:.1e} converged={results.converged} "
    f"true_max_violation={viol.max():.5f} final_pos_err={final_err:.5f}"
)
