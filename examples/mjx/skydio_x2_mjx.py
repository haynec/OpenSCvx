"""Skydio X2 quadrotor racing through sequential gates with MuJoCo MJX dynamics.

This mirrors the scenario structure in `examples/drone/drone_racing.py`:
- sequential gate traversal constraints at prescribed nodes
- minimum-time objective
- loop closure in position (start equals end)

Model loading (in priority order)
----------------------------------
1. **MuJoCo Menagerie submodule** — if ``third_party/mujoco_menagerie`` is
   present (``git submodule update --init third_party/mujoco_menagerie``),
   the official Skydio X2 model is loaded from there, complete with the
   low-poly mesh and texture used for Viser visualisation.
2. **Inline XML fallback** — if the submodule is absent the example uses a
   self-contained inline XML that matches the menagerie physics (same rotor
   positions, gear vectors, masses, hover thrust) but replaces mesh assets
   with primitive geoms so the example works with no extra files.

Visualisation
-------------
Uses ``examples.plotting_viser.create_animated_plotting_server`` — same layout
as ``examples/drone/drone_racing.py`` (gates, ghost path, thrust vector, controls).
When the menagerie asset and ``trimesh`` are available, pass the Skydio X2
low-poly mesh so the drone body is drawn instead of the default attitude axes.

Requires:
    pip install openscvx[mjx]
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import mujoco
    import mujoco.mjx as mjx
except ImportError:
    print(
        "MuJoCo MJX is not installed. Install with: pip install openscvx[mjx]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
)
from openscvx import ByofSpec, Problem
from openscvx.integrations import mjx_byof
from openscvx.utils import gen_vertices, rot

HOVER_CTRL = 3.2495625  # N per motor for level hover (from menagerie keyframe)
START_POS = np.array([10.0, 0.0, 20.0])
HOVER_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # w=1 → level attitude

# ── Load MuJoCo model — try menagerie first, fall back to inline XML ──────────
_menagerie_xml_path: "str | None" = None
from openscvx.integrations.menagerie import get_xml_path

_menagerie_xml_path = str(get_xml_path("skydio_x2"))
mj_model = mujoco.MjModel.from_xml_path(_menagerie_xml_path)
print(f"[skydio_x2] loaded from MuJoCo Menagerie: {_menagerie_xml_path}")
# Disable contact solver: MJX's contact pipeline uses lax.while_loop which is
# not forward-mode differentiable. Quadrotors don't rely on contact dynamics.
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)

n_q = int(mjx_model.nq)  # 7 — xyz + quaternion (free joint)
n_v = int(mjx_model.nv)  # 6 — linear + angular velocity
n_u = int(mjx_model.nu)  # 4 — rotor thrusts

n = 22
total_time = 24.0

# ── State / control definitions ───────────────────────────────────────────────
qpos = ox.State("qpos", shape=(n_q,))
qpos.min = np.array([-200.0, -100.0, 15.0, -1.0, -1.0, -1.0, -1.0])
qpos.max = np.array([200.0, 100.0, 200.0, 1.0, 1.0, 1.0, 1.0])
qpos.initial = np.concatenate([START_POS, HOVER_QUAT])
qpos.final = [10.0, 0.0, 20.0, ("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]

qvel = ox.State("qvel", shape=(n_v,))
qvel.min = np.array([-100.0, -100.0, -100.0, -10.0, -10.0, -10.0])
qvel.max = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
qvel.initial = np.zeros(n_v)
qvel.final = [("free", 0.0)] * n_v

ctrl = ox.Control("ctrl", shape=(n_u,))
ctrl.min = np.zeros(n_u)
ctrl.max = 13.0 * np.ones(n_u)
ctrl.guess = HOVER_CTRL * np.ones((n, n_u))

states = [qpos, qvel]
controls = [ctrl]

# ── Dynamics via BYOF ─────────────────────────────────────────────────────────
# The free joint has nq=7 but nv=6 (quaternion adds one extra position DOF).
# nq=7, nv=6 (free joint): mjx_byof detects nq > nv and automatically
# includes quaternion kinematics for "qpos" alongside the MJX "qvel" dynamics.
byof: ByofSpec = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

# ── Gate parameters (matching examples/drone/drone_racing.py) ───────────────
n_gates = 10
initial_gate_centers = [
    np.array([59.436, 0.000, 20.0000]),
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
radii = np.array([2.5, 1e-4, 2.5])
A_gate_const = rot @ np.diag(1 / radii) @ rot.T

modified_centers = []
for center in initial_gate_centers:
    modified_center = center.copy()
    modified_center[0] = modified_center[0] + 2.5
    modified_center[2] = modified_center[2] + 2.5
    modified_centers.append(modified_center)

nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
gate_centers = np.array(modified_centers)

# ── Constraints ───────────────────────────────────────────────────────────────
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

# Enforce sequential gate traversal using nodal constraints on qpos position.
position = ox.Concat(qpos[0], qpos[1], qpos[2])
for node, gate_center in zip(gate_nodes, gate_centers):
    constraints.append(
        (
            ox.linalg.Norm(
                A_gate_const @ position - A_gate_const @ ox.Constant(gate_center),
                ord="inf",
            )
            <= 1.0
        )
        .convex()
        .at([node])
    )

# ── Initial guess: piecewise-linear through all gates, level attitude ─────────
pos_guess = ox.init.linspace(
    keyframes=[START_POS] + modified_centers + [START_POS],
    nodes=[0] + list(gate_nodes) + [n - 1],
)
quat_guess = np.tile(HOVER_QUAT, (n, 1))
qpos.guess = np.column_stack([pos_guess, quat_guess])
qvel.guess = np.zeros((n, n_v))

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics={},  # all dynamics go through BYOF
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    byof=byof,
    algorithm={
        "lam_prox": 1e-1,
        "lam_cost": 1e-2,
        "lam_vc": 1e1,
        # "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
)


def load_skydio_x2_vehicle_mesh() -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(vertices, faces)`` for Viser, or ``None`` to use default attitude axes.

    Matches MuJoCo Menagerie visual geom: scale 0.01 and visual euler rotation.
    """
    if _menagerie_xml_path is None:
        return None
    try:
        from pathlib import Path

        import trimesh  # type: ignore

        asset_dir = Path(_menagerie_xml_path).parent / "assets"
        obj_path = asset_dir / "X2_lowpoly.obj"
        tm = trimesh.load(obj_path, force="mesh", process=False)
        tm.apply_scale(0.01)
        # Visual geom quat="0 0 1 1" (MuJoCo [w,x,y,z]) → fixed rotation matrix
        r_vis = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        tm.vertices = (tm.vertices @ r_vis.T).astype(np.float32)
        verts = np.asarray(tm.vertices, dtype=np.float32)
        faces = np.asarray(tm.faces, dtype=np.uint32)
        return verts, faces
    except Exception:
        return None


if __name__ == "__main__":
    print("Skydio X2 drone racing — MuJoCo MJX + OpenSCvx")
    print("=" * 60)
    print(f"nq={n_q}, nv={n_v}, nu={n_u}, N={n}")
    print(f"Start/Finish: {START_POS}")
    print(f"Gates: {n_gates} sequential constraints")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_pos = results.nodes["qpos"][-1, :3]
    final_vel = results.nodes["qvel"][-1]
    pos_err = np.linalg.norm(final_pos - START_POS)

    print()
    print(f"Final position: {final_pos}")
    print(f"Loop-closure position error: {pos_err:.4f} m")
    print(f"Final velocity:  {np.linalg.norm(final_vel):.4f} m/s")
    print()

    # ── Viser: same template as examples/drone/drone_racing.py ────────────────
    traj = results.trajectory
    traj["position"] = np.asarray(traj["qpos"][:, :3], dtype=np.float64)
    traj["velocity"] = np.asarray(traj["qvel"][:, :3], dtype=np.float64)
    traj["attitude"] = np.asarray(traj["qpos"][:, 3:7], dtype=np.float64)
    ctrl_tr = np.asarray(traj["ctrl"], dtype=np.float64)
    thrust_body = np.zeros((ctrl_tr.shape[0], 3), dtype=np.float64)
    thrust_body[:, 2] = np.sum(ctrl_tr, axis=1)
    traj["thrust_force"] = thrust_body

    gate_vertices = [gen_vertices(center, radii) for center in modified_centers]
    results.update(
        {
            "vertices": gate_vertices,
            "gate_centers": modified_centers,
            "A_gate": A_gate_const,
            "A_gate_c_params": [A_gate_const @ np.asarray(c) for c in modified_centers],
        }
    )

    vehicle_mesh = load_skydio_x2_vehicle_mesh()
    if vehicle_mesh is not None:
        print("[viser] vehicle_mesh: Skydio X2 low-poly (menagerie assets)")
    else:
        print("[viser] vehicle_mesh: None — default axes (see load_skydio_x2_vehicle_mesh)")

    traj_server = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        viewcone_scale=10.0,
        show_control_plot="ctrl",
        show_control_norm_plot="ctrl",
        vehicle_mesh=vehicle_mesh,
    )
    scp_server = create_scp_animated_plotting_server(
        results,
        position_slice=slice(0, 3),
        attitude_slice=slice(3, 7),
    )
    traj_server.sleep_forever()
