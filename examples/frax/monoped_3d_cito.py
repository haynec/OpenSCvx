"""3D monoped contact-implicit locomotion on flat ground (Frax + CITO BYOF).

Minimum-time hop with fixed floating-base position at start and end, free leg joints,
and contact forces / complementarity via ``CitoFraxDynamics``.

Requires:
    pip install openscvx[frax]

Set ``OPENSCVX_VISUALIZE=1`` to launch Viser after a successful solve (default off).
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import frax  # noqa: F401 — verify optional extra is installed
except ImportError:
    print(
        "frax is not installed. Install with: pip install openscvx[frax]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox
from openscvx.integrations.frax_cito import (
    MONOPED_N_DISPLAY_LINKS,
    capture_initial_multishot,
    configure_impulsive_nodes,
    estimate_normal_force_guess,
    monoped_display_chain_trajectory,
    monoped_standing_pose,
    seed_cito_initial_guess,
    sync_cito_kinematic_qd_guess,
)

# ── CITO dynamics adapter ─────────────────────────────────────────────────────
config = ox.ContactModelConfig(
    delta=1.0,
    mu=1.0,
    z_ground=0.0,
    enable_impulses=True,
    enable_cross_complementarity=True,
    enable_friction_complementarity=True,
)
dyn = ox.CitoFraxDynamics(config=config)
robot = dyn.robot

q, qd, *aux_states = dyn.states
(tau, *contact_controls) = dyn.controls

nj = robot.num_joints
na = int(tau.shape[0])
n = 40
total_time = 1.

BASE_POS = slice(0, 3)
LEG = slice(6, 8)

q_start = monoped_standing_pose(robot, foot_xy=(0.0, 0.0), z_ground=config.z_ground)
q_goal = monoped_standing_pose(robot, foot_xy=(1.0, 0.0), z_ground=config.z_ground)

def _q_boundary(values: np.ndarray, *, fix_base_position: bool, fix_leg: bool) -> list:
    bc = []
    for i in range(nj):
        if BASE_POS.start <= i < BASE_POS.stop and fix_base_position:
            bc.append(float(values[i]))
        elif LEG.start <= i < LEG.stop and fix_leg:
            bc.append(float(values[i]))
        else:
            bc.append(ox.Free(float(values[i])))
    return bc


q.initial = _q_boundary(q_start, fix_base_position=True, fix_leg=True)
q.final = _q_boundary(q_goal, fix_base_position=False, fix_leg=False)

qd.initial = np.zeros(nj)
qd.final = np.zeros(nj)

for aux in aux_states:
    aux.initial = np.zeros(1)
    aux.final = np.zeros(1)

configure_impulsive_nodes(dyn.controls, n)

constraints = []
# Box constraints on physical states/controls only; auxiliary ``y_*`` integrators are
# governed by cross-complementarity BYOF, not nodal CTCS boxes.
for state in (q, qd):
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

# Fixed horizon first — min-time cost fights contact feasibility early in SCvx.
time = ox.Time(
    initial=0.,
    final=ox.Free(total_time),
    min=0,
    max=1.5 * total_time,
    uniform_time_grid=False,
)

q_guess = seed_cito_initial_guess(
    robot,
    config,
    q,
    qd,
    aux_states,
    tau,
    contact_controls,
    q_initial=q_start,
    q_final=q_goal,
    n_nodes=n,
    rng=np.random.default_rng(0),
    time_state=time,
    total_time=total_time,
)

problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_prox": 0.5,
        "lam_vc": 1e3,
        "lam_cost": 0.0,
        "lam_vb": 1e3,
        "ep_tr": 1e-3,
        "ep_vb": 1e-3,
        "ep_vc": 1e-3,
        "k_max": 100,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
    solver={
        "cvx_solver": "CLARABEL",
        "solver_args": {},
    },
)


def configuration_from_V_multishot(
    V: np.ndarray,
    *,
    q_slice: slice,
    qd_slice: slice,
    n_x: int,
    n_u: int,
    t_nodes: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Unpack ``q`` and ``qd`` from the SCP multi-shoot matrix ``V``.

    ``V`` has shape ``((N - 1) * i4, n_substeps)`` with ``i4 = n_x + n_x² + 2·n_x·n_u``.
    Returns one configuration sample per integration substep (skipping duplicate segment joints).
    """
    if V.size == 0:
        return None, None, None
    i4 = n_x + n_x * n_x + 2 * n_x * n_u
    n_rows, n_sub = V.shape
    if i4 <= 0 or n_rows % i4 != 0 or n_sub < 1:
        return None, None, None
    n_seg = n_rows // i4
    if n_seg != len(t_nodes) - 1:
        return None, None, None

    q_rows: list[np.ndarray] = []
    qd_rows: list[np.ndarray] = []
    t_rows: list[float] = []
    for seg in range(n_seg):
        t0 = float(t_nodes[seg])
        t1 = float(t_nodes[seg + 1])
        j0 = 0 if seg == 0 else 1
        for j in range(j0, n_sub):
            alpha = j / (n_sub - 1) if n_sub > 1 else 0.0
            t_s = (1.0 - alpha) * t0 + alpha * t1
            row0 = seg * i4
            x_vec = np.asarray(V[row0 : row0 + n_x, j], dtype=np.float64).ravel()
            q_rows.append(x_vec[q_slice])
            qd_rows.append(x_vec[qd_slice])
            t_rows.append(t_s)
    if not q_rows:
        return None, None, None
    return (
        np.stack(q_rows, axis=0),
        np.stack(qd_rows, axis=0),
        np.asarray(t_rows, dtype=np.float64),
    )


def _multishot_iterate_label(iter_idx: int, n_iter: int, *, has_initial: bool) -> str:
    if has_initial and iter_idx == 0:
        return "0 — initial guess (pre-solve)"
    if iter_idx == n_iter - 1:
        return f"{iter_idx} — final"
    return f"{iter_idx} — after SCP iter {iter_idx}"


def _standing_q_on_times(
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    t_ms: np.ndarray,
    t_nodes: np.ndarray,
    *,
    z_ground: float,
) -> np.ndarray:
    """Standing poses along the straight-line foot path at each sample time."""
    foot0 = np.asarray(robot.foot_transform(q_start), dtype=float)[:3, 3]
    foot1 = np.asarray(robot.foot_transform(q_goal), dtype=float)[:3, 3]
    t0, t1 = float(t_nodes[0]), float(t_nodes[-1])
    t_ms = np.asarray(t_ms, dtype=float).reshape(-1)
    q_rows = []
    for t_s in t_ms:
        alpha = (float(t_s) - t0) / max(t1 - t0, 1e-9)
        foot_xy = (
            (1.0 - alpha) * foot0[0] + alpha * foot1[0],
            (1.0 - alpha) * foot0[1] + alpha * foot1[1],
        )
        q_rows.append(monoped_standing_pose(robot, foot_xy=foot_xy, z_ground=z_ground))
    return np.stack(q_rows, axis=0)


def _nodes_from_xu(results, x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    nodes: dict[str, np.ndarray] = {}
    for st in results._states:
        if st._slice is not None:
            nodes[st.name] = np.asarray(x[:, st._slice], dtype=float)
    for c in results._controls:
        if c._slice is not None:
            nodes[c.name] = np.asarray(u[:, c._slice], dtype=float)
    return nodes


def _time_nodes_from_x(x: np.ndarray, results) -> np.ndarray:
    time_slice = next(s._slice for s in results._states if s.name == "time")
    return np.asarray(x[:, time_slice], dtype=float).reshape(-1)


def _build_multishot_view(
    robot,
    V: np.ndarray,
    *,
    results,
    x: np.ndarray,
    u: np.ndarray,
    q_slice: slice,
    qd_slice: slice,
    n_x: int,
    n_u: int,
    t_nodes: np.ndarray,
    enable_impulses: bool,
    q_start: np.ndarray | None = None,
    q_goal: np.ndarray | None = None,
    z_ground: float = 0.0,
    use_kinematic_foot_path: bool = False,
):
    from openscvx.plotting.viser import compute_velocity_colors
    from openscvx.plotting.viser.cito_contact import cito_trajectory_on_multishot_times

    q_traj, qd_traj, t_ms = configuration_from_V_multishot(
        np.asarray(V, dtype=np.float64),
        q_slice=q_slice,
        qd_slice=qd_slice,
        n_x=n_x,
        n_u=n_u,
        t_nodes=t_nodes,
    )
    if q_traj is None or qd_traj is None or t_ms is None:
        return None

    if use_kinematic_foot_path:
        if q_start is None or q_goal is None:
            raise ValueError("q_start and q_goal required when use_kinematic_foot_path=True")
        q_traj = _standing_q_on_times(robot, q_start, q_goal, t_ms, t_nodes, z_ground=z_ground)
        # Match velocities to the reference configuration path.
        qd_traj = np.gradient(q_traj, t_ms, axis=0, edge_order=2)

    nodes = _nodes_from_xu(results, x, u)
    trajectory = cito_trajectory_on_multishot_times(q_traj, qd_traj, t_ms, nodes, t_nodes)
    keypoints, foot_pos = monoped_display_chain_trajectory(robot, q_traj)
    foot_vel = np.gradient(foot_pos, t_ms, axis=0, edge_order=2)
    foot_colors = compute_velocity_colors(foot_vel)
    return {
        "q": q_traj,
        "t": t_ms,
        "keypoints": keypoints,
        "foot_pos": foot_pos,
        "foot_colors": foot_colors,
        "trajectory": trajectory,
        "nodes": nodes,
        "node_times": np.asarray(t_nodes, dtype=float),
        "enable_impulses": enable_impulses,
    }


def visualize(
    results,
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    cito_config: ox.ContactModelConfig | None = None,
    initial_multishot: dict[str, np.ndarray] | None = None,
) -> None:
    """Animate multi-shoot integrated trajectories from ``V`` (one SCP iterate at a time)."""
    if cito_config is None:
        cito_config = config

    import threading
    import time as time_mod

    from openscvx.plotting.viser import add_target_markers, create_server
    from openscvx.plotting.viser.cito_contact import (
        add_cito_contact_visualization_view_state,
        refresh_cito_contact_view_state,
    )

    V_history = list(getattr(results, "discretization_history", None) or [])
    if not V_history and initial_multishot is None:
        print("No multi-shoot discretization history (V) — skipping Viser.", file=sys.stderr)
        return

    q_slice = next(s._slice for s in results._states if s.name == "q")
    qd_slice = next(s._slice for s in results._states if s.name == "qd")

    def _append_view(
        V: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        label: str,
        *,
        use_kinematic_foot_path: bool = False,
    ) -> None:
        t_nodes = _time_nodes_from_x(x, results)
        view = _build_multishot_view(
            robot,
            V,
            results=results,
            x=x,
            u=u,
            q_slice=q_slice,
            qd_slice=qd_slice,
            n_x=x.shape[1],
            n_u=u.shape[1],
            t_nodes=t_nodes,
            enable_impulses=cito_config.enable_impulses,
            q_start=q_start,
            q_goal=q_goal,
            z_ground=cito_config.z_ground,
            use_kinematic_foot_path=use_kinematic_foot_path,
        )
        if view is None:
            print(f"Could not decode V for {label}; skipping.", file=sys.stderr)
            return
        multishot_views.append(view)

    multishot_views: list[dict] = []
    has_initial = initial_multishot is not None
    if has_initial:
        _append_view(
            initial_multishot["V"],
            initial_multishot["x"],
            initial_multishot["u"],
            "initial guess",
        )

    n_scp = min(len(V_history), len(results.X))
    for i in range(n_scp):
        _append_view(
            V_history[i],
            np.asarray(results.X[i]),
            np.asarray(results.U[i]),
            f"SCP history {i}",
        )

    if not multishot_views:
        print("No decodable multi-shoot trajectories — skipping Viser.", file=sys.stderr)
        return

    n_segs = MONOPED_N_DISPLAY_LINKS
    foot_start = np.asarray(robot.foot_transform(q_start))[:3, 3]
    foot_goal = np.asarray(robot.foot_transform(q_goal))[:3, 3]

    view_state = {"iter_idx": 0, **multishot_views[0]}

    server = create_server(view_state["foot_pos"], show_grid=False)
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.15, axes_radius=0.004)

    add_target_markers(
        server,
        [foot_start, foot_goal],
        radius=0.02,
        colors=[(100, 150, 255), (255, 80, 80)],
    )

    ghost_colors = (view_state["foot_colors"] * 0.3).astype(np.uint8)
    ghost_handle = server.scene.add_point_cloud(
        "/multishot/ghost_traj",
        points=view_state["foot_pos"],
        colors=ghost_colors,
        point_size=0.008,
    )

    foot_xy = view_state["foot_pos"].astype(np.float32)
    if len(foot_xy) >= 2:
        foot_path_segs = np.stack(
            [np.stack([foot_xy[i], foot_xy[i + 1]], axis=0) for i in range(len(foot_xy) - 1)],
            axis=0,
        )
        foot_path_handle = server.scene.add_line_segments(
            "/multishot/foot_path",
            points=foot_path_segs,
            colors=np.array([120, 170, 255], dtype=np.uint8),
            line_width=3.0,
        )
    else:
        foot_path_handle = None

    trail_handle = server.scene.add_point_cloud(
        "/multishot/trail",
        points=view_state["foot_pos"][:1],
        colors=view_state["foot_colors"][:1],
        point_size=0.012,
    )
    marker_handle = server.scene.add_icosphere(
        "/multishot/current_foot",
        radius=0.02,
        color=(100, 200, 255),
        position=tuple(float(v) for v in view_state["foot_pos"][0]),
    )

    def update_trail(frame_idx: int) -> None:
        idx = frame_idx + 1
        trail_handle.points = view_state["foot_pos"][:idx]
        trail_handle.colors = view_state["foot_colors"][:idx]

    def update_marker(frame_idx: int) -> None:
        marker_handle.position = tuple(float(v) for v in view_state["foot_pos"][frame_idx])

    link_rgb = np.linspace([80, 100, 220], [255, 140, 60], n_segs).astype(np.uint8)
    link_colors = np.stack([link_rgb, link_rgb], axis=1)

    def _leg_segments(frame_idx: int) -> np.ndarray:
        kp = view_state["keypoints"]
        return np.stack(
            [np.stack([kp[frame_idx, k], kp[frame_idx, k + 1]]) for k in range(n_segs)]
        ).astype(np.float32)

    arm_handle = server.scene.add_line_segments(
        "/multishot/monoped_leg",
        points=_leg_segments(0),
        colors=link_colors,
        line_width=4.0,
    )

    def update_arm(frame_idx: int) -> None:
        arm_handle.points = _leg_segments(frame_idx)

    anim_state = {"playing": False, "sim_time": float(view_state["t"][0])}

    def _time_bounds() -> tuple[float, float, float]:
        t = view_state["t"].flatten()
        t_start, t_end = float(t[0]), float(t[-1])
        return t_start, t_end, max(t_end - t_start, 1e-6) / 100.0

    contact_updates = add_cito_contact_visualization_view_state(
        server,
        robot,
        view_state,
        mu=cito_config.mu,
        z_ground=cito_config.z_ground,
        enable_impulses=cito_config.enable_impulses,
    )

    def update_all(sim_t: float) -> None:
        traj_time = view_state["t"].flatten()
        n_frames = len(traj_time)
        idx = int(
            np.clip(np.searchsorted(traj_time, sim_t, side="right") - 1, 0, max(n_frames - 1, 0))
        )
        update_arm(idx)
        update_trail(idx)
        update_marker(idx)
        for cb in contact_updates:
            cb(idx)

    iterate_options = [
        _multishot_iterate_label(i, len(multishot_views), has_initial=has_initial)
        for i in range(len(multishot_views))
    ]
    with server.gui.add_folder("Multi-shoot"):
        iterate_dropdown = server.gui.add_dropdown(
            "SCP iterate",
            options=iterate_options,
            initial_value=iterate_options[0],
        )

    t_start, t_end, t_step = _time_bounds()
    with server.gui.add_folder("Animation"):
        play_button = server.gui.add_button("Play")
        reset_button = server.gui.add_button("Reset")
        time_slider = server.gui.add_slider(
            "Time (s)",
            min=t_start,
            max=t_end,
            step=t_step,
            initial_value=t_start,
        )
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    def _apply_iterate(iter_idx: int) -> None:
        iter_idx = int(np.clip(iter_idx, 0, len(multishot_views) - 1))
        view_state.update({"iter_idx": iter_idx, **multishot_views[iter_idx]})
        refresh_cito_contact_view_state(
            view_state,
            robot,
            z_ground=cito_config.z_ground,
            enable_impulses=cito_config.enable_impulses,
        )
        ghost_handle.points = view_state["foot_pos"]
        ghost_handle.colors = (view_state["foot_colors"] * 0.3).astype(np.uint8)
        if foot_path_handle is not None:
            foot_xy = view_state["foot_pos"].astype(np.float32)
            if len(foot_xy) >= 2:
                foot_path_handle.points = np.stack(
                    [np.stack([foot_xy[i], foot_xy[i + 1]], axis=0) for i in range(len(foot_xy) - 1)],
                    axis=0,
                )
        t0, _, _ = _time_bounds()
        anim_state["sim_time"] = t0
        time_slider.value = t0
        update_all(t0)

    @iterate_dropdown.on_update
    def _on_iterate_change(_) -> None:
        label = iterate_dropdown.value
        iter_idx = iterate_options.index(label)
        _apply_iterate(iter_idx)

    @play_button.on_click
    def _(_) -> None:
        anim_state["playing"] = not anim_state["playing"]
        play_button.name = "Pause" if anim_state["playing"] else "Play"

    @reset_button.on_click
    def _(_) -> None:
        t0, _, _ = _time_bounds()
        anim_state["sim_time"] = t0
        time_slider.value = t0
        update_all(t0)

    @time_slider.on_update
    def _(_) -> None:
        if not anim_state["playing"]:
            anim_state["sim_time"] = float(time_slider.value)
            update_all(anim_state["sim_time"])

    def animation_loop() -> None:
        last_time = time_mod.time()
        while True:
            time_mod.sleep(0.016)
            current = time_mod.time()
            dt = current - last_time
            last_time = current
            if not anim_state["playing"]:
                continue
            _, t_end, _ = _time_bounds()
            anim_state["sim_time"] += dt * speed_slider.value
            if anim_state["sim_time"] >= t_end:
                t0, _, _ = _time_bounds()
                if loop_checkbox.value:
                    anim_state["sim_time"] = t0
                else:
                    anim_state["sim_time"] = t_end
                    anim_state["playing"] = False
                    play_button.name = "Play"
            time_slider.value = anim_state["sim_time"]
            update_all(anim_state["sim_time"])

    threading.Thread(target=animation_loop, daemon=True).start()
    update_all(t_start)
    if has_initial and multishot_views:
        foot = multishot_views[0]["foot_pos"]
        print(
            "Initial-guess multishoot V foot path: "
            f"x∈[{foot[:, 0].min():.3f}, {foot[:, 0].max():.3f}] "
            f"max|y|={np.max(np.abs(foot[:, 1])):.4f} "
            f"max|z|={np.max(np.abs(foot[:, 2])):.4f}"
        )
    print(
        f"Viser: animating multi-shoot V trajectories ({len(multishot_views)} iterates). "
        "Iterate 0 is pre-solve V propagation; later entries are SCP history."
    )
    server.sleep_forever()


if __name__ == "__main__":
    # visualize_enabled = os.environ.get("OPENSCVX_VISUALIZE", "0").strip().lower() in (
    #     "1",
    #     "true",
    #     "yes",
    # )
    visualize_enabled = True

    print("3D monoped CITO flat-ground hop (frax)")
    print("=" * 60)
    print(f"num_joints = {nj}, num_actuated = {na}, N = {n}, delta = {config.delta}")
    phi0 = estimate_normal_force_guess(robot, q_guess[0], z_ground=config.z_ground)
    print(f"q_start pos={q_start[BASE_POS]}  phi_n_guess≈{phi0:.1f} N")
    print(f"q_goal  pos={q_goal[BASE_POS]}")
    print()

    problem.initialize()
    sync_cito_kinematic_qd_guess(problem)
    initial_multishot = capture_initial_multishot(problem)
    results = problem.solve()
    results = problem.post_process()

    # from openscvx.plotting import plot_virtual_control_heatmap
    # from openscvx.plotting import plot_states, plot_controls, plot_scp_iterations
    # plot_states(results).show()
    # plot_controls(results).show()
    # plot_virtual_control_heatmap(results).show()
    # plot_scp_iterations(results).show()

    st = problem._state
    print()
    print(f"converged: {results.converged}")
    print(f"iterations: {st.k}")
    print(f"J_tr={float(st.J_tr):.4e}  J_vb={float(st.J_vb):.4e}  J_vc={float(st.J_vc):.4e}")
    t_fin = np.asarray(results.t_final).reshape(-1)
    print(f"t_final={float(t_fin[-1]):.4f} s")

    q_final = np.asarray(results.nodes["q"][-1])
    print(f"base pos err: {np.linalg.norm(q_final[BASE_POS] - q_goal[BASE_POS]):.4e}")

    if visualize_enabled:
        print("Launching Viser...")
        # visualize(
        #     results,
        #     robot,
        #     q_start,
        #     q_goal,
        #     cito_config=config,
        #     initial_multishot=initial_multishot,
        # )
        visualize(
            results,
            robot,
            q_start,
            q_goal,
            cito_config=config,
        )

