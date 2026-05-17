"""Interactive real-time visualization for 6-DoF powered descent guidance."""

import importlib.util
import os
import sys
import threading
import time

import matplotlib
import numpy as np
import viser

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

_base_path = os.path.join(current_dir, "base_problems", "6DoF_pdg_realtime_base.py")
_spec = importlib.util.spec_from_file_location("pdg6dof_realtime_base", _base_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load 6DoF PDG realtime base module: {_base_path}")
pdg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdg)

from examples.plotting_viser import (
    build_scp_step_results,
    compute_velocity_colors_realtime,
    extract_multishoot_trajectory,
    format_metrics_markdown,
    get_print_queue_data,
)

_viridis_cmap = matplotlib.colormaps["viridis"]
VISER_SCENE_SCALE = 2.0

pdg.initialize_problem_quiet(pdg.problem)
pdg.print_scp_table_header_once(pdg.problem)


def _viser_tuple_from_model(p: np.ndarray) -> tuple[float, float, float]:
    v = pdg.model_vec_to_viser_xyz(np.asarray(p, dtype=np.float64).reshape(3)) * VISER_SCENE_SCALE
    return (float(v[0]), float(v[1]), float(v[2]))


def _model_vec_from_viser(triple) -> np.ndarray:
    """Invert ``_viser_tuple_from_model`` (same linear map)."""
    return pdg.model_vec_to_viser_xyz(np.asarray(triple, dtype=np.float64) / VISER_SCENE_SCALE)


def _target_viser_on_grid_plane(fx: float, fy: float) -> tuple[float, float, float]:
    """Viser ``add_grid`` is XY at z=0.

    Horizontal axes are swapped vs naive (fx,fy)→(fx,fy) so they match ``model_vec_to_viser_xyz``:
    model ``(fx, fy, 0)`` maps to Viser ``(0, fy*s, fx*s)``, so grid-plane motion uses
    ``vx = fy*s``, ``vy = fx*s``.
    """
    s = VISER_SCENE_SCALE
    return (float(fy) * s, float(fx) * s, 0.0)


def _cone_half_angle_deg_from_gamma(gamma_deg: float) -> float:
    """Match ``0.1 * ||(y,z)|| <= tan(gamma) * x`` → half-angle ≈ atan(10 tan γ)."""
    g = np.radians(float(gamma_deg))
    return float(np.degrees(np.arctan(10.0 * np.tan(g))))


def _generate_cone_mesh(
    apex: np.ndarray,
    height: float,
    half_angle_deg: float,
    n_segments: int = 32,
    axis: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32),
) -> tuple[np.ndarray, np.ndarray]:
    half_angle_rad = np.radians(half_angle_deg)
    base_radius = height * np.tan(half_angle_rad)

    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(axis[2]) < 0.9
        else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    u = ref - np.dot(ref, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    vertices = [apex.copy()]
    base_center = apex + height * axis
    for i in range(n_segments):
        angle = 2.0 * np.pi * i / n_segments
        offset = base_radius * (np.cos(angle) * u + np.sin(angle) * v)
        vertices.append(base_center + offset)
    vertices.append(base_center.copy())
    vertices = np.array(vertices, dtype=np.float32)

    faces = []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([0, i + 1, next_i + 1])
    base_center_idx = n_segments + 1
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([base_center_idx, next_i + 1, i + 1])
    return vertices, np.array(faces, dtype=np.int32)


def _sync_problem_parameters_from_base() -> None:
    p = pdg.problem
    p.parameters["gI"] = float(pdg.gI.value)
    p.parameters["l"] = float(pdg.l_arm.value)
    p.parameters["J_diag"] = np.asarray(pdg.J_diag.value, dtype=np.float64).reshape(3)
    p.parameters["g0"] = float(pdg.g0.value)
    p.parameters["Isp"] = float(pdg.Isp.value)
    p.parameters["m_dry"] = float(pdg.m_dry.value)
    p.parameters["v_max"] = float(pdg.v_max.value)
    p.parameters["w_max"] = float(pdg.w_max.value)
    p.parameters["del_max"] = float(pdg.del_max.value)
    p.parameters["theta_max"] = float(pdg.theta_max.value)
    p.parameters["T_min"] = float(pdg.T_min.value)
    p.parameters["T_max"] = float(pdg.T_max.value)
    p.parameters["gamma"] = float(pdg.gamma.value)
    p.parameters["beta"] = float(pdg.beta.value)
    p.parameters["c_ax"] = float(pdg.c_ax.value)
    p.parameters["c_ayz"] = float(pdg.c_ayz.value)
    p.parameters["S_a"] = float(pdg.S_a.value)
    p.parameters["rho"] = float(pdg.rho.value)
    p.parameters["l_p"] = float(pdg.l_p.value)
    p.parameters["initial_position"] = np.asarray(
        pdg.initial_position.value, dtype=np.float64
    ).reshape(3)
    p.parameters["final_position"] = np.asarray(pdg.final_position.value, dtype=np.float64).reshape(
        2
    )


def create_realtime_server(optimization_problem) -> viser.ViserServer:
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    def _set_param(name: str, val) -> None:
        optimization_problem.parameters[name] = val

    server.scene.add_grid(
        "/grid",
        width=30.0 * VISER_SCENE_SCALE,
        height=30.0 * VISER_SCENE_SCALE,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=8.0 * VISER_SCENE_SCALE,
    )

    trajectory_handle = server.scene.add_point_cloud(
        "/trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 255, 0),
        point_size=(12.0 / 200.0) * VISER_SCENE_SCALE,
    )
    init_pos = np.asarray(pdg.initial_position.value, dtype=np.float64)
    start_handle = server.scene.add_icosphere(
        "/start",
        radius=0.35 * VISER_SCENE_SCALE,
        color=(100, 255, 100),
        position=_viser_tuple_from_model(init_pos),
    )
    _final_model = np.array(
        [float(pdg.final_position.value[0]), float(pdg.final_position.value[1]), 0.0],
        dtype=np.float64,
    )
    target_handle = server.scene.add_icosphere(
        "/target",
        radius=0.35 * VISER_SCENE_SCALE,
        color=(255, 100, 100),
        position=_target_viser_on_grid_plane(float(_final_model[0]), float(_final_model[1])),
    )
    start_drag = server.scene.add_transform_controls(
        "/start_drag",
        position=_viser_tuple_from_model(init_pos),
        scale=0.4 * VISER_SCENE_SCALE,
        disable_rotations=True,
        visible=True,
    )
    target_drag = server.scene.add_transform_controls(
        "/target_drag",
        position=target_handle.position,
        scale=0.4 * VISER_SCENE_SCALE,
        disable_rotations=True,
        visible=True,
    )

    def _cone_height_scaled() -> float:
        ix = float(np.asarray(pdg.initial_position.value, dtype=np.float64)[0])
        return max(abs(ix), 1.0) * VISER_SCENE_SCALE

    _axis_plus_x_model = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    _axv = pdg.model_vec_to_viser_xyz(_axis_plus_x_model).astype(np.float64)
    _cone_axis_vis = (_axv / np.linalg.norm(_axv)).astype(np.float32)

    def _update_glideslope_cone() -> None:
        apex = np.array(
            _target_viser_on_grid_plane(
                float(pdg.final_position.value[0]), float(pdg.final_position.value[1])
            ),
            dtype=np.float32,
        )
        half_deg = _cone_half_angle_deg_from_gamma(float(pdg.gamma.value))
        cone_vertices, cone_faces = _generate_cone_mesh(
            apex=apex,
            height=_cone_height_scaled(),
            half_angle_deg=half_deg,
            n_segments=48,
            axis=_cone_axis_vis,
        )
        glideslope_cone_handle.vertices = cone_vertices
        glideslope_cone_handle.faces = cone_faces

    apex0 = np.array(
        _target_viser_on_grid_plane(
            float(pdg.final_position.value[0]), float(pdg.final_position.value[1])
        ),
        dtype=np.float32,
    )
    cone_vertices, cone_faces = _generate_cone_mesh(
        apex=apex0,
        height=_cone_height_scaled(),
        half_angle_deg=_cone_half_angle_deg_from_gamma(float(pdg.gamma.value)),
        n_segments=48,
        axis=_cone_axis_vis,
    )
    glideslope_cone_handle = server.scene.add_mesh_simple(
        "/constraints/glideslope_cone",
        vertices=cone_vertices,
        faces=cone_faces,
        color=(80, 200, 120),
        wireframe=False,
        opacity=0.2,
    )

    state = {"running": True, "reset_requested": False}

    with server.gui.add_folder("Optimization Metrics"):
        metrics_text = server.gui.add_markdown(
            format_metrics_markdown(
                {
                    "iter": 0,
                    "J_tr": 0.0,
                    "J_vb": 0.0,
                    "J_vc": 0.0,
                    "cost": 0.0,
                    "dis_time": 0.0,
                    "solve_time": 0.0,
                    "prob_stat": "--",
                }
            )
        )

    with server.gui.add_folder("Problem Control", expand_by_default=True):
        reset_button = server.gui.add_button("Apply Changes + Reset Problem")

        @reset_button.on_click
        def _(_) -> None:
            state["reset_requested"] = True

    with server.gui.add_folder("Algorithm Weights"):
        lam_cost = server.gui.add_number(
            "lam_cost",
            initial_value=optimization_problem.algorithm.lam_cost,
            min=1e-8,
            max=1e5,
            step=0.01,
        )
        lam_vc = server.gui.add_number(
            "lam_vc",
            initial_value=optimization_problem.algorithm.lam_vc,
            min=1e-8,
            max=1e5,
            step=0.01,
        )
        lam_prox = server.gui.add_number(
            "lam_prox",
            initial_value=optimization_problem.algorithm.lam_prox,
            min=1e-8,
            max=1e5,
            step=0.01,
        )

        @lam_cost.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_cost = float(lam_cost.value)

        @lam_vc.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_vc = float(lam_vc.value)

        @lam_prox.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_prox = float(lam_prox.value)

    with server.gui.add_folder("Dynamics / Constraint Parameters"):
        gI_in = server.gui.add_number(
            "gI", initial_value=float(pdg.gI.value), min=0.01, max=20.0, step=0.01
        )
        g0_in = server.gui.add_number(
            "g0", initial_value=float(pdg.g0.value), min=0.01, max=20.0, step=0.01
        )
        isp_in = server.gui.add_number(
            "Isp", initial_value=float(pdg.Isp.value), min=1.0, max=500.0, step=1.0
        )
        m_dry_in = server.gui.add_number(
            "m_dry", initial_value=float(pdg.m_dry.value), min=0.5, max=2.0, step=0.01
        )
        v_max_in = server.gui.add_number(
            "v_max", initial_value=float(pdg.v_max.value), min=0.1, max=20.0, step=0.05
        )
        w_max_in = server.gui.add_number(
            "w_max", initial_value=float(pdg.w_max.value), min=0.01, max=5.0, step=0.01
        )
        del_max_in = server.gui.add_number(
            "del_max (deg)", initial_value=float(pdg.del_max.value), min=1.0, max=89.0, step=0.5
        )
        theta_max_in = server.gui.add_number(
            "theta_max (deg)", initial_value=float(pdg.theta_max.value), min=1.0, max=89.0, step=0.5
        )
        t_min_in = server.gui.add_number(
            "T_min", initial_value=float(pdg.T_min.value), min=0.1, max=20.0, step=0.1
        )
        t_max_in = server.gui.add_number(
            "T_max", initial_value=float(pdg.T_max.value), min=0.1, max=20.0, step=0.1
        )
        gamma_in = server.gui.add_number(
            "gamma (deg)", initial_value=float(pdg.gamma.value), min=1.0, max=89.0, step=0.5
        )
        beta_in = server.gui.add_number(
            "beta", initial_value=float(pdg.beta.value), min=0.0, max=1.0, step=0.001
        )
        c_ax_in = server.gui.add_number(
            "c_ax", initial_value=float(pdg.c_ax.value), min=0.0, max=5.0, step=0.05
        )
        c_ayz_in = server.gui.add_number(
            "c_ayz", initial_value=float(pdg.c_ayz.value), min=0.0, max=5.0, step=0.05
        )
        s_a_in = server.gui.add_number(
            "S_a", initial_value=float(pdg.S_a.value), min=0.0, max=5.0, step=0.05
        )
        rho_in = server.gui.add_number(
            "rho", initial_value=float(pdg.rho.value), min=0.0, max=5.0, step=0.05
        )
        l_arm_in = server.gui.add_number(
            "l (arm)", initial_value=float(pdg.l_arm.value), min=0.01, max=2.0, step=0.01
        )
        l_p_in = server.gui.add_number(
            "l_p", initial_value=float(pdg.l_p.value), min=0.0, max=1.0, step=0.01
        )
        j_diag = np.asarray(pdg.J_diag.value, dtype=np.float64).reshape(3)
        j0 = server.gui.add_number(
            "J_diag[0]", initial_value=float(j_diag[0]), min=1e-6, max=1.0, step=1e-4
        )
        j1 = server.gui.add_number(
            "J_diag[1]", initial_value=float(j_diag[1]), min=1e-6, max=1.0, step=1e-4
        )
        j2 = server.gui.add_number(
            "J_diag[2]", initial_value=float(j_diag[2]), min=1e-6, max=1.0, step=1e-4
        )
        initial_pos_input = server.gui.add_vector3(
            "initial_position",
            initial_value=tuple(np.asarray(pdg.initial_position.value, dtype=np.float64)),
            step=0.1,
        )
        final_xy_input = server.gui.add_vector2(
            "final_position [x,y]",
            initial_value=tuple(np.asarray(pdg.final_position.value, dtype=np.float64)),
            step=0.1,
        )

        @gI_in.on_update
        def _(_) -> None:
            pdg.gI.value = float(gI_in.value)
            _set_param("gI", float(gI_in.value))

        @g0_in.on_update
        def _(_) -> None:
            pdg.g0.value = float(g0_in.value)
            _set_param("g0", float(g0_in.value))

        @isp_in.on_update
        def _(_) -> None:
            pdg.Isp.value = float(isp_in.value)
            _set_param("Isp", float(isp_in.value))

        @m_dry_in.on_update
        def _(_) -> None:
            pdg.m_dry.value = float(m_dry_in.value)
            _set_param("m_dry", float(m_dry_in.value))

        @v_max_in.on_update
        def _(_) -> None:
            pdg.v_max.value = float(v_max_in.value)
            _set_param("v_max", float(v_max_in.value))

        @w_max_in.on_update
        def _(_) -> None:
            pdg.w_max.value = float(w_max_in.value)
            _set_param("w_max", float(w_max_in.value))

        @del_max_in.on_update
        def _(_) -> None:
            pdg.del_max.value = float(del_max_in.value)
            _set_param("del_max", float(del_max_in.value))

        @theta_max_in.on_update
        def _(_) -> None:
            pdg.theta_max.value = float(theta_max_in.value)
            _set_param("theta_max", float(theta_max_in.value))

        @t_min_in.on_update
        def _(_) -> None:
            pdg.T_min.value = float(t_min_in.value)
            _set_param("T_min", float(t_min_in.value))

        @t_max_in.on_update
        def _(_) -> None:
            pdg.T_max.value = float(t_max_in.value)
            _set_param("T_max", float(t_max_in.value))

        @gamma_in.on_update
        def _(_) -> None:
            pdg.gamma.value = float(gamma_in.value)
            _set_param("gamma", float(gamma_in.value))
            _update_glideslope_cone()

        @beta_in.on_update
        def _(_) -> None:
            pdg.beta.value = float(beta_in.value)
            _set_param("beta", float(beta_in.value))

        @c_ax_in.on_update
        def _(_) -> None:
            pdg.c_ax.value = float(c_ax_in.value)
            _set_param("c_ax", float(c_ax_in.value))

        @c_ayz_in.on_update
        def _(_) -> None:
            pdg.c_ayz.value = float(c_ayz_in.value)
            _set_param("c_ayz", float(c_ayz_in.value))

        @s_a_in.on_update
        def _(_) -> None:
            pdg.S_a.value = float(s_a_in.value)
            _set_param("S_a", float(s_a_in.value))

        @rho_in.on_update
        def _(_) -> None:
            pdg.rho.value = float(rho_in.value)
            _set_param("rho", float(rho_in.value))

        @l_arm_in.on_update
        def _(_) -> None:
            pdg.l_arm.value = float(l_arm_in.value)
            _set_param("l", float(l_arm_in.value))

        @l_p_in.on_update
        def _(_) -> None:
            pdg.l_p.value = float(l_p_in.value)
            _set_param("l_p", float(l_p_in.value))

        @j0.on_update
        def _(_) -> None:
            v = np.array([float(j0.value), float(j1.value), float(j2.value)], dtype=np.float64)
            pdg.J_diag.value = v
            _set_param("J_diag", v)

        @j1.on_update
        def _(_) -> None:
            v = np.array([float(j0.value), float(j1.value), float(j2.value)], dtype=np.float64)
            pdg.J_diag.value = v
            _set_param("J_diag", v)

        @j2.on_update
        def _(_) -> None:
            v = np.array([float(j0.value), float(j1.value), float(j2.value)], dtype=np.float64)
            pdg.J_diag.value = v
            _set_param("J_diag", v)

        @initial_pos_input.on_update
        def _(_) -> None:
            new_initial = np.array(initial_pos_input.value, dtype=np.float64)
            pdg.initial_position.value = new_initial
            optimization_problem.parameters["initial_position"] = new_initial
            pdg.position.initial = new_initial
            start_handle.position = _viser_tuple_from_model(new_initial)
            start_drag.position = _viser_tuple_from_model(new_initial)
            _update_glideslope_cone()

        @final_xy_input.on_update
        def _(_) -> None:
            vec = np.array(final_xy_input.value, dtype=np.float64)
            pdg.final_position.value = vec
            optimization_problem.parameters["final_position"] = vec
            target_handle.position = _target_viser_on_grid_plane(float(vec[0]), float(vec[1]))
            target_drag.position = target_handle.position
            _update_glideslope_cone()

    @start_drag.on_update
    def _(_) -> None:
        new_initial = _model_vec_from_viser(start_drag.position)
        pdg.initial_position.value = new_initial
        optimization_problem.parameters["initial_position"] = new_initial
        pdg.position.initial = new_initial
        start_handle.position = tuple(start_drag.position)
        initial_pos_input.value = tuple(new_initial)
        _update_glideslope_cone()

    @target_drag.on_update
    def _(_) -> None:
        vx, vy, _ = target_drag.position
        s = VISER_SCENE_SCALE
        target_drag.position = (vx, vy, 0.0)
        target_handle.position = (vx, vy, 0.0)
        new_final_xy = np.array([vy / s, vx / s], dtype=np.float64)
        pdg.final_position.value = new_final_xy
        optimization_problem.parameters["final_position"] = new_final_xy
        final_xy_input.value = tuple(new_final_xy)
        _update_glideslope_cone()

    def apply_all_state_edits() -> None:
        new_initial = np.asarray(pdg.initial_position.value, dtype=np.float64).reshape(3)
        pdg.position.initial = new_initial
        vm = float(pdg.v_max.value)
        pdg.velocity.max = np.array([vm, vm, vm], dtype=np.float64)
        pdg.velocity.min = np.array([-vm, -vm, -vm], dtype=np.float64)
        wm = float(pdg.w_max.value)
        pdg.angular_velocity.max = np.array([wm, wm, wm], dtype=np.float64)
        pdg.angular_velocity.min = np.array([-wm, -wm, -wm], dtype=np.float64)
        tm = float(pdg.T_max.value)
        pdg.thrust.max = np.array([tm, tm, tm], dtype=np.float64)
        pdg.thrust.min = np.array([-tm, -tm, -tm], dtype=np.float64)
        md = float(pdg.m_dry.value)
        pdg.mass.min = np.array([md], dtype=np.float64)

        final_xyz = np.array(
            [float(pdg.final_position.value[0]), float(pdg.final_position.value[1]), 0.0],
            dtype=np.float64,
        )
        pdg.position.guess = np.linspace(new_initial, final_xyz, pdg.n)
        pdg.velocity.guess = np.linspace(
            np.asarray(pdg.velocity.initial, dtype=np.float64),
            np.asarray(pdg.velocity.final, dtype=np.float64),
            pdg.n,
        )
        q_lo = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        q_hi = np.asarray(pdg.attitude.final, dtype=np.float64)
        pdg.attitude.guess = np.linspace(q_lo, q_hi, pdg.n)
        pdg.angular_velocity.guess = np.linspace(
            np.asarray(pdg.angular_velocity.initial, dtype=np.float64),
            np.asarray(pdg.angular_velocity.final, dtype=np.float64),
            pdg.n,
        )
        m0 = float(np.asarray(pdg.mass.initial, dtype=np.float64).flatten()[0])
        m_term = max(float(md), m0 - 0.2)
        pdg.mass.guess = np.linspace(np.array([m0]), np.array([m_term]), pdg.n).reshape(-1, 1)
        pdg.thrust.guess = np.linspace(
            np.array([float(pdg.gI.value) * m0, 0.0, 0.0]),
            np.array([float(pdg.gI.value) * md, 0.0, 0.0]),
            pdg.n,
        ).reshape(-1, 3)
        _sync_problem_parameters_from_base()

    def update_trajectory(V_multi_shoot: np.ndarray) -> None:
        try:
            n_x = optimization_problem.settings.sim.n_states
            n_u = optimization_problem.settings.sim.n_controls
            positions, velocities = extract_multishoot_trajectory(
                V_multi_shoot,
                n_x,
                n_u,
                position_slice=slice(1, 4),
                velocity_slice=slice(4, 7),
            )
            if len(positions) > 0:
                colors = compute_velocity_colors_realtime(velocities, _viridis_cmap)
                trajectory_handle.points = (
                    pdg.model_vec_to_viser_xyz(np.asarray(positions, dtype=np.float64))
                    * VISER_SCENE_SCALE
                ).astype(np.float32)
                trajectory_handle.colors = colors
        except Exception:
            x_traj = np.asarray(optimization_problem.state.x)
            if x_traj.size and x_traj.shape[1] >= 4:
                pos = x_traj[:, 1:4]
                points = (pdg.model_vec_to_viser_xyz(pos) * VISER_SCENE_SCALE).astype(np.float32)
                trajectory_handle.points = points
                trajectory_handle.colors = np.tile(
                    np.array([[255, 255, 0]], dtype=np.uint8), (points.shape[0], 1)
                )

    def optimization_loop() -> None:
        while state["running"]:
            try:
                if state["reset_requested"]:
                    apply_all_state_edits()
                    optimization_problem.reset()
                    state["reset_requested"] = False

                t0 = time.time()
                step_result = optimization_problem.step()
                solve_time_ms = (time.time() - t0) * 1000.0

                results = build_scp_step_results(step_result, solve_time_ms)
                results.update(get_print_queue_data(optimization_problem))
                metrics_text.content = format_metrics_markdown(results)

                if optimization_problem.history.V_history:
                    V_multi_shoot = np.asarray(optimization_problem.history.V_history[-1])
                    update_trajectory(V_multi_shoot)

                time.sleep(0.05)
            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(0.5)

    threading.Thread(target=optimization_loop, daemon=True).start()
    return server


if __name__ == "__main__":
    print("Starting 6DoF PDG realtime optimization.")
    print("Open the URL shown below in your browser.\n")
    server = create_realtime_server(pdg.problem)
    server.sleep_forever()
