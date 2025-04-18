import numpy as np
import numpy.linalg as la
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.utils import qdcm, SSMP, SSM
from openscvx.constraints.boundary import BoundaryConstraint as bc
from openscvx.constraints.decorators import ctcs

n = 12  # Number of Nodes
total_time = 40.0  # Total time for the simulation


fuel_inds = -3  # Fuel Index in State
t_inds = -2  # Time Index in State
y_inds = -1  # Constraint Violation Index in State
s_inds = -1  # Time dilation index in Control


max_state = np.array(
    [200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 2000, 40]
)  # Upper Bound on the states
min_state = np.array(
    [-100.0, -100, -10, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0]
)  # Lower Bound on the states

initial_state = bc(jnp.array([8.0, -0.2, 2.2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
initial_state.type[6:13] = "Free"

final_state = bc(jnp.array([-10.0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 40]))
final_state.type[0:13] = "Free"
final_state.type[13] = "Minimize"

max_control = np.array(
    [0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562]
)
min_control = np.array([0, 0, 0, -18.665, -18.665, -0.55562])
initial_control = np.array([0, 0, 10, 0, 0, 0])

init_pose = np.array([13.0, 0.0, 2.0])
min_range = 4.0
max_range = 16.0

### View Planning Params ###
n_subs = 1  # Number of Subjects
alpha_x = 6.0  # Angle for the x-axis of Sensor Cone
alpha_y = 8.0  # Angle for the y-axis of Sensor Cone
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)  # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame
norm_type = np.inf  # Norm Type
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


def dynamics(x, u):
    m = 1.0  # Mass of the drone
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone
    # Unpack the state and control vectors
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    f = u[:3]
    tau = u[3:]

    q_norm = jnp.linalg.norm(q)
    q = q / q_norm

    # Compute the time derivatives of the state variables
    r_dot = v
    v_dot = (1 / m) * qdcm(q) @ f + jnp.array([0, 0, g_const])
    q_dot = 0.5 * SSMP(w) @ q
    w_dot = jnp.diag(1 / J_b) @ (tau - SSM(w) @ jnp.diag(J_b) @ w)
    fuel_dot = jnp.linalg.norm(u)[None]
    return jnp.hstack([r_dot, v_dot, q_dot, w_dot, fuel_dot])


def get_kp_pose(t):
    loop_time = 40.0
    loop_radius = 20.0

    t_angle = t / loop_time * (2 * jnp.pi)
    x = loop_radius * jnp.sin(t_angle)
    y = x * jnp.cos(t_angle)
    z = 0.5 * x * jnp.sin(t_angle)
    return jnp.array([x, y, z]).T + init_pose


def g_vp(x):
    p_s_I = get_kp_pose(x[t_inds])
    p_s_s = R_sb @ qdcm(x[6:10]).T @ (p_s_I - x[:3])
    return jnp.linalg.norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)


def g_min(x):
    p_s_I = get_kp_pose(x[t_inds])
    return min_range - jnp.linalg.norm(p_s_I - x[:3])


def g_max(x):
    p_s_I = get_kp_pose(x[t_inds])
    return jnp.linalg.norm(p_s_I - x[:3]) - max_range


constraints = [
    ctcs(lambda x, u: np.sqrt(2e1) * g_vp(x)),
    ctcs(lambda x, u: x[:-1] - max_state),
    ctcs(lambda x, u: min_state - x[:-1]),
    ctcs(lambda x, u: g_min(x)),
    ctcs(lambda x, u: g_max(x)),
]


u_bar = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
x_bar = np.linspace(initial_state.value, final_state.value, n)

x_bar[:, :3] = get_kp_pose(x_bar[:, t_inds]) + jnp.array([-5, 0.2, 0.2])[None, :]

R_sb = R_sb  # Sensor to body frame
b = R_sb @ np.array([0, 1, 0])
for k in range(n):
    kp = get_kp_pose(x_bar[k, t_inds])
    a = kp - x_bar[k, :3]
    # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the relative position vector
    q_xyz = np.cross(b, a)
    q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
    q_no_norm = np.hstack((q_w, q_xyz))
    q = q_no_norm / la.norm(q_no_norm)
    x_bar[k, 6:10] = q

problem = TrajOptProblem(
    dynamics=dynamics,
    constraints=constraints,
    N=n,
    time_init=total_time,
    x_guess=x_bar,
    u_guess=u_bar,
    initial_state=initial_state,  # Initial State
    final_state=final_state,
    x_max=max_state,
    x_min=min_state,
    u_max=max_control,  # Upper Bound on the controls
    u_min=min_control,  # Lower Bound on the controls
    ctcs_augmentation_max=1e-8,
)

problem.params.prp.dt = 0.1
problem.params.dis.custom_integrator = True

problem.params.scp.w_tr = 4e0  # Weight on the Trust Reigon
problem.params.scp.lam_cost = 1e-2  # Weight on the Minimal Fuel Objective
problem.params.scp.lam_vc = (
    1e1  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
)
problem.params.scp.ep_tr = 5e-6  # Trust Region Tolerance
problem.params.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.params.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
problem.params.scp.w_tr_adapt = 1.3  # Trust Region Adaptation Factor
problem.params.scp.w_tr_max_scaling_factor = 1e3  # Maximum Trust Region Weight

problem.params.dyn.R_sb = R_sb
problem.params.dyn.init_pose = init_pose
problem.params.dyn.get_kp_pose = get_kp_pose

plotting_dict = dict(
    n_subs=n_subs,
    alpha_x=alpha_x,
    alpha_y=alpha_y,
    R_sb=R_sb,
    init_poses=init_pose,
    norm_type=norm_type,
    min_range=min_range,
    max_range=max_range,
)
