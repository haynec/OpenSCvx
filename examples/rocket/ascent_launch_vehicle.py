import os
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

try:
    from .helpers import orbital_elements_2_cartesian_rv
except ImportError:
    from helpers import orbital_elements_2_cartesian_rv

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import openscvx as ox


def piecewise_by_node(values, ranges):
    """Return a node-indexed piecewise expression using Cond(None, ...)."""
    expr = float(values[-1])
    for value, node_range in zip(values[-2::-1], ranges[::-1]):
        expr = ox.Cond(None, float(value), expr, node_ranges=[node_range])
    return expr


def build_true_anomaly_transfer_guess(r_start, r_end, mu, n_nodes):
    """Guess trajectory on a conic with linearly spaced true anomaly."""
    r0_norm = np.linalg.norm(r_start)
    rf_norm = np.linalg.norm(r_end)

    h_vec = np.cross(r_start, r_end)
    h_norm = np.linalg.norm(h_vec)

    x_hat = r_start / r0_norm
    z_hat = h_vec / h_norm
    y_hat = np.cross(z_hat, x_hat)
    y_norm = np.linalg.norm(y_hat)
    y_hat = y_hat / y_norm

    cos_dnu = np.dot(r_start, r_end) / (r0_norm * rf_norm)
    dnu = float(np.arccos(cos_dnu))

    denom = rf_norm * np.cos(dnu) - r0_norm
    e = (r0_norm - rf_norm) / denom

    p = r0_norm * (1.0 + e)

    nu_grid = np.linspace(0.0, dnu, n_nodes)
    cos_nu = np.cos(nu_grid)
    sin_nu = np.sin(nu_grid)

    radial_denom = 1.0 + e * cos_nu
    radial_denom = np.where(np.abs(radial_denom) <= 1e-8, 1e-8, radial_denom)
    radius = p / radial_denom

    r_pqw = np.column_stack([radius * cos_nu, radius * sin_nu, np.zeros(n_nodes)])
    speed_factor = np.sqrt(mu / p)
    v_pqw = np.column_stack(
        [-speed_factor * sin_nu, speed_factor * (e + cos_nu), np.zeros(n_nodes)]
    )

    R_i_pqw = np.column_stack([x_hat, y_hat, z_hat])
    r_guess = r_pqw @ R_i_pqw.T
    v_guess = v_pqw @ R_i_pqw.T
    return r_guess, v_guess


def cartesian_rv_to_orbital_elements_symbolic(r_vect, v_vect, gravitational_parameter):
    """Symbolic mapping from Cartesian radius/velocity to classical orbital elements."""
    mu = float(gravitational_parameter)

    r_norm = ox.linalg.Norm(r_vect)
    v_norm = ox.linalg.Norm(v_vect)

    h_vect = ox.spatial.SSM(r_vect) @ v_vect
    h_norm = ox.linalg.Norm(h_vect)

    e_vect = (ox.spatial.SSM(v_vect) @ h_vect) / mu - r_vect / r_norm
    eccentricity = ox.linalg.Norm(e_vect)

    a_denom = (2.0 / r_norm ) - (v_norm**2) / mu
    semimajor = 1.0 / a_denom 

    k_vect = np.array([0.0, 0.0, 1.0])
    n_vect = ox.spatial.SSM(k_vect) @ h_vect
    n_norm = ox.linalg.Norm(n_vect)

    inclination = ox.Acos(h_vect[2] / h_norm)
    right_ascension = ox.Atan2(n_vect[1], n_vect[0])

    dot_ne = ox.Sum(n_vect * e_vect)
    cross_ne = ox.spatial.SSM(n_vect) @ e_vect
    sin_argp = ox.Sum(h_vect * cross_ne) / (h_norm * n_norm * eccentricity)
    cos_argp = dot_ne / (n_norm * eccentricity )
    arg_periapsis = ox.Atan2(sin_argp, cos_argp)

    return semimajor, eccentricity, inclination, right_ascension, arg_periapsis


def plot_altitude_and_speed_vs_time(results, earth_radius_scaled, scales_dict):
    """Plot altitude and speed norm side by side in one figure."""
    time_scale = scales_dict["time"]
    length_scale = scales_dict["length"]
    speed_scale = scales_dict["speed"]

    t_nodes_s = np.asarray(results.nodes["time"]).flatten() * time_scale
    r_nodes = np.asarray(results.nodes["position"])
    v_nodes = np.asarray(results.nodes["velocity"])
    h_nodes_km = (np.linalg.norm(r_nodes, axis=1) - earth_radius_scaled) * length_scale / 1000.0
    speed_nodes_kms = np.linalg.norm(v_nodes, axis=1) * speed_scale / 1000.0

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Altitude vs Time", "Speed Norm vs Time"),
    )

    if bool(results.trajectory):
        if "time" in results.trajectory and "position" in results.trajectory:
            t_traj_s = np.asarray(results.trajectory["time"]).flatten() * time_scale
            r_traj = np.asarray(results.trajectory["position"])
            h_traj_km = (
                (np.linalg.norm(r_traj, axis=1) - earth_radius_scaled) * length_scale / 1000.0
            )
            fig.add_trace(
                go.Scatter(
                    x=t_traj_s,
                    y=h_traj_km,
                    mode="lines",
                    name="Trajectory",
                    line={"color": "green", "width": 2},
                    legendgroup="trajectory",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        if "time" in results.trajectory and "velocity" in results.trajectory:
            t_traj_s = np.asarray(results.trajectory["time"]).flatten() * time_scale
            v_traj = np.asarray(results.trajectory["velocity"])
            speed_traj_kms = np.linalg.norm(v_traj, axis=1) * speed_scale / 1000.0
            fig.add_trace(
                go.Scatter(
                    x=t_traj_s,
                    y=speed_traj_kms,
                    mode="lines",
                    name="Trajectory",
                    line={"color": "green", "width": 2},
                    legendgroup="trajectory",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.add_trace(
        go.Scatter(
            x=t_nodes_s,
            y=h_nodes_km,
            mode="markers",
            name="Nodes",
            marker={"color": "cyan", "size": 6},
            legendgroup="nodes",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_nodes_s,
            y=speed_nodes_kms,
            mode="markers",
            name="Nodes",
            marker={"color": "cyan", "size": 6},
            legendgroup="nodes",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Altitude and Speed Norm",
        template="plotly_dark",
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Altitude [km]", row=1, col=1)
    fig.update_yaxes(title_text="Speed [km/s]", row=1, col=2)
    return fig


tog_scaling = True
guess_mode = "true_anomaly"  # Options: "true_anomaly", "linear"

# Physical data (SI units)
earth_radius = 6378145.0
gravitational_parameter = 3.986012e14
initial_mass = 301454.0
angular_rate_earth = 7.29211585e-5
sealevel_density = 1.225
density_scale_height = 7200.0
sealevel_gravity = 9.80665

scale_keys = [
    "length",
    "speed",
    "time",
    "acceleration",
    "mass",
    "force",
    "area",
    "volume",
    "density",
    "gravitational_parameter",
]
scales = {key: 1.0 for key in scale_keys}

if tog_scaling:
    scales["length"] = earth_radius
    scales["speed"] = np.sqrt(gravitational_parameter / scales["length"])
    scales["time"] = scales["length"] / scales["speed"]
    scales["acceleration"] = scales["speed"] / scales["time"]
    scales["mass"] = initial_mass
    scales["force"] = scales["mass"] * scales["acceleration"]
    scales["area"] = scales["length"] ** 2
    scales["volume"] = scales["area"] * scales["length"]
    scales["density"] = scales["mass"] / scales["volume"]
    scales["gravitational_parameter"] = scales["acceleration"] * scales["length"] ** 2

omega = angular_rate_earth * scales["time"]
auxiliary_data = {
    "angular_rate_earth": omega,
    "angular_rate_earth_matrix": omega * np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "gravitational_parameter": gravitational_parameter / scales["gravitational_parameter"],
    "sealevel_density": sealevel_density / scales["density"],
    "density_scale_height": density_scale_height / scales["length"],
    "earth_radius": earth_radius / scales["length"],
    "sealevel_gravity": sealevel_gravity / scales["acceleration"],
    "cd": 0.5,
    "reference_area": (4.0 * np.pi) / scales["area"],
}

lat0 = np.deg2rad(28.5)
r_0 = np.array(
    [
        auxiliary_data["earth_radius"] * np.cos(lat0),
        0.0,
        auxiliary_data["earth_radius"] * np.sin(lat0),
    ]
)
v_0 = r_0 @ auxiliary_data["angular_rate_earth_matrix"].T
v_0 = v_0 + (5.0 / scales["speed"]) * (r_0 / np.linalg.norm(r_0))

bt_srb = 75.2 / scales["time"]
bt_first = 261.0 / scales["time"]
bt_second = 700.0 / scales["time"]
t1 = 75.2 / scales["time"]
t2 = 150.4 / scales["time"]
t3 = 261.0 / scales["time"]
t4 = 961.0 / scales["time"]

m_tot_srb = 19290.0 / scales["mass"]
m_prop_srb = 17010.0 / scales["mass"]
m_dry_srb = m_tot_srb - m_prop_srb
m_tot_first = 104380.0 / scales["mass"]
m_prop_first = 95550.0 / scales["mass"]
m_dry_first = m_tot_first - m_prop_first
m_tot_second = 19300.0 / scales["mass"]
m_prop_second = 16820.0 / scales["mass"]
m_payload = 4164.0 / scales["mass"]

thrust_srb = 628500.0 / scales["force"]
thrust_first = 1083100.0 / scales["force"]
thrust_second = 110094.0 / scales["force"]

m_dot_srb = m_prop_srb / bt_srb
isp_srb = thrust_srb / (auxiliary_data["sealevel_gravity"] * m_dot_srb)
m_dot_first = m_prop_first / bt_first
isp_first = thrust_first / (auxiliary_data["sealevel_gravity"] * m_dot_first)
m_dot_second = m_prop_second / bt_second
isp_second = thrust_second / (auxiliary_data["sealevel_gravity"] * m_dot_second)

af = 24361140.0 / scales["length"]
ef = 0.7308
incf = np.deg2rad(28.5)
omf = np.deg2rad(269.8)
omf_arg = np.deg2rad(130.5)
omf_wrapped = (omf + np.pi) % (2.0 * np.pi) - np.pi
omf_arg_wrapped = (omf_arg + np.pi) % (2.0 * np.pi) - np.pi
nuguess = 0.0
oe = np.array([af, ef, incf, omf, omf_arg, nuguess])
r_f, v_f = orbital_elements_2_cartesian_rv(oe, auxiliary_data["gravitational_parameter"])

m_0 = m_payload + m_tot_second + m_tot_first + 9.0 * m_tot_srb
m_f = m_payload

r_min = -2.0 * auxiliary_data["earth_radius"]
r_max = -r_min
v_min = -10000.0 / scales["speed"]
v_max = -v_min

# Discretization and phase transition nodes
n_1 = 5
n_2 = 10
n_3 = 15
n_4 = 20
n   = n_4
k_1 = n_1 - 1
k_2 = n_2 - 1
k_3 = n_3 - 1
transition_nodes = [k_1, k_2, k_3]
phase_ranges = [(0, k_1), (k_1, k_2), (k_2, k_3)]

thrust_profile = np.array(
    [
        6.0 * thrust_srb + thrust_first,
        3.0 * thrust_srb + thrust_first,
        thrust_first,
        thrust_second,
    ]
)
m_dot_profile = np.array(
    [
        -(6.0 * thrust_srb) / (auxiliary_data["sealevel_gravity"] * isp_srb)
        - (thrust_first) / (auxiliary_data["sealevel_gravity"] * isp_first),
        -(3.0 * thrust_srb) / (auxiliary_data["sealevel_gravity"] * isp_srb)
        - (thrust_first) / (auxiliary_data["sealevel_gravity"] * isp_first),
        -(thrust_first) / (auxiliary_data["sealevel_gravity"] * isp_first),
        -(thrust_second) / (auxiliary_data["sealevel_gravity"] * isp_second),
    ]
)

guess_mode = guess_mode.lower()
if guess_mode == "true_anomaly":
    r_guess, v_guess = build_true_anomaly_transfer_guess(
        r_0, r_f, auxiliary_data["gravitational_parameter"], n
    )
    if r_guess is None or v_guess is None:
        r_guess = np.linspace(r_0, r_f, n)
        v_guess = np.linspace(v_0, v_f, n)
elif guess_mode == "linear":
    r_guess = np.linspace(r_0, r_f, n)
    v_guess = np.linspace(v_0, v_f, n)
else:
    raise ValueError(
        f"Unsupported guess_mode='{guess_mode}'. Use 'true_anomaly' or 'linear'."
    )
r_guess[0, :] = r_0
r_guess[-1, :] = r_f
v_guess[0, :] = v_0
v_guess[-1, :] = v_f

# States
r = ox.State("position", shape=(3,))
r.max = r_max * np.ones((3,))
r.min = r_min * np.ones((3,))
r.initial = r_0
r.final = [ox.Free(r_f[0]), ox.Free(r_f[1]), ox.Free(r_f[2])]
r.guess = r_guess

v = ox.State("velocity", shape=(3,))
v.max = v_max * np.ones((3,))
v.min = v_min * np.ones((3,))
v.initial = v_0
v.final = [ox.Free(v_f[0]), ox.Free(v_f[1]), ox.Free(v_f[2])]
v.guess = v_guess

m = ox.State("mass", shape=(1,))
m.max = np.array([m_0])
m.min = np.array([m_f])
m.initial = [m_0]
m.final = [ox.Maximize(m_f)]
m.guess = np.linspace(np.array([m_0]), np.array([m_f]), n)

time = ox.Time(
    initial=0.0,
    final=ox.Free(t4),
    min=0.0,
    max=t4,
    guess=np.linspace(0.0, t4, n).reshape(-1, 1),
    time_dilation_min=1e-2
)

# Controls: continuous thrust direction + impulsive stage-drop mass
u = ox.Control("body_direction", shape=(3,))
u.max = 10.0 * np.ones((3,))
u.min = -u.max
u_guess = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
u_guess[:k_2, :] = np.array([0.0, 1.0, 0.0])  # phases 1-2
u_guess[k_2:, :] = np.array([1.0, 0.0, 0.0])  # phases 3-4
u.guess = u_guess

delta_m = ox.Control(
    "delta_mass",
    shape=(1,),
    parameterization="impulsive",
    nodes=transition_nodes,
)
delta_m.max = np.array([0.0])
delta_m.min = np.array([-10.0 * m_dry_srb])
delta_m_guess = np.zeros((n, 1))
delta_m_guess[k_1, 0] = -6.0 * m_dry_srb
delta_m_guess[k_2, 0] = -3.0 * m_dry_srb
delta_m_guess[k_3, 0] = -1.0 * m_dry_first
delta_m.guess = delta_m_guess

states = [r, v, m]
controls = [u, delta_m]

r_norm = ox.linalg.Norm(r)
mass_scalar = m[0]
v_rel = v - r @ auxiliary_data["angular_rate_earth_matrix"].T
speed_rel = ox.linalg.Norm(v_rel)
altitude = r_norm - auxiliary_data["earth_radius"]
rho = auxiliary_data["sealevel_density"] * ox.Exp(-altitude / auxiliary_data["density_scale_height"])
drag_force = -0.5 * auxiliary_data["cd"] * auxiliary_data["reference_area"] * rho * speed_rel * v_rel

thrust_piecewise = piecewise_by_node(thrust_profile, phase_ranges)
m_dot_piecewise = piecewise_by_node(m_dot_profile, phase_ranges)

dynamics = {
    "position": v,
    "velocity": drag_force / mass_scalar
    + (thrust_piecewise / mass_scalar) * u
    - (auxiliary_data["gravitational_parameter"] / (r_norm**3)) * r,
    "mass": m_dot_piecewise,
}

dynamics_discrete = {
    "position": r,
    "velocity": v,
    "mass": m + delta_m,
}

constraints = []

constraints.append((ox.linalg.Norm(u) <= 1.0).convex())
constraints.append(ox.ctcs(altitude >= 0))

# Impose phase boundary times at specific nodes.
constraints.append((time.at(k_1) == t1).convex())
constraints.append((time.at(k_2) == t2).convex())
constraints.append((time.at(k_3) == t3).convex())
constraints.append((time.at(n - 1) >= t3).convex())

# Force the exact stage-separation mass jumps at those same nodes.
constraints.append((delta_m.at(k_1) == np.array([-6.0 * m_dry_srb])).convex())
constraints.append((delta_m.at(k_2) == np.array([-3.0 * m_dry_srb])).convex())
constraints.append((delta_m.at(k_3) == np.array([-1.0 * m_dry_first])).convex())

a_f_sym, e_f_sym, i_f_sym, Om_f_sym, om_f_sym = cartesian_rv_to_orbital_elements_symbolic(
    r.at(n - 1),
    v.at(n - 1),
    auxiliary_data["gravitational_parameter"],
)
oe_tol = {
    "a": 1e-3,
    "e": 5e-4,
    "i": 1e-4,
    "Om": 1e-3,
    "om": 1e-3,
}
constraints.append(a_f_sym <= af + oe_tol["a"])
constraints.append(a_f_sym >= af - oe_tol["a"])
constraints.append(e_f_sym <= ef + oe_tol["e"])
constraints.append(e_f_sym >= ef - oe_tol["e"])
constraints.append(i_f_sym <= incf + oe_tol["i"])
constraints.append(i_f_sym >= incf - oe_tol["i"])
constraints.append(Om_f_sym <= omf_wrapped + oe_tol["Om"])
constraints.append(Om_f_sym >= omf_wrapped - oe_tol["Om"])
constraints.append(om_f_sym <= omf_arg_wrapped + oe_tol["om"])
constraints.append(om_f_sym >= omf_arg_wrapped - oe_tol["om"])

problem = ox.Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    float_dtype="float64",
    discretizer=ox.LinearizeDiscretizeSparse(),
    algorithm={'autotuner': ox.AugmentedLagrangian(eta_lambda=1E-1, ep=0.99)}
)

# Tighten integration tolerance and use a denser propagation time grid for plotting.
problem.settings.prp.rtol   = 1e-9
problem.settings.prp.atol   = 1e-9

problem.algorithm.lam_cost  = 1e-2
problem.algorithm.lam_prox  = 2e-1  
problem.algorithm.lam_vc    = 1e0
problem.algorithm.lam_vb    = 1e-2

problem.algorithm.k_max = 40

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    try:
        from openscvx.plotting import plot_controls, plot_states

        plot_states(results).show()
        plot_controls(results).show()
        plot_altitude_and_speed_vs_time(results, auxiliary_data["earth_radius"], scales).show()
    except Exception as exc:
        print(f"Plotting unavailable: {exc}")
