import os
import sys

import jax
# use float64
jax.config.update("jax_enable_x64", True)

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox

from openscvx.plotting import plot_states, plot_controls

n           = 30 

p           = ox.State("p", shape=(2,))
p.max       = np.array([5.0, 5.0]) 
p.min       = np.array([-5.0, -5.0]) 
p.initial   = np.array([-5.0, 5.0])
p.final     = np.array([5.0, -5.0])
p.guess     = np.linspace( p.initial, p.final, n )

v           = ox.State("v", shape=(2,))
v.max       = np.array([3.0, 3.0])  
v.min       = np.array([-3.0, -3.0]) 
v.initial   = np.array([0.0, 0.0])
v.final     = np.array([0.0, 0.0])
v.guess     = np.linspace( v.initial, v.final, n )

u_1         = ox.Control("u_1", shape=(2,))
u_1.max     = np.array([1.0, 1.0])
u_1.min     = np.array([-1.0, -1.0])
u_1.guess   = np.linspace( np.array([0.0, 0.0]), np.array([0.0, 0.0]), n )

u_2         = ox.Control("u_2", shape=(2,))
u_2.max     = np.array([5.0, 5.0])
u_2.min     = np.array([-5.0, -5.0])
u_2.guess   = np.linspace( np.array([0.0, 0.0]), np.array([0.0, 0.0]), n )

dynamics_1 = {
        "p": v,
        "v": u_1, 
}

dynamics_2 = {
        "p": v,
        "v": u_2, 
}

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(10.0),  # Free final time with initial guess
    min=0.0,
    max=15.0,  # Maximum time in seconds
)

# States and controls formulation
states      = [p, v, time]
controls    = [u_1, u_2]

# Additional constraints 
constraints         = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# Multi-phase-like formulation
n_1         = 9
n_2         = 21

p_dynamics_1_to_2 = ox.Cond(
    None, 
    dynamics_1["p"],
    dynamics_2["p"],
    node_ranges=[(0, n_1-1), (n_2-1, n)]
)

v_dynamics_1_to_2 = ox.Cond(
    None, 
    dynamics_1["v"],
    dynamics_2["v"],
    node_ranges=[(0, n_1-1), (n_2-1, n)]
)

dynamics = {
    "p": p_dynamics_1_to_2,
    "v": v_dynamics_1_to_2, 
    "time": 1.0,
    }

constraints.append((time.at( n_1 - 1 )                      == 1.0).convex())
constraints.append((time.at( n   - 1 ) - time.at( n_2 - 1 ) == 1.0).convex())

prob = ox.Problem(
    dynamics=dynamics,
    states=states, 
    controls=controls,
    time=time, 
    constraints=constraints,
    N=n
)

# Set solver parameters
prob.settings.scp.lam_cost  = 1e0  
ratio_prox_cost             = 1e-2
ratio_vc_cost               = 1e1

prob.settings.scp.lam_prox  = prob.settings.scp.lam_cost * ratio_prox_cost
prob.settings.scp.lam_vc    = prob.settings.scp.lam_cost * ratio_vc_cost
prob.settings.scp.k_max     = 200

prob.settings.scp.ep_vc = 1e-6

if __name__ == "__main__":
    prob.initialize()
    results = prob.solve()
    results = prob.post_process()
    plot_states(results).show()
    plot_controls(results).show()
