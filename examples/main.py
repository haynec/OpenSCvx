import sys
import os
import pickle
import jax

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from examples.params.dr_vp import problem, plotting_dict

from openscvx.ptr import PTR_main
from openscvx.plotting import plot_camera_polytope_animation, plot_camera_animation, plot_animation, plot_scp_animation, plot_constraint_violation, plot_control, plot_state, plot_losses, plot_conic_view_animation, plot_camera_view
from openscvx.config import Config

################################
# Author: Chris Hayner  
# Autonomous Controls Laboratory
################################

jax.config.update('jax_default_device', jax.devices('cpu')[0])

problem.initialize()
results = problem.solve()

# Check if results folder exists
if not os.path.exists('results'):
    os.makedirs('results')

# Save results
with open('results/results.pickle', 'wb') as f:
    pickle.dump(results, f) 

# Load results
with open('results/results.pickle', 'rb') as f:
    results = pickle.load(f) 

# results = problem.post_process(results)
# results.update(plotting_dict)
# plot_animation(results, problem.params)
# plot_camera_animation(results, problem.params)
