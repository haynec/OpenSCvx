from quadsim.ptr import PTR_main
from quadsim.params.obstacle_avoidance import params
from quadsim.plotting import plot_camera_polytope_animation, plot_rocket_animation, plot_camera_view_ral, plot_conic_view_ral, plot_main_ral_dr, plot_main_ral_cine, plot_camera_animation, plot_animation, plot_scp_animation, plot_constraint_violation, plot_control, plot_state, plot_losses, plot_conic_view_animation, plot_camera_view
from quadsim.config import Config
import os

import pickle
import warnings
warnings.filterwarnings("ignore")

################################
# Author: Chris Hayner  
# Autonomous Controls Laboratory
################################

params = Config.from_config(params, savedir="results/")
results = PTR_main(params) 

# Check if results folder exists
if not os.path.exists('results'):
    os.makedirs('results')
    
# Save results
with open('results/results.pickle', 'wb') as f:
    pickle.dump(results, f) 

# Load results
with open('results/results.pickle', 'rb') as f:
    results = pickle.load(f) 

plot_animation(results, params)