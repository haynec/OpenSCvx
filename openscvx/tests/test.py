import subprocess
import jax
import json
import os
import sys
import importlib

# Add the parent directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from openscvx.ptr import PTR_main
from openscvx.config import Config

def test_main():
    params_files = [
        'examples.params.cinema_vp',
        'examples.params.dr_vp_polytope',
        'examples.params.dr_vp',
        'examples.params.drone_racing',
        'examples.params.obstacle_avoidance',
    ]

    params_list = []

    for params_file in params_files:
        module = importlib.import_module(params_file)
        params = getattr(module, 'params')
        params_list.append(params)

    for params in params_list:
        jax.config.update('jax_default_device', jax.devices('cpu')[0])
        
        config_params = Config.from_config(params, savedir="results/")
        result = PTR_main(config_params)
        
        # Assuming PTR_main returns a dictionary
        output_dict = result
        
        if output_dict['converged']:
            print("Results converged successfully.")
        else:
            print("Results did not converge.")

        assert output_dict['converged'], f"Process failed with output: {output_dict}"
        
test_main()