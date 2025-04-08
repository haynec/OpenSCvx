import subprocess
import jax
import json
import os
import sys
import importlib

# Add the parent directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openscvx.ptr import PTR_main
from openscvx.config import Config

def test_main():
    problem_files = [
        'examples.params.cinema_vp',
        'examples.params.dr_vp_polytope',
        'examples.params.dr_vp',
        'examples.params.drone_racing',
        'examples.params.obstacle_avoidance',
    ]

    problems_list = []

    for problem_file in problem_files:
        module = importlib.import_module(problem_file)
        problem = getattr(module, 'problem')
        problems_list.append(problem)

    for problem in problems_list:
        # Force jax to use cpu
        # jax.config.update('jax_default_device', jax.devices('cpu')[0])

        result = problem.solve()
        
        # Assuming PTR_main returns a dictionary
        output_dict = result
        
        if output_dict['converged']:
            print("Results converged successfully.")
        else:
            print("Results did not converge.")
        
        # Clean up jax memory usage
        jax.clear_caches()

        # assert output_dict['converged'], f"Process failed with output"

test_main()