import pytest

import jax

from examples.params.cinema_vp import problem as cinema_vp_problem
from examples.params.dr_vp_polytope import problem as dr_vp_polytope_problem
from examples.params.dr_vp import problem as dr_vp_problem
from examples.params.drone_racing import problem as drone_racing_problem
from examples.params.obstacle_avoidance import problem as obstacle_avoidance_problem


def test_obstacle_avoidance():
    # This test is specific to the obstacle avoidance problem
    problem = obstacle_avoidance_problem
    problem.initialize()
    result = problem.solve()
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    
    assert output_dict['converged'], "Obstacle Avoidance Process failed with output"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_drone_racing():
    # This test is specific to the drone racing problem
    problem = drone_racing_problem
    problem.initialize()
    result = problem.solve()
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    
    assert output_dict['converged'], "Drone Racing Process failed with output"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_dr_vp():
    # This test is specific to the dr_vp problem
    problem = dr_vp_problem
    problem.initialize()
    result = problem.solve()
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    
    assert output_dict['converged'], "DR VP Process failed with output"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_dr_vp_polytope():
    # This test is specific to the dr_vp_polytope problem
    problem = dr_vp_polytope_problem
    problem.initialize()
    result = problem.solve()
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    
    assert output_dict['converged'], "DR VP Polytope Process failed with output"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_cinema_vp():
    # This test is specific to the cinema_vp problem
    problem = cinema_vp_problem
    problem.initialize()
    result = problem.solve()
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    
    assert output_dict['converged'], "Cinema VP Process failed with output"
    
    # Clean up jax memory usage
    jax.clear_caches()