"""Shared builder for ``make_scp_iteration`` tests.

Assembles a fused iteration body from an initialized ``Problem`` using
``jax.jit`` discretization solvers — the production no-disk path. Underscore
prefix keeps pytest from collecting it as a test module.
"""

from __future__ import annotations

import jax

from openscvx.algorithms.scvx.iteration import make_scp_iteration
from openscvx.discretization import get_impulsive_discretization_solver


def build_iteration_fn(prob):
    """Build ``iteration_fn`` from an initialized problem's components."""
    dis_continuous = jax.jit(prob.discretizer.get_solver(prob.lowered.dynamics, prob.settings))
    dis_impulsive = jax.jit(get_impulsive_discretization_solver(prob.lowered.dynamics_discrete))
    return make_scp_iteration(
        dis_continuous=dis_continuous,
        dis_impulsive=dis_impulsive,
        jax_constraints=prob._compiled_constraints,
        solver_callback=prob.solver.iteration_callback(),
        autotuner=prob.algorithm.autotuner,
        settings=prob.settings,
    )
