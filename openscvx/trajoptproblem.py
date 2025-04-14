import jax.numpy as jnp
from typing import List

import cvxpy as cp
from jax import jit
import numpy as np

from openscvx.config import ScpConfig, SimConfig, Config
from openscvx.dynamics import Dynamics
from openscvx.discretization import ExactDis
from openscvx.constraints.boundary import BoundaryConstraint
from openscvx.ptr import PTR_init, PTR_main, PTR_post


# TODO: (norrisg) Decide whether to have constraints`, `cost`, alongside `dynamics`, ` etc.
class TrajOptProblem:
    def __init__(
        self,
        dynamics: callable,
        constraints: List[callable],
        N: int,
        time_init: float,
        x_guess: jnp.ndarray,
        u_guess: jnp.ndarray,
        initial_state: BoundaryConstraint,
        final_state: BoundaryConstraint,
        initial_control: jnp.ndarray,
        x_max: jnp.ndarray,
        x_min: jnp.ndarray,
        u_max: jnp.ndarray,
        u_min: jnp.ndarray,
        scp: ScpConfig = None,
        sim: SimConfig = None,
    ):
        
        x_min_augmented = np.hstack([x_min, 0])
        x_max_augmented = np.hstack([x_max, 1e-4])

        if sim is None:
            sim = SimConfig(
                x_bar=x_guess,
                u_bar=u_guess,
                initial_state=initial_state,
                final_state=final_state,
                max_state=x_max_augmented,
                min_state=x_min_augmented,
                initial_control=initial_control,
                max_control=u_max,
                min_control=u_min,
                total_time=time_init,
                n_states=len(x_max),
                dt=0.1,
            )

        if scp is None:
            scp = ScpConfig(
                n=N,
                k_max=200,
                w_tr=1e1,  # Weight on the Trust Reigon
                lam_cost=1e1,  # Weight on the Nonlinear Cost
                lam_vc=1e2,  # Weight on the Virtual Control Objective
                lam_vb=0e0, # Weight on the Virtual Buffer Objective (only for penalized nodal constraints)
                ep_tr=1e-4,  # Trust Region Tolerance
                ep_vb=1e-4,  # Virtual Control Tolerance
                ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
                cost_drop=4,  # SCP iteration to relax minimal final time objective
                cost_relax=0.5,  # Minimal Time Relaxation Factor
                w_tr_adapt=1.2,  # Trust Region Adaptation Factor
                w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
            )
        else:
            assert (
                self.scp.n == N
            ), "Number of segments must be the same as in the config"

        self.constraints_ctcs = []
        self.constraints_nodal = []

        for constraint in constraints:
            if constraint.constraint_type == "ctcs":
                self.constraints_ctcs.append(
                    lambda x, u, func=constraint: jnp.sum(func.penalty(func(x, u)))
                )
            elif constraint.constraint_type == "nodal":
                self.constraints_nodal.append(constraint)
            else:
                raise ValueError(
                    f"Unknown constraint type: {constraint.constraint_type}, All constraints must be decorated with @ctcs or @nodal"
                )

        veh = Dynamics(
            dynamics,
            self.constraints_ctcs,
            self.constraints_nodal,  # TODO (norrisg) Maybe move this outside of the dynamics?
            initial_state=initial_state,
            final_state=final_state,
        )

        self.params = Config(
            sim=sim,
            scp=scp,
            veh=veh,
        )

        self.ocp: cp.Problem = None
        self.dynamics_discretized: ExactDis = None
        self.cpg_solve = None

    def initialize(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        self.ocp, self.dynamics_discretized, self.cpg_solve = PTR_init(self.params)

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Define indices for slicing the augmented state vector
        self.i0 = 0
        self.i1 = n_x
        self.i2 = self.i1 + n_x * n_x
        self.i3 = self.i2 + n_x * n_u
        self.i4 = self.i3 + n_x * n_u
        self.i5 = self.i4 + n_x

        if not self.params.sim.debug:
            if self.params.sim.custom_integrator:
                calculate_discretization_lower = jit(self.dynamics_discretized.calculate_discretization).lower(np.ones((self.params.scp.n, self.params.sim.n_states)), np.ones((self.params.scp.n, self.params.sim.n_controls)))
                self.dynamics_discretized.calculate_discretization = calculate_discretization_lower.compile()
            else:
                dVdt_lower = jit(self.dynamics_discretized.dVdt).lower(0.0, np.ones(int(self.i5*(self.params.scp.n-1))), np.ones((self.params.scp.n-1, self.params.sim.n_controls)), np.ones((self.params.scp.n-1, self.params.sim.n_controls)))
                self.dynamics_discretized.dVdt = dVdt_lower.compile()


    def solve(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        if self.ocp is None or self.dynamics_discretized is None:
            raise ValueError(
                "Problem has not been initialized. Call initialize() before solve()"
            )

        return PTR_main(
            self.params, self.ocp, self.dynamics_discretized, self.cpg_solve
        )

    def post_process(self, result):
        return PTR_post(self.params, result, self.dynamics_discretized)
