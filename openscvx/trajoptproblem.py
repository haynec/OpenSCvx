import jax.numpy as jnp
from typing import List

import cvxpy as cp
from jax import jit
import numpy as np

from openscvx.config import (
    ScpConfig,
    SimConfig,
    ConvexSolverConfig,
    DiscretizationConfig,
    PropagationConfig,
    DevConfig,
    Config,
)
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
        x_max: jnp.ndarray,
        x_min: jnp.ndarray,
        u_max: jnp.ndarray,
        u_min: jnp.ndarray,
        scp: ScpConfig = None,
        dis: DiscretizationConfig = None,
        prp: PropagationConfig = None,
        sim: SimConfig = None,
        dev: DevConfig = None,
        cvx: ConvexSolverConfig = None,
        ctcs_augmentation_min=0.0,
        ctcs_augmentation_max=1e-4,
        time_dilation_factor_min=0.3,
        time_dilation_factor_max=3.0,
    ):

        # TODO (norrisg) move this into some augmentation function, if we want to make this be executed after the init (i.e. within problem.initialize) need to rethink how problem is defined

        x_min_augmented = np.hstack([x_min, ctcs_augmentation_min])
        x_max_augmented = np.hstack([x_max, ctcs_augmentation_max])

        u_min_augmented = np.hstack([u_min, time_dilation_factor_min * time_init])
        u_max_augmented = np.hstack([u_max, time_dilation_factor_max * time_init])

        x_bar_augmented = np.hstack([x_guess, np.full((x_guess.shape[0], 1), 0)])
        u_bar_augmented = np.hstack(
            [u_guess, np.full((u_guess.shape[0], 1), time_init)]
        )

        if dis is None:
            dis = DiscretizationConfig()

        if sim is None:
            sim = SimConfig(
                x_bar=x_bar_augmented,
                u_bar=u_bar_augmented,
                initial_state=initial_state,
                final_state=final_state,
                max_state=x_max_augmented,
                min_state=x_min_augmented,
                max_control=u_max_augmented,
                min_control=u_min_augmented,
                total_time=time_init,
                n_states=len(x_max),
            )

        if scp is None:
            scp = ScpConfig(
                n=N,
                k_max=200,
                w_tr=1e1,  # Weight on the Trust Reigon
                lam_cost=1e1,  # Weight on the Nonlinear Cost
                lam_vc=1e2,  # Weight on the Virtual Control Objective
                lam_vb=0e0,  # Weight on the Virtual Buffer Objective (only for penalized nodal constraints)
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

        if dev is None:
            dev = DevConfig()
        if cvx is None:
            cvx = ConvexSolverConfig()
        if prp is None:
            prp = PropagationConfig()

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
            dyn=veh,
            dis=dis,
            dev=dev,
            cvx=cvx,
            prp=prp,
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

        if not self.params.dev.debug:
            if self.params.dis.custom_integrator:
                calculate_discretization_lower = jit(
                    self.dynamics_discretized.calculate_discretization
                ).lower(
                    np.ones((self.params.scp.n, self.params.sim.n_states)),
                    np.ones((self.params.scp.n, self.params.sim.n_controls)),
                )
                self.dynamics_discretized.calculate_discretization = (
                    calculate_discretization_lower.compile()
                )
            else:
                dVdt_lower = jit(self.dynamics_discretized.dVdt).lower(
                    0.0,
                    np.ones(int(self.i5 * (self.params.scp.n - 1))),
                    np.ones((self.params.scp.n - 1, self.params.sim.n_controls)),
                    np.ones((self.params.scp.n - 1, self.params.sim.n_controls)),
                )
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
