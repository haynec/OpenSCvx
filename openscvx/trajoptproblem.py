import jax.numpy as jnp
from typing import List

from openscvx.config import ScpConfig, SimConfig, Config
from openscvx.dynamics import Dynamics
from openscvx.constraints.boundary import BoundaryConstraint
# from openscvx.constraints.custom import CustomConstraint
from openscvx.ptr import PTR_main


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

        if sim is None:
            sim = SimConfig(
                x_bar=x_guess,
                u_bar=u_guess,
                initial_state=initial_state,
                final_state=final_state,
                max_state=x_max,
                min_state=x_min,
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
                lam_vc=1e2,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
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
                self.constraints_ctcs.append(lambda x,u: jnp.sum(jnp.maximum(0, constraint(x, u)) ** 2))

        veh = Dynamics(
            dynamics,
            self.constraints_ctcs,
            initial_state=initial_state,
            final_state=final_state,
        )

        self.params = Config(
            sim=sim,
            scp=scp,
            veh=veh,
        )

    def solve(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        return PTR_main(self.params)
