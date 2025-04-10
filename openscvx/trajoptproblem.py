import jax.numpy as jnp
from typing import List

from openscvx.config import ScpConfig, SimConfig, Config
from openscvx.dynamics import Dynamics
from openscvx.constraints.boundary import BoundaryConstraint

# from openscvx.constraints.custom import CustomConstraint
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
        init_params: dict = None,
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

        if init_params is not None:
            sim.debug = init_params["debug"]
            sim.profiling = init_params["profiling"]
            sim.cvxpygen = init_params["cvxpygen"]

        self.constraints_ctcs = []
        self.constraints_nodal = []
        self.constraints_ncvx_nodal = []

        for constraint in constraints:
            if constraint.constraint_type == "ctcs":
                # Bind the current 'constraint' function to 'func' to prevent late binding issue for lambda functions
                if constraint.penalty == 'Default':
                    self.constraints_ctcs.append(
                        lambda x, u, func=constraint: jnp.sum(
                            constraint.penalty(func(x,u))
                        )
                    )
            elif constraint.constraint_type == "nodal":
                self.constraints_nodal.append(constraint)
            elif constraint.constraint_type == "ncvx_nodal":
                if constraint.nodes == 'All':
                    constraint.nodes = list(range(N))
                # constraint.g = 
                self.constraints_ncvx_nodal.append(constraint)
            else:
                raise ValueError(
                    f"Unknown constraint type: {constraint.constraint_type}, All constraints must be decorated with @ctcs or @nodal"
                )

        veh = Dynamics(
            dynamics,
            self.constraints_ctcs,
            self.constraints_nodal,  # TODO (norrisg) Maybe move this outside of the dynamics?
            self.constraints_ncvx_nodal,
            initial_state=initial_state,
            final_state=final_state,
        )

        self.params = Config(
            sim=sim,
            scp=scp,
            veh=veh,
        )

        self.ocp, self.aug_dy, self.cpg_solve = PTR_init(self.params)

    def solve(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        return PTR_main(self.params, self.ocp, self.aug_dy, self.cpg_solve)

    def post_process(self, result):
        return PTR_post(self.params, result, self.aug_dy)
