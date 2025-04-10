from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp


@dataclass
class Dynamics:
    def __init__(
        self,
        dynamics: callable,
        constraints_ctcs: List[callable],
        constraints_nodal: List[callable],
        constraints_ncvx_nodal: List[callable],
        initial_state,
        final_state,
    ):

        self.dynamics = dynamics
        self.constraints_ctcs = constraints_ctcs
        self.constraints_nodal = constraints_nodal
        self.constraints_ncvx_nodal = constraints_ncvx_nodal

        # CTCS Functions
        self.g_jit = jax.jit(self.g_func)
        self.g_vec = jax.vmap(self.g_jit, in_axes=(0, 0))

        # Dynamics Functions
        self.state_dot = jax.vmap(self.dyn_aug)
        self.A = jax.jit(jax.vmap(jax.jacfwd(self.dyn_aug, argnums=0), in_axes=(0, 0)))
        self.B = jax.jit(jax.vmap(jax.jacfwd(self.dyn_aug, argnums=1), in_axes=(0, 0)))

        self.initial_state = initial_state
        self.final_state = final_state

    def g_func(self, x: jnp.array, u: jnp.array) -> jnp.array:
        g_sum = 0
        for g in self.constraints_ctcs:
            g_sum += g(x, u)
        return g_sum

    def dyn_aug(self, x: jnp.array, u: jnp.array) -> jnp.array:
        # TODO: (norrisg) handle varying lengths of x and u due to augmentation more elegantly
        self.t_inds = -2
        self.y_inds = -1
        self.s_inds = -1

        x_dot = self.dynamics(x[:-1], u)
        t_dot = 1
        y_dot = self.g_jit(x, u)
        return jnp.hstack([x_dot, t_dot, y_dot])
