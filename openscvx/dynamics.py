from abc import abstractmethod, ABC

import jax
import jax.numpy as jnp


class Dynamics(ABC):
    def __init__(self):

        # CTCS Functions
        self.g_jit = jax.jit(self.g_func)
        self.g_vec = jax.vmap(self.g_jit, in_axes=(0))

        # Dynamics Functions
        self.state_dot = jax.vmap(self.dynamics)
        self.A = jax.jit(jax.vmap(jax.jacfwd(self.dynamics, argnums=0), in_axes=(0, 0)))
        self.B = jax.jit(jax.vmap(jax.jacfwd(self.dynamics, argnums=1), in_axes=(0, 0)))

    @abstractmethod
    def dynamics(self, x: jnp.array, u: jnp.array) -> jnp.array:
        pass

    @abstractmethod
    def g_func(self, x: jnp.array) -> jnp.array:
        pass
