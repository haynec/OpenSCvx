from abc import abstractmethod, ABC

import jax
import jax.numpy as jnp


class Dynamics(ABC):
    def __post_init__(self):

        # CTCS Functions
        self.g_jit = jax.jit(self.g_func)
        self.g_vec = jax.vmap(self.g_jit, in_axes=(0))

        # Dynamics Functions
        self.state_dot = jax.vmap(self.dynamics_augmented)
        self.A = jax.jit(jax.vmap(jax.jacfwd(self.dynamics_augmented, argnums=0), in_axes=(0, 0)))
        self.B = jax.jit(jax.vmap(jax.jacfwd(self.dynamics_augmented, argnums=1), in_axes=(0, 0)))

    @abstractmethod
    def dynamics(self, x: jnp.array, u: jnp.array) -> jnp.array:
        pass

    @abstractmethod
    def g_func(self, x: jnp.array) -> jnp.array:
        pass
    
    def dynamics_augmented(self, x: jnp.array, u: jnp.array) -> jnp.array:
        # TODO: (norrisg) handle varying lengths of x and u due to augmentation more elegantly
        x_dot = self.dynamics(x[:-1], u)
        y_dot = self.g_jit(x)
        return jnp.hstack([x_dot, y_dot])