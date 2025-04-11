from jax import jit, vmap, jacfwd
import jax.numpy as jnp


def ctcs(func: callable, penalty="squared_relu") -> callable:
    """Decorator to mark a function as a 'ctcs' constraint."""
    func.constraint_type = "ctcs"
    if penalty == "squared_relu":
        func.penalty = lambda x: jnp.maximum(0, x) ** 2
    elif penalty == "huber":
        # https://en.wikipedia.org/wiki/Huber_loss
        delta = 0.25
        func.penalty = lambda x: jnp.where(
            jnp.maximum(0, x) < delta,
            0.5 * jnp.maximum(0, x) ** 2,
            jnp.maximum(0, x) - 0.5 * delta,
        )
    elif penalty == "smooth_relu":
        # https://arxiv.org/pdf/2405.10996
        c = 1e-8
        func.penalty = lambda x: (jnp.maximum(0, x) ** 2 + c**2) ** 0.5 - c
    else:
        func.penalty = penalty
    return func


def nodal(func: callable, nodes: list[int] = None, convex: bool = False) -> callable:
    """Decorator to mark a function as a 'nodal' constraint."""
    func.constraint_type = "nodal"
    func.nodes = nodes
    func.convex = convex
    if not convex:
        # TODO: (haynec) switch to AOT instead of JIT
        func.g = vmap(jit(func), in_axes=(0, 0))
        func.grad_g_x = jit(vmap(jacfwd(func, argnums=0), in_axes=(0, 0)))
        func.grad_g_u = jit(vmap(jacfwd(func, argnums=1), in_axes=(0, 0)))
    return func