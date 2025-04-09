import jax.numpy as jnp


def ctcs(func: callable, penalty="squared_relu") -> callable:
    """Decorator to mark a function as a 'ctcs' constraint."""
    func.constraint_type = "ctcs"
    if penalty == "squared_relu":
        func.penalty = lambda x: jnp.maximum(0, x) ** 2
    elif penalty == "huber":
        delta = 0.25
        func.penalty = lambda x: jnp.where(
            jnp.maximum(0, x) < delta,
            0.5 * jnp.maximum(0, x) ** 2,
            jnp.maximum(0, x) - 0.5 * delta,
        )
    elif penalty == "smooth_relu":
        c = 1e-8
        func.penalty = lambda x: (jnp.maximum(0, x) ** 2 + c**2) ** 0.5 - c
    else:
        func.penalty = penalty
    return func


def nodal(func: callable, nodes: list[int] = None) -> callable:
    """Decorator to mark a function as a 'nodal' constraint."""
    func.constraint_type = "nodal"
    func.nodes = nodes
    return func
