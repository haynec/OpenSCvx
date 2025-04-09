import jax.numpy as jnp


def ctcs(func: callable, penalty="squared_relu"):
    """Decorator to mark a function as a 'ctcs' constraint."""
    func.constraint_type = "ctcs"
    if penalty == "squared_relu":
        func.penalty = lambda x: jnp.maximum(0, x) ** 2
    elif penalty == "huber":
        # TODO: (norrisg) figure out why this doesn't work :(
        delta = 10.0
        func.penalty = lambda x: jnp.maximum(
            0, jnp.where(jnp.abs(x) < delta, 0.5 * x**2, jnp.abs(x) - 0.5 * delta)
        )
    else:
        func.penalty = penalty
    return func


def nodal(func: callable, nodes: list = None):
    """Decorator to mark a function as a 'nodal' constraint."""
    func.constraint_type = "nodal"
    func.nodes = nodes
    return func
