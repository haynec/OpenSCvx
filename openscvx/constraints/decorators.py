import jax.numpy as jnp

def ctcs(func: callable):
    """Decorator to mark a function as a 'ctcs' constraint."""
    func.constraint_type = "ctcs"
    return func

def nodal(func: callable):
    """Decorator to mark a function as a 'nodal' constraint."""
    # TODO: (norrisg) add ability to specify which nodes to apply the constraint to
    func.constraint_type = "nodal"
    return func