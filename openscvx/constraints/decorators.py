import jax.numpy as jnp

def ctcs(func: callable, penalty = 'Default'):
    """Decorator to mark a function as a 'ctcs' constraint."""
    func.constraint_type = "ctcs"
    if penalty == 'Default':
        func.penalty = lambda x: jnp.maximum(0, x)**2
    else:
        func.penalty = penalty
    return func

def nodal(func: callable):
    """Decorator to mark a function as a 'nodal' constraint."""
    # TODO: (norrisg) add ability to specify which nodes to apply the constraint to
    func.constraint_type = "nodal"
    return func