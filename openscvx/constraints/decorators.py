from jax import jit, vmap, jacfwd
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

def ncvx_nodal(func: callable, nodes = 'All'):
    """Decorator to mark a function as a 'ncvx_nodal' constraint."""
    # TODO: (haynec) switch to AOT instead of JIT
    func.constraint_type = "ncvx_nodal"
    func.g = vmap(jit(func), in_axes=(0, 0))
    func.grad_g_x = jit(vmap(jacfwd(func, argnums=0), in_axes=(0, 0)))
    func.grad_g_u = jit(vmap(jacfwd(func, argnums=1), in_axes=(0, 0)))
    func.nodes = nodes
    return func