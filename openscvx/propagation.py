import numpy as np
import jax.numpy as jnp

def jax_interp1d(x, y):
    def interpolate(x_new):
        indices = np.searchsorted(x, x_new, side='left')
        indices = np.clip(indices, 1, len(x) - 1)
        
        x0, x1 = x[indices - 1], x[indices]
        y0, y1 = y[indices - 1], y[indices]
        
        slope = (y1 - y0) / (x1 - x0)
        y_new = y0 + slope * (x_new - x0)
        
        return y_new
    
    return interpolate


def u_lambda(u, t):
    """
    Generate a lambda function that linearly interpolates between the control input given a time.

    Parameters:
    u (np.ndarray): Array of control inputs (shape: m x n).
    t (np.ndarray): Array of time points corresponding to the control inputs (shape: n).
    params (dict): Additional parameters if needed.

    Returns:
    function: A lambda function that interpolates the control input at a given time.
    """

    # Ensure t is a 1D array
    t = t.flatten()

    u = jnp.asarray(u)
    t = jnp.asarray(t)
    interpolate = lambda new_t: jnp.array([jnp.interp(new_t, t, u[:, i]) for i in range(u.shape[1])]).T

    return interpolate