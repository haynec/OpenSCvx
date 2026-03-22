from dataclasses import dataclass, field
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class Dynamics:
    """Dataclass to hold a system dynamics function and its Jacobians.

    This dataclass is used internally by openscvx to store the compiled dynamics
    function and its gradients after symbolic expressions are lowered to JAX.
    Users typically don't instantiate this class directly.

    Attributes:
        f (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
            Function defining the continuous time nonlinear system dynamics
            as x_dot = f(x, u, ...params).
            - x: 1D array (state at a single node), shape (n_x,)
            - u: 1D array (control at a single node), shape (n_u,)
            - Additional parameters: passed as keyword arguments with names
              matching the parameter name plus an underscore (e.g., g_ for
              Parameter('g')).
            If you use vectorized integration or batch evaluation, x and u
            may be 2D arrays (N, n_x) and (N, n_u).
        A (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of ``f`` w.r.t. ``x``.
        B (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of ``f`` w.r.t. ``u``.
        A_c_sparsity (Optional[np.ndarray]): Boolean ``(n_x, n_x)`` array
            giving the structural sparsity of the continuous-time state
            Jacobian ``df/dx``.  Computed from the symbolic AST.
        B_c_sparsity (Optional[np.ndarray]): Boolean ``(n_x, n_u)`` array
            giving the structural sparsity of the continuous-time control
            Jacobian ``df/du``.  Computed from the symbolic AST.

    !!! note
        The ``A`` and ``B`` fields are kept for convenience but may be
        deprecated in the future. Jacobian computation is now handled by
        the :class:`~openscvx.discretization.base.Discretizer`, so these
        fields are no longer set during lowering or used internally.
    """

    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    A: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    B: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    A_c_sparsity: Optional[np.ndarray] = field(default=None, repr=False)
    B_c_sparsity: Optional[np.ndarray] = field(default=None, repr=False)
