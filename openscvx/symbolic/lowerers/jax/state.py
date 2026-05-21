"""JAX visitors for state/time/STM expressions.

Visitors: State, Time, STMPhysical, STMImpulse

STM leaves are no longer states. The discretizer propagates them along the
current iterate and exposes the per-node value via
``params["__stm_phi__"][name]`` (shape ``(N, n_phys, n_phys)`` for
``STMPhysical`` / ``(N, n_phys)`` for ``STMImpulse``, both flattened by the
visitor before returning). Approx-mode reads are wrapped in
``jax.lax.stop_gradient`` so the SCP Jacobian treats Φ as frozen.
"""

import jax
import jax.numpy as jnp

from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.stm import STMImpulse, STMPhysical
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


_STM_PHI_KEY = "__stm_phi__"


@visitor(STMPhysical)
@visitor(STMImpulse)
def _visit_stm(lowerer, node):
    """Lower an STM leaf (Φ or Φ_imp) to a per-node params lookup.

    The discretizer writes the propagated value of each STM into
    ``params[_STM_PHI_KEY][node.name]``; this visitor returns a callable
    that extracts the slice for the current node and reshapes to the
    symbolic leaf's declared flat shape. In ``mode="approx"`` the read is
    wrapped in ``jax.lax.stop_gradient`` so the SCP linearization treats
    Φ as a frozen input.
    """
    name = node.name
    mode = getattr(node, "mode", "approx")
    if mode == "approx":
        def _read(x, u, node_idx, params, _name=name):
            arr = params[_STM_PHI_KEY][_name]
            return jax.lax.stop_gradient(arr[node_idx].reshape(-1))
        return _read

    def _read(x, u, node_idx, params, _name=name):
        arr = params[_STM_PHI_KEY][_name]
        return arr[node_idx].reshape(-1)

    return _read


@visitor(Time)
@visitor(State)
def _visit_state(lowerer, node: State):
    """Lower a non-STM state (or Time) variable to a unified-state slice read."""
    sl = node._slice
    if sl is None:
        raise ValueError(f"State {node.name!r} has no slice assigned")
    return lambda x, u, node, params: x[sl]
