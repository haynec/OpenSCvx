"""``solve_batched(save_compiled=True)`` refuses the CVXPy backend.

CVXPy's :meth:`iteration_callback` wraps the host solve in a
``jax.pure_callback``, which ``jax.export`` cannot serialize. Rather than
silently degrading to an uncached in-process solve — defeating the whole reason
the user set ``save_compiled`` — ``solve_batched`` raises a teaching error
pointing at the exportable backends. With ``save_compiled=False`` the same
problem still runs ``B`` sequential CVXPy solves.
"""

import jax.numpy as jnp
import pytest

from tests.solvers._iteration_callback_helpers import build_brachistochrone


def test_cvxpy_export_raises_teaching_error():
    prob = build_brachistochrone("cvxpy", n=8, k_max=2)
    prob.settings.sim.save_compiled = True
    prob.initialize()

    x_init = prob.state.x_init_pin
    x_term = prob.state.x_term_pin
    x0_stack = jnp.broadcast_to(x_init, (3,) + x_init.shape)
    xf_stack = jnp.broadcast_to(x_term, (3,) + x_term.shape)

    with pytest.raises(ValueError, match="pure_callback|QPAX|Moreau"):
        prob.solve_batched(x0_stack, xf_stack)


def test_cvxpy_sequential_path_still_works_without_export():
    prob = build_brachistochrone("cvxpy", n=8, k_max=20)
    prob.settings.sim.save_compiled = False
    prob.initialize()

    x_init = prob.state.x_init_pin
    x_term = prob.state.x_term_pin
    x0_stack = jnp.stack([x_init.at[0].set(x_init[0] + s) for s in (0.0, 0.3)])
    xf_stack = jnp.broadcast_to(x_term, x0_stack.shape)

    batched = prob.solve_batched(x0_stack, xf_stack)
    assert batched.x.shape == (2, 8, x_init.shape[0])
