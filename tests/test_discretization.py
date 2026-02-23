import jax
import jax.numpy as jnp
import pytest
from jax import export

from openscvx.discretization import LinearizeDiscretize
from openscvx.discretization.linearize_discretize import _dVdt

# --- fixtures for dummy params, state_dot, A, B  ------------------


# dummy parameter namespace
class Dummy:
    pass


@pytest.fixture
def settings():
    p = Dummy()
    p.sim = Dummy()
    p.sim.n_states = 2
    p.sim.n_controls = 2  # 1 vehicle control + 1 time-dilation (unified)
    p.sim.S_x = jnp.eye(p.sim.n_states)
    p.sim.c_x = jnp.zeros(p.sim.n_states)
    p.sim.S_u = jnp.eye(p.sim.n_controls)
    p.sim.c_u = jnp.zeros(p.sim.n_controls)
    p.sim.inv_S_x = jnp.eye(p.sim.n_states)
    p.sim.inv_S_u = jnp.eye(p.sim.n_controls)
    p.sim.n = 5
    p.dev = Dummy()
    p.dev.debug = False
    return p


def state_dot(x, u, node, params):
    # simple time-dilated dynamics: x' = s * (x + u_vehicle)
    # u = [u_vehicle, s] includes both vehicle control and time-dilation
    # This is the un-vmapped version (single sample, not batched)
    s = u[1]
    u_v = u[0]
    return s * (x + u_v)


@pytest.fixture
def dynamics():
    d = Dummy()
    d.f = state_dot
    return d


# --- tests ---------------------------------------------------------


def test_discretization_shapes(settings, dynamics):
    # build solver via LinearizeDiscretize (custom_integrator for speed)
    discretizer = LinearizeDiscretize(custom_integrator=True)
    solver = discretizer.get_solver(dynamics, settings)

    # dummy x,u (n_controls already includes time-dilation)
    x = jnp.ones((settings.sim.n, settings.sim.n_states))
    u = jnp.ones((settings.sim.n, settings.sim.n_controls))

    A_bar, B_bar, C_bar, x_prop, Vmulti = solver(x, u, {})

    # expected shapes
    N = settings.sim.n
    n_x, n_u = settings.sim.n_states, settings.sim.n_controls
    assert A_bar.shape == ((N - 1), n_x, n_x)
    assert B_bar.shape == ((N - 1), n_x, n_u)
    assert C_bar.shape == ((N - 1), n_x, n_u)
    assert x_prop.shape == ((N - 1), n_x)


def test_jit_dVdt_compiles(settings):
    # prepare trivial inputs (n_u already includes time-dilation)
    n_x, n_u = settings.sim.n_states, settings.sim.n_controls
    N = settings.sim.n
    aug_dim = n_x + n_x * n_x + 2 * n_x * n_u

    tau = jnp.array(0.3)
    V_flat = jnp.ones((N - 1) * aug_dim)
    u_cur = jnp.ones((N - 1, n_u))
    u_next = jnp.ones((N - 1, n_u))

    # Create vmapped versions of dynamics and Jacobians (as _dVdt expects)
    f_vmapped = jax.vmap(state_dot, in_axes=(0, 0, 0, None))
    A_vmapped = jax.vmap(jax.jacfwd(state_dot, argnums=0), in_axes=(0, 0, 0, None))
    B_vmapped = jax.vmap(jax.jacfwd(state_dot, argnums=1), in_axes=(0, 0, 0, None))

    # bind out the Python callables & settings
    def wrapped(tau_, V_):
        return _dVdt(
            tau_,
            V_,
            u_cur,
            u_next,
            f_vmapped,
            A_vmapped,
            B_vmapped,
            n_x,
            n_u,
            N,
            "FOH",
            settings.sim.S_x,
            settings.sim.c_x,
            settings.sim.S_u,
            settings.sim.c_u,
            settings.sim.inv_S_x,
            settings.sim.inv_S_u,
            {},
        )

    # now JIT only over (tau_, V_)
    jitted = jax.jit(wrapped)
    lowered = jitted.lower(tau, V_flat)
    # compile will fail if there's a trace issue
    lowered.compile()


@pytest.mark.parametrize("integrator", ["custom_integrator", "diffrax"])
def test_jit_discretization_solver_compiles(settings, dynamics, integrator):
    # build the solver via LinearizeDiscretize with the chosen integrator
    discretizer = LinearizeDiscretize(custom_integrator=(integrator == "custom_integrator"))
    solver = discretizer.get_solver(dynamics, settings)

    # dummy x,u (n_controls already includes time-dilation)
    x = jnp.ones((settings.sim.n, settings.sim.n_states))
    u = jnp.ones((settings.sim.n, settings.sim.n_controls))

    # jit & lower & compile
    jitted = jax.jit(solver)
    export.export(jitted)(x, u, {})
