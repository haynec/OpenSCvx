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
    p.sim.n_controls = 1
    p.sim.S_x = jnp.eye(p.sim.n_states)
    p.sim.c_x = jnp.zeros(p.sim.n_states)
    p.sim.S_u = jnp.eye(p.sim.n_controls)
    p.sim.c_u = jnp.zeros(p.sim.n_controls)
    p.sim.inv_S_x = jnp.eye(p.sim.n_states)
    p.sim.inv_S_u = jnp.eye(p.sim.n_controls)
    p.scp = Dummy()
    p.scp.n = 5
    p.dis = Dummy()
    p.dis.custom_integrator = True
    p.dis.solver = "Tsit5"
    p.dis.rtol = 1e-3
    p.dis.atol = 1e-6
    p.dis.args = {}
    p.dis.dis_type = "FOH"
    p.dev = Dummy()
    p.dev.debug = False
    return p


def state_dot(x, u, node, params):
    # simple linear: x' = A_true x + B_true u
    return x + u


def A(x, u, node, params):
    batch = x.shape[0]
    eye = jnp.eye(2)
    return jnp.broadcast_to(eye, (batch, 2, 2))


def B(x, u, node, params):
    batch = x.shape[0]
    ones = jnp.ones((2, 1))
    return jnp.broadcast_to(ones, (batch, 2, 1))


@pytest.fixture
def dynamics():
    d = Dummy()
    d.f = state_dot
    return d


# --- tests ---------------------------------------------------------


def test_discretization_shapes(settings, dynamics):
    # build solver via LinearizeDiscretize
    discretizer = LinearizeDiscretize()
    solver = discretizer.get_solver(dynamics, settings)

    # dummy x,u
    x = jnp.ones((settings.scp.n, settings.sim.n_states))
    u = jnp.ones((settings.scp.n, settings.sim.n_controls + 1))  # +1 slack

    A_bar, B_bar, C_bar, x_prop, Vmulti = solver(x, u, {})

    # expected shapes
    N = settings.scp.n
    n_x, n_u = settings.sim.n_states, settings.sim.n_controls
    assert A_bar.shape == ((N - 1), n_x, n_x)
    assert B_bar.shape == ((N - 1), n_x, n_u)
    assert C_bar.shape == ((N - 1), n_x, n_u)
    assert x_prop.shape == ((N - 1), n_x)
    # assert Vmulti.shape == (N, (n_x + n_x*n_x + 2*n_x*n_u + n_x))


def test_jit_dVdt_compiles(settings):
    # prepare trivial inputs
    n_x, n_u = settings.sim.n_states, settings.sim.n_controls
    N = settings.scp.n
    aug_dim = n_x + n_x * n_x + 2 * n_x * n_u

    tau = jnp.array(0.3)
    V_flat = jnp.ones((N - 1) * aug_dim)
    u_cur = jnp.ones((N - 1, n_u + 1))
    u_next = jnp.ones((N - 1, n_u + 1))

    # bind out the Python callables & settings
    def wrapped(tau_, V_):
        return _dVdt(
            tau_,
            V_,
            u_cur,
            u_next,
            state_dot,
            A,
            B,
            n_x,
            n_u,
            N,
            settings.dis.dis_type,
            {},
            settings.sim.S_x,
            settings.sim.c_x,
            settings.sim.S_u,
            settings.sim.c_u,
            settings.sim.inv_S_x,
            settings.sim.inv_S_u,
        )

    # now JIT only over (tau_, V_)
    jitted = jax.jit(wrapped)
    lowered = jitted.lower(tau, V_flat)
    # compile will fail if there's a trace issue
    lowered.compile()


@pytest.mark.parametrize("integrator", ["custom_integrator", "diffrax"])
def test_jit_discretization_solver_compiles(settings, dynamics, integrator):
    # flip between the two modes
    if integrator == "custom_integrator":
        settings.dis.custom_integrator = True
    elif integrator == "diffrax":
        settings.dis.custom_integrator = False

    # build the solver via LinearizeDiscretize
    discretizer = LinearizeDiscretize()
    solver = discretizer.get_solver(dynamics, settings)

    # dummy x,u (including slack column)
    x = jnp.ones((settings.scp.n, settings.sim.n_states))
    u = jnp.ones((settings.scp.n, settings.sim.n_controls + 1))

    # jit & lower & compile
    jitted = jax.jit(solver)
    export.export(jitted)(x, u, {})
