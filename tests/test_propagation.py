# test_propagation.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

from openscvx.propagation import get_propagation_solver, prop_aug_dy, s_to_t, t_to_tau
from openscvx.symbolic.expr import Control, State


# simple scalar decay with time-dilation: x' = s * (-x)
# u[:, -1] is the time-dilation factor s (included symbolically)
def decay(x, u, node, params):
    return u[:, -1:] * (-x)


class Dummy:
    @property
    def time_slice(self):
        """Mock property to return idx_t for backward compatibility."""
        return self.idx_t if hasattr(self, "idx_t") else None

    @property
    def time_dilation_slice(self):
        """Mock property to return idx_s for backward compatibility."""
        return self.idx_s if hasattr(self, "idx_s") else None


@pytest.mark.parametrize("dis_type,beta_expected", [("ZOH", 0.0), ("FOH", 1.0)])
def test_prop_aug_dy_linear(dis_type, beta_expected):
    """
    prop_aug_dy should compute:
      u = u_cur + beta*(u_next - u_cur)
      return state_dot(x_batch, u).squeeze()
    for both ZOH (beta=0) and FOH (beta=(tau-tau_init)*N).
    Time-dilation is already included in state_dot symbolically.
    """
    tau = 0.2
    tau_init = 0.0
    N = 5
    x = np.array([1.0, 2.0])
    u_cur = np.array([[0.5, 3.0]])
    u_next = np.array([[1.5, 5.0]])

    node = 0  # dummy node index

    # compute beta
    beta = 0.0 if dis_type == "ZOH" else (tau - tau_init) * N
    assert pytest.approx(beta) == beta_expected

    # manually compute expected
    u = u_cur + beta * (u_next - u_cur)
    # state_dot already includes time-dilation: s * (x + u_vehicle)
    # Here we use a simple mock where the full u vector is passed through
    expected = u[:, 1] * (x + u[:, 0])

    out = prop_aug_dy(
        tau,
        x,
        u_cur,
        u_next,
        tau_init,
        node,
        # state_dot includes time-dilation: s * (x + u_vehicle)
        lambda x_batch, u_full, node, params: u_full[:, 1:] * (x_batch + u_full[:, :1]),
        dis_type,
        N,
        {},
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_s_to_t_basic(dis_type):
    """
    s_to_t should accumulate time steps correctly under both ZOH and FOH.
    """
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 4
    p.sim = Dummy()
    p.sim.initial_state = Dummy()
    p.sim.initial_state.value = np.array([0])
    p.sim.idx_t = slice(0, 1)

    # build u with slack values [1,2,3,4]
    u = Control("u", shape=(2,))  # 2 controls, last is slack
    u.guess = np.stack([[0.0, float(s)] for s in [1, 2, 3, 4]])
    x = State("x", shape=(1,))  # dummy initial state
    x.guess = np.array([[0.0], [1.0]])
    # Pass arrays instead of State/Control objects
    t = s_to_t(x.guess, u.guess, p, dis_type)

    # manually reconstruct expected t
    tau = np.linspace(0, 1, p.scp.n)
    expected = [0.0]
    for k in range(1, p.scp.n):
        s_kp = u.guess[k - 1, -1]
        s_k = u.guess[k, -1]
        if dis_type == "ZOH":
            dt = (tau[k] - tau[k - 1]) * s_kp
        else:
            dt = 0.5 * (s_k + s_kp) * (tau[k] - tau[k - 1])
        expected.append(expected[-1] + dt)

    np.testing.assert_allclose(np.array(t).squeeze(), np.array(expected).squeeze(), rtol=1e-6)


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_t_to_tau_constant_slack(dis_type):
    """
    t_to_tau should invert s_to_t back to the original tau grid when slack is constant.
    Also, the interpolated u should exactly match u_nodal in that case.
    """
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 4
    p.sim = Dummy()
    p.sim.initial_state = Dummy()
    p.sim.initial_state.value = np.array([0])
    p.sim.idx_t = slice(0, 1)

    # constant slack = 2.0, control doesn't matter
    x = State("x", shape=(1,))  # dummy initial state
    x.guess = np.array([[0.0], [1.0]])  # dummy initial state guess

    N = p.scp.n

    u = Control("u", shape=(2,))  # 2 controls, last is slack
    u.guess = np.tile(np.array([0.0, 2.0]), (N, 1))  # constant slack of 2.0

    # get the "nodal" times via s_to_t - pass arrays instead of State/Control objects
    t_nodal = s_to_t(x.guess, u.guess, p, dis_type)

    # invert back - pass array instead of Control object
    tau, u_interp = t_to_tau(
        u.guess, np.array(t_nodal).squeeze(), np.array(t_nodal).squeeze(), p, dis_type
    )

    np.testing.assert_allclose(tau, np.linspace(0, 1, N), rtol=1e-6)
    # since slack & control are constant, interpolation must reprodu


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_propagation_solver_decay(dis_type):
    """
    Propagation solver should approximate exp(-t) over [0,1] at t=0.5 with ~1% error.
    """
    # Build dummy params
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 2  # only one segment needed
    p.prp = Dummy()
    p.prp.solver = "Tsit5"
    p.prp.rtol = 1e-6
    p.prp.atol = 1e-3
    p.prp.args = {}
    p.sim = Dummy()

    solver = get_propagation_solver(decay, p, dis_type)

    # Initial conditions
    V0 = jnp.array([1.0])
    tau_grid = jnp.array([0.0, 1.0])
    u_cur = jnp.array([[0.0, 1.0]])  # slack = 1
    u_next = jnp.array([[0.0, 1.0]])  # slack = 1
    tau_init = jnp.array([[0.0]])
    node = jnp.array([[0]])

    # We only care about t = 0.5
    save_time = jnp.array([0.5])
    mask = jnp.array([True])  # Only one point

    # Call the solver
    sol = solver(V0, tau_grid, u_cur, u_next, tau_init, node, save_time, mask, {})

    # Extract solution
    y_half = float(sol[0][0])

    # Check against exact solution
    expected = np.exp(-0.5)
    assert np.isclose(y_half, expected, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_jit_propagation_solver_compiles(dis_type):
    """
    Ensure that the propagation solver's .call output can be jitted and exported without errors.
    """

    # — build dummy params —
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 5
    p.prp = Dummy()
    p.prp.solver = "Tsit5"
    p.prp.rtol = 1e-6
    p.prp.atol = 1e-3
    p.prp.args = {}
    p.sim = Dummy()

    solver = get_propagation_solver(decay, p, dis_type)

    # — dummy inputs —
    V0 = jnp.array([1.0])
    tau_grid = jnp.array([0.0, 1.0])
    u_cur = jnp.array([[0.0, 1.0]])
    u_next = jnp.array([[0.0, 1.0]])
    tau_init = jnp.array([[0.0]])
    node = jnp.array([[0]])

    MAX_TAU_LEN = 20
    save_time = jnp.linspace(0.0, 1.0, MAX_TAU_LEN)
    mask = jnp.ones_like(save_time, dtype=bool)

    # JIT and export the solver
    jitted = jax.jit(
        lambda V0, tau_grid, u_cur, u_next, tau_init, node, save_time, mask: solver(
            V0, tau_grid, u_cur, u_next, tau_init, node, save_time, mask, {}
        )
    )

    # Export
    exported = export.export(jitted)(V0, tau_grid, u_cur, u_next, tau_init, node, save_time, mask)
    exported.serialize()
