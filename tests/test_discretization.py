from types import SimpleNamespace

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

from openscvx.discretization import (
    LinearizeDiscretize,
    LinearizeDiscretizeSparse,
    color_columns,
    make_sparse_jacobian_fns,
    resolve_discretizer_config,
)
from openscvx.discretization.base import _make_foh_mask, _resolve_foh_mask
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
    p.sim.u = SimpleNamespace(foh_mask=None)
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
    settings.dev.debug = True
    discretizer = LinearizeDiscretize()
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
    foh_mask = np.ones(n_u, dtype=np.float64)

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
            foh_mask,
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


def test_jit_discretization_solver_compiles(settings, dynamics):
    settings.dev.debug = True  # force RK45 path
    discretizer = LinearizeDiscretize()
    solver = discretizer.get_solver(dynamics, settings)

    # dummy x,u (n_controls already includes time-dilation)
    x = jnp.ones((settings.sim.n, settings.sim.n_states))
    u = jnp.ones((settings.sim.n, settings.sim.n_controls))

    # jit & lower & compile
    jitted = jax.jit(solver)
    export.export(jitted)(x, u, {})


# --- sparse discretization tests ----------------------------------------


def _rocket_dynamics(x, u, node, params):
    """3-DOF rocket: pos(3), vel(3), mass(1) with thrust(3).

    Jacobian A_c has clear sparsity: position rows depend only on velocity,
    mass row depends only on thrust magnitude (through controls, not states
    beyond mass).
    """
    vel = x[3:6]
    m = x[6]
    T = u[:3]
    s = u[3]  # time-dilation
    g = 3.71
    Isp_ge = 225.0 * 9.807
    T_norm = jnp.sqrt(jnp.sum(T**2) + 1e-8)
    pos_dot = vel
    vel_dot = T / m - jnp.array([0.0, 0.0, g])
    mass_dot = -T_norm / Isp_ge
    return s * jnp.concatenate([pos_dot, vel_dot, jnp.array([mass_dot])])


@pytest.fixture
def rocket_settings():
    p = Dummy()
    p.sim = Dummy()
    p.sim.n_states = 7
    p.sim.n_controls = 4  # thrust(3) + time-dilation(1)
    p.sim.u = SimpleNamespace(foh_mask=None)
    p.sim.S_x = jnp.eye(7)
    p.sim.c_x = jnp.zeros(7)
    p.sim.S_u = jnp.eye(4)
    p.sim.c_u = jnp.zeros(4)
    p.sim.inv_S_x = jnp.eye(7)
    p.sim.inv_S_u = jnp.eye(4)
    p.sim.n = 8
    p.dev = Dummy()
    p.dev.debug = False
    return p


@pytest.fixture
def rocket_dynamics():
    d = Dummy()
    d.f = _rocket_dynamics
    d.A_c_sparsity = None
    d.B_c_sparsity = None
    return d


def _get_rocket_sparsity():
    """Compute the A_c and B_c boolean sparsity patterns for the rocket.

    Evaluates at multiple random points to capture all structural nonzeros
    (some entries may be zero at a particular operating point but are
    structurally nonzero).
    """
    n_x, n_u = 7, 4
    rng = np.random.default_rng(0)
    A_c = np.zeros((n_x, n_x), dtype=bool)
    B_c = np.zeros((n_x, n_u), dtype=bool)
    for _ in range(5):
        x0 = jnp.array(rng.normal(size=n_x))
        x0 = x0.at[6].set(jnp.abs(x0[6]) + 100.0)
        u0 = jnp.array(rng.normal(size=n_u))
        u0 = u0.at[3].set(jnp.abs(u0[3]) + 0.1)
        A = jax.jacfwd(_rocket_dynamics, argnums=0)(x0, u0, 0, {})
        B = jax.jacfwd(_rocket_dynamics, argnums=1)(x0, u0, 0, {})
        A_c |= np.array(np.abs(A) > 1e-12)
        B_c |= np.array(np.abs(B) > 1e-12)
    return A_c, B_c


def test_color_columns_validity():
    """Coloring respects the structural orthogonality constraint."""
    A_c, _ = _get_rocket_sparsity()
    colors = color_columns(A_c)
    n = A_c.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                # Same color → no shared nonzero row
                shared = A_c[:, i] & A_c[:, j]
                assert not shared.any(), (
                    f"Columns {i} and {j} share color {colors[i]} but have overlapping nonzero rows"
                )


def test_sparse_jacobian_matches_dense():
    """Sparse Jacobian via coloring matches dense jacfwd numerically."""
    A_c, B_c = _get_rocket_sparsity()
    n_x, n_u = 7, 4
    A_vm, B_vm = make_sparse_jacobian_fns(_rocket_dynamics, A_c, B_c, n_x, n_u)

    A_dense_fn = jax.vmap(jax.jacfwd(_rocket_dynamics, 0), in_axes=(0, 0, 0, None))
    B_dense_fn = jax.vmap(jax.jacfwd(_rocket_dynamics, 1), in_axes=(0, 0, 0, None))

    rng = np.random.default_rng(42)
    batch = 4
    x = jnp.array(rng.normal(size=(batch, n_x)))
    x = x.at[:, 6].set(jnp.abs(x[:, 6]) + 100.0)  # mass > 0
    u = jnp.array(rng.normal(size=(batch, n_u)))
    u = u.at[:, 3].set(jnp.abs(u[:, 3]) + 0.1)  # time-dilation > 0
    nodes = jnp.arange(batch)

    A_sp = A_vm(x, u, nodes, {})
    A_dn = A_dense_fn(x, u, nodes, {})
    np.testing.assert_allclose(np.array(A_sp), np.array(A_dn), atol=1e-5)

    B_sp = B_vm(x, u, nodes, {})
    B_dn = B_dense_fn(x, u, nodes, {})
    np.testing.assert_allclose(np.array(B_sp), np.array(B_dn), atol=1e-5)


def test_sparse_discretization_matches_dense(rocket_settings, rocket_dynamics):
    """Sparse and dense discretization paths produce matching A_d, B_d, C_d."""
    rocket_settings.dev.debug = True
    A_c, B_c = _get_rocket_sparsity()

    # Dense path
    dense_disc = LinearizeDiscretize(dis_type="FOH")
    dense_solver = dense_disc.get_solver(rocket_dynamics, rocket_settings)

    # Sparse path
    rocket_dynamics.A_c_sparsity = A_c
    rocket_dynamics.B_c_sparsity = B_c
    sparse_disc = LinearizeDiscretizeSparse(dis_type="FOH")
    sparse_solver = sparse_disc.get_solver(rocket_dynamics, rocket_settings)

    N = rocket_settings.sim.n
    n_x = rocket_settings.sim.n_states
    n_u = rocket_settings.sim.n_controls

    rng = np.random.default_rng(123)
    x = jnp.array(rng.normal(size=(N, n_x))) * 10.0
    x = x.at[:, 6].set(jnp.abs(x[:, 6]) + 100.0)
    u = jnp.array(rng.normal(size=(N, n_u)))
    u = u.at[:, 3].set(jnp.abs(u[:, 3]) + 0.5)

    A_dense, B_dense, C_dense, xp_dense, _ = dense_solver(x, u, {})
    A_sparse, B_sparse, C_sparse, xp_sparse, _ = sparse_solver(x, u, {})

    np.testing.assert_allclose(np.array(A_sparse), np.array(A_dense), atol=1e-4)
    np.testing.assert_allclose(np.array(B_sparse), np.array(B_dense), atol=1e-4)
    np.testing.assert_allclose(np.array(C_sparse), np.array(C_dense), atol=1e-4)
    np.testing.assert_allclose(np.array(xp_sparse), np.array(xp_dense), atol=1e-4)


@pytest.mark.parametrize("dis_type", ["FOH", "ZOH"])
def test_compact_v_matches_dense(rocket_settings, rocket_dynamics, dis_type):
    """Compact-V (sparse) integration produces the same A_d, B_d, C_d as dense."""
    rocket_settings.dev.debug = True
    A_c, B_c = _get_rocket_sparsity()
    n_x = rocket_settings.sim.n_states
    n_u = rocket_settings.sim.n_controls
    N = rocket_settings.sim.n

    from openscvx.symbolic.sparsity import discrete_sparsity

    Ad_pat, Bd_pat, Cd_pat = discrete_sparsity(A_c, B_c, dis_type)
    nnz_Ad = int(Ad_pat.sum())
    nnz_Bd = int(Bd_pat.sum())
    nnz_Cd = int(Cd_pat.sum())
    aug_dim_dense = n_x + n_x * n_x + 2 * n_x * n_u
    aug_dim_compact = n_x + nnz_Ad + nnz_Bd + nnz_Cd
    assert aug_dim_compact < aug_dim_dense, (
        f"compact ({aug_dim_compact}) should be smaller than dense ({aug_dim_dense})"
    )

    dense_disc = LinearizeDiscretize(dis_type=dis_type)
    dense_solver = dense_disc.get_solver(rocket_dynamics, rocket_settings)

    rocket_dynamics.A_c_sparsity = A_c
    rocket_dynamics.B_c_sparsity = B_c
    sparse_disc = LinearizeDiscretizeSparse(dis_type=dis_type)
    sparse_solver = sparse_disc.get_solver(rocket_dynamics, rocket_settings)

    rng = np.random.default_rng(99)
    x = jnp.array(rng.normal(size=(N, n_x))) * 10.0
    x = x.at[:, 6].set(jnp.abs(x[:, 6]) + 100.0)
    u = jnp.array(rng.normal(size=(N, n_u)))
    u = u.at[:, 3].set(jnp.abs(u[:, 3]) + 0.5)

    A_dense, B_dense, C_dense, xp_dense, _ = dense_solver(x, u, {})
    A_sparse, B_sparse, C_sparse, xp_sparse, V_sparse = sparse_solver(x, u, {})

    np.testing.assert_allclose(np.array(A_sparse), np.array(A_dense), atol=1e-4)
    np.testing.assert_allclose(np.array(B_sparse), np.array(B_dense), atol=1e-4)
    np.testing.assert_allclose(np.array(C_sparse), np.array(C_dense), atol=1e-4)
    np.testing.assert_allclose(np.array(xp_sparse), np.array(xp_dense), atol=1e-4)

    # Verify that the dense-reconstructed Vmulti is compatible with from_V
    from openscvx.algorithms.history import DiscretizationResult

    disc_result = DiscretizationResult.from_V(np.asarray(V_sparse), n_x=n_x, n_u=n_u, N=N)
    np.testing.assert_allclose(disc_result.A_d, np.array(A_dense), atol=1e-4)
    np.testing.assert_allclose(disc_result.B_d, np.array(B_dense), atol=1e-4)


def test_sparse_fallback_when_dense(settings, dynamics):
    """When Jacobian is fully dense, sparse path falls back to dense jacfwd."""
    settings.dev.debug = True
    dynamics.A_c_sparsity = np.ones((2, 2), dtype=bool)
    dynamics.B_c_sparsity = np.ones((2, 2), dtype=bool)
    disc = LinearizeDiscretizeSparse()
    solver = disc.get_solver(dynamics, settings)

    x = jnp.ones((settings.sim.n, settings.sim.n_states))
    u = jnp.ones((settings.sim.n, settings.sim.n_controls))
    A_d, B_d, C_d, xp, _ = solver(x, u, {})

    assert A_d.shape == (settings.sim.n - 1, 2, 2)


def test_sparse_fallback_when_no_sparsity(settings, dynamics):
    """When no sparsity info is available, LinearizeDiscretizeSparse falls back to dense."""
    settings.dev.debug = True
    disc = LinearizeDiscretizeSparse()
    solver = disc.get_solver(dynamics, settings)

    x = jnp.ones((settings.sim.n, settings.sim.n_states))
    u = jnp.ones((settings.sim.n, settings.sim.n_controls))
    A_d, B_d, C_d, xp, _ = solver(x, u, {})

    assert A_d.shape == (settings.sim.n - 1, 2, 2)


def test_discretizer_spec_build_preserves_stepsize_controller():
    """Dict-config discretizers must not serialize Diffrax controllers to dicts."""
    controller = dfx.StepTo(ts=np.linspace(0.0, 1.0, 12))
    disc = resolve_discretizer_config(
        {"diffrax_kwargs": {"stepsize_controller": controller}}
    ).build()
    assert disc.diffrax_kwargs["stepsize_controller"] is controller


def test_diffrax_kwargs_routed_to_diffrax_kwargs():
    controller = dfx.ConstantStepSize()
    disc = LinearizeDiscretize(
        diffrax_kwargs={
            "solver_name": "Dopri5",
            "rtol": 1e-7,
            "atol": 1e-8,
            "num_substeps": 13,
            "stepsize_controller": controller,
            "max_steps": 2048,
        }
    )

    kwargs = disc._resolve_diffrax_kwargs()
    assert kwargs["solver_name"] == "Dopri5"
    assert kwargs["rtol"] == pytest.approx(1e-7)
    assert kwargs["atol"] == pytest.approx(1e-8)
    assert kwargs["num_substeps"] == 13
    assert kwargs["extra_kwargs"]["stepsize_controller"] is controller
    assert kwargs["extra_kwargs"]["max_steps"] == 2048


def test_diffrax_kwargs_can_set_tolerances():
    disc = LinearizeDiscretize(
        diffrax_kwargs={"rtol": 1e-7, "atol": 1e-8},
    )
    kwargs = disc._resolve_diffrax_kwargs()
    assert kwargs["rtol"] == pytest.approx(1e-7)
    assert kwargs["atol"] == pytest.approx(1e-8)


def test_default_diffrax_tolerances_when_not_set():
    disc = LinearizeDiscretize()
    kwargs = disc._resolve_diffrax_kwargs()
    assert kwargs["rtol"] == pytest.approx(1e-6)
    assert kwargs["atol"] == pytest.approx(1e-3)


def test_make_foh_mask_string_and_sequence():
    np.testing.assert_array_equal(_make_foh_mask("FOH", 3), [True, True, True])
    np.testing.assert_array_equal(_make_foh_mask("ZOH", 2), [False, False])
    np.testing.assert_array_equal(
        _make_foh_mask(["FOH", "ZOH", "FOH"], 3),
        [True, False, True],
    )


def test_resolve_foh_mask_merges_unified_mask_with_discretizer_default():
    """Explicit entries in ``u_foh_mask`` override ``dis_type``; ``nan`` defers."""
    u_partial = np.array([np.nan, 0.0, 1.0], dtype=float)
    merged_foh = _resolve_foh_mask("FOH", 3, u_partial)
    np.testing.assert_array_equal(merged_foh, [1.0, 0.0, 1.0])

    merged_zoh = _resolve_foh_mask("ZOH", 3, u_partial)
    np.testing.assert_array_equal(merged_zoh, [0.0, 0.0, 1.0])

    assert _resolve_foh_mask("ZOH", 2, None).tolist() == [0.0, 0.0]


def test_diffrax_kwargs_routed_to_rk45_kwargs():
    """Fixed-step RK45 only reads num_substeps / tau_0; Diffrax-only keys are dropped."""
    disc = LinearizeDiscretize(
        diffrax_kwargs={
            "num_substeps": 77,
            "tau_0": 0.2,
            # Adaptive / Diffrax stepping — not applicable to fixed-step RK45
            "stepsize_controller": dfx.ConstantStepSize(),
        },
    )
    kwargs = disc._resolve_rk45_kwargs(is_not_compiled=False)
    assert kwargs["num_substeps"] == 77
    assert kwargs["tau_0"] == pytest.approx(0.2)
    assert kwargs["is_not_compiled"] is False
    assert "stepsize_controller" not in kwargs
