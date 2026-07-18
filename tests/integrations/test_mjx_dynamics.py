"""Unit tests for the `MjxDynamics` adapter and `Problem` dispatch.

The lower-level callable contract (``mjx_dynamics``, ``_free_joint_qpos_dynamics``)
is covered by ``test_mjx.py``; this file focuses on the adapter surface, the
``Problem`` dispatch, and the BYOF-merge helper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import openscvx as ox
from openscvx.integrations.base import DynamicsAdapter, _merge_byof

_CARTPOLE_XML = """
<mujoco model="test_cartpole">
  <option gravity="0 0 -9.81" timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.2 0.1 0.1" mass="1.0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.03" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" name="cart_force" gear="10"
           ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

_FREE_BODY_XML = """
<mujoco model="free_body">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def cartpole_mjx_model():
    """nq == nv == 2, nu == 1, contacts disabled."""
    import mujoco
    import mujoco.mjx as mjx

    mj_model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return mjx.put_model(mj_model)


@pytest.fixture(scope="module")
def free_body_mjx_model():
    """nq == 7, nv == 6 (single free joint), nu == 0, contacts disabled."""
    import mujoco
    import mujoco.mjx as mjx

    mj_model = mujoco.MjModel.from_xml_string(_FREE_BODY_XML)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return mjx.put_model(mj_model)


# ===========================================================================
# DynamicsAdapter / _merge_byof
# ===========================================================================


def test_merge_byof_passes_extra_through_when_user_byof_is_none():
    extra = {"dynamics": {"qvel": (lambda x, u, n, p: x)}}
    merged = _merge_byof(None, extra)
    assert set(merged.keys()) == {"dynamics"}
    assert set(merged["dynamics"].keys()) == {"qvel"}


def test_merge_byof_combines_user_dynamics_with_adapter_dynamics():
    user = {"dynamics": {"pos": (lambda x, u, n, p: x)}}
    extra = {"dynamics": {"qvel": (lambda x, u, n, p: x)}}
    merged = _merge_byof(user, extra)
    assert set(merged["dynamics"].keys()) == {"pos", "qvel"}


def test_merge_byof_preserves_user_byof_top_level_keys():
    """Non-``dynamics`` byof keys come through verbatim."""
    cross_fn = lambda X, U, p: X[:, 0] - 1.0  # noqa: E731
    user = {"cross_nodal_constraints": [cross_fn]}
    extra = {"dynamics": {"qvel": (lambda x, u, n, p: x)}}
    merged = _merge_byof(user, extra)
    assert merged["cross_nodal_constraints"] is user["cross_nodal_constraints"]
    assert "qvel" in merged["dynamics"]


def test_merge_byof_raises_on_dynamics_key_collision():
    user = {"dynamics": {"qvel": (lambda x, u, n, p: x)}}
    extra = {"dynamics": {"qvel": (lambda x, u, n, p: x)}}
    with pytest.raises(ValueError, match="collide"):
        _merge_byof(user, extra)


# ===========================================================================
# MjxDynamics — construction and .expand() shape
# ===========================================================================


@pytest.mark.mjx
def test_mjx_dynamics_subclasses_dynamics_adapter(cartpole_mjx_model):
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    assert isinstance(dyn, DynamicsAdapter)


@pytest.mark.mjx
def test_mjx_dynamics_states_and_controls_match_model_dims(cartpole_mjx_model):
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    assert len(dyn.states) == 2
    qpos, qvel = dyn.states
    assert qpos.name == "qpos" and qpos.shape == (cartpole_mjx_model.nq,)
    assert qvel.name == "qvel" and qvel.shape == (cartpole_mjx_model.nv,)

    assert len(dyn.controls) == 1
    (ctrl,) = dyn.controls
    assert ctrl.name == "ctrl" and ctrl.shape == (cartpole_mjx_model.nu,)


@pytest.mark.mjx
def test_mjx_dynamics_expand_returns_qpos_kinematics_symbolic_when_nq_eq_nv(
    cartpole_mjx_model,
):
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    dynamics_dict, byof_dict = dyn.expand()

    assert set(dynamics_dict.keys()) == {"qpos"}
    assert dynamics_dict["qpos"] is dyn.states[1]  # qvel state

    assert set(byof_dict.keys()) == {"dynamics"}
    assert set(byof_dict["dynamics"].keys()) == {"qvel"}
    assert callable(byof_dict["dynamics"]["qvel"])


@pytest.mark.mjx
def test_mjx_dynamics_expand_routes_both_through_byof_when_nq_gt_nv(
    free_body_mjx_model,
):
    assert free_body_mjx_model.nq == 7 and free_body_mjx_model.nv == 6
    dyn = ox.MjxDynamics(free_body_mjx_model)
    dynamics_dict, byof_dict = dyn.expand()

    assert dynamics_dict == {}
    assert set(byof_dict["dynamics"].keys()) == {"qpos", "qvel"}
    assert callable(byof_dict["dynamics"]["qpos"])
    assert callable(byof_dict["dynamics"]["qvel"])


# ===========================================================================
# Problem dispatch — adapter goes in the dynamics= slot
# ===========================================================================


def _minimal_time(total_time: float = 1.0) -> ox.Time:
    return ox.Time(
        initial=0.0,
        final=ox.Minimize(total_time),
        min=0.0,
        max=2.0 * total_time,
        time_dilation_min=0.05 * total_time,
        time_dilation_max=2.0 * total_time,
    )


def _set_simple_bounds(state, lo, hi, init, final):
    state.min = np.asarray(lo)
    state.max = np.asarray(hi)
    state.initial = np.asarray(init)
    state.final = np.asarray(final)


@pytest.mark.mjx
def test_problem_accepts_mjx_dynamics_adapter_for_cartpole(cartpole_mjx_model):
    """Construct a Problem with an MjxDynamics in the dynamics= slot (nq == nv)."""
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    qpos, qvel = dyn.states
    (ctrl,) = dyn.controls

    _set_simple_bounds(qpos, [-3.0, -2 * np.pi], [3.0, 2 * np.pi], [0.0, np.pi], [0.0, 0.0])
    _set_simple_bounds(qvel, [-10.0, -15.0], [10.0, 15.0], [0.0, 0.0], [0.0, 0.0])
    ctrl.min = np.array([-1.0])
    ctrl.max = np.array([1.0])

    n = 10
    qpos.guess = np.column_stack([np.zeros(n), np.linspace(np.pi, 0.0, n)])
    qvel.guess = np.zeros((n, 2))
    ctrl.guess = np.zeros((n, 1))

    constraints = []
    for s in (qpos, qvel):
        constraints.extend([ox.ctcs(s <= s.max), ox.ctcs(s.min <= s)])
    constraints.extend([ox.ctcs(ctrl <= ctrl.max), ox.ctcs(ctrl.min <= ctrl)])

    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        time=_minimal_time(),
        constraints=constraints,
        N=n,
        float_dtype="float64",
    )

    # Symbolic preprocessing succeeded and merged byof has the qvel callable.
    assert problem._byof is not None
    assert "qvel" in problem._byof.dynamics
    # qpos kinematics came from the symbolic side (nq == nv branch).
    state_names = {s.name for s in problem.symbolic.states}
    assert {"qpos", "qvel"}.issubset(state_names)


@pytest.mark.mjx
def test_problem_accepts_mjx_dynamics_adapter_for_free_joint(free_body_mjx_model):
    """nq > nv free-joint model: both qpos and qvel must be in byof."""
    dyn = ox.MjxDynamics(free_body_mjx_model)
    qpos, qvel = dyn.states
    (ctrl,) = dyn.controls  # nu == 0 → zero-length Control

    qpos.min = np.full(7, -1e3)
    qpos.max = np.full(7, 1e3)
    qpos.initial = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    qpos.final = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    qvel.min = np.full(6, -1e3)
    qvel.max = np.full(6, 1e3)
    qvel.initial = np.zeros(6)
    qvel.final = np.zeros(6)

    # nu = 0 → empty arrays still satisfy the bounds-required validator.
    ctrl.min = np.zeros(0)
    ctrl.max = np.zeros(0)

    n = 8
    qpos.guess = np.tile(qpos.initial, (n, 1))
    qvel.guess = np.zeros((n, 6))
    ctrl.guess = np.zeros((n, 0))

    constraints = [ox.ctcs(qvel <= qvel.max), ox.ctcs(qvel.min <= qvel)]

    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        time=_minimal_time(),
        constraints=constraints,
        N=n,
        float_dtype="float64",
    )

    assert problem._byof is not None
    assert {"qpos", "qvel"}.issubset(problem._byof.dynamics)


@pytest.mark.mjx
def test_problem_accepts_mjx_dynamics_with_extra_prop_state(cartpole_mjx_model):
    """User can layer an extra propagation-only state on top of the adapter."""
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    qpos, qvel = dyn.states
    (ctrl,) = dyn.controls

    _set_simple_bounds(qpos, [-3.0, -2 * np.pi], [3.0, 2 * np.pi], [0.0, np.pi], [0.0, 0.0])
    _set_simple_bounds(qvel, [-10.0, -15.0], [10.0, 15.0], [0.0, 0.0], [0.0, 0.0])
    ctrl.min = np.array([-1.0])
    ctrl.max = np.array([1.0])

    n = 8
    qpos.guess = np.column_stack([np.zeros(n), np.linspace(np.pi, 0.0, n)])
    qvel.guess = np.zeros((n, 2))
    ctrl.guess = np.zeros((n, 1))

    # An extra propagation-only state (e.g. cumulative ∫u² for diagnostics).
    energy = ox.State("energy", shape=(1,))
    energy.initial = np.array([0.0])

    constraints = []
    for s in (qpos, qvel):
        constraints.extend([ox.ctcs(s <= s.max), ox.ctcs(s.min <= s)])
    constraints.extend([ox.ctcs(ctrl <= ctrl.max), ox.ctcs(ctrl.min <= ctrl)])

    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        time=_minimal_time(),
        constraints=constraints,
        N=n,
        states_prop=[energy],
        dynamics_prop={"energy": ctrl[0] * ctrl[0]},
        float_dtype="float64",
    )

    prop_state_names = {s.name for s in problem.symbolic.states_prop}
    assert "energy" in prop_state_names


# ===========================================================================
# Sanity — MjxDynamics callables produce correct output shapes after dispatch
# ===========================================================================


@pytest.mark.mjx
def test_mjx_dynamics_callables_run_after_slice_assignment(cartpole_mjx_model):
    """After Problem construction wires slices, the byof callables run."""
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    qpos, qvel = dyn.states
    (ctrl,) = dyn.controls

    _set_simple_bounds(qpos, [-3.0, -2 * np.pi], [3.0, 2 * np.pi], [0.0, np.pi], [0.0, 0.0])
    _set_simple_bounds(qvel, [-10.0, -15.0], [10.0, 15.0], [0.0, 0.0], [0.0, 0.0])
    ctrl.min = np.array([-1.0])
    ctrl.max = np.array([1.0])

    n = 6
    qpos.guess = np.zeros((n, 2))
    qvel.guess = np.zeros((n, 2))
    ctrl.guess = np.zeros((n, 1))

    constraints = []
    for s in (qpos, qvel):
        constraints.extend([ox.ctcs(s <= s.max), ox.ctcs(s.min <= s)])
    constraints.extend([ox.ctcs(ctrl <= ctrl.max), ox.ctcs(ctrl.min <= ctrl)])

    problem = ox.Problem(
        dynamics=dyn,
        states=dyn.states,
        controls=dyn.controls,
        time=_minimal_time(),
        constraints=constraints,
        N=n,
        float_dtype="float64",
    )

    # Pull the byof qvel callable from the validated ByofSpec; it should
    # accept a unified x vector and return (nv,).
    qvel_fn = problem._byof.dynamics["qvel"]
    n_x = sum(s.shape[0] for s in problem.symbolic.states)
    n_u = sum(c.shape[0] for c in problem.symbolic.controls)
    x = jnp.zeros(n_x)
    u = jnp.zeros(n_u)
    out = qvel_fn(x, u, 0, {})
    assert out.shape == (int(cartpole_mjx_model.nv),)
    np.testing.assert_array_equal(np.array(out).shape, (cartpole_mjx_model.nv,))


# ===========================================================================
# Joint-structure validation
# ===========================================================================


_BALL_JOINT_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint type="ball"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_FREE_AFTER_HINGE_XML = """
<mujoco>
  <worldbody>
    <body name="hinged">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body name="floater">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _put_model(xml: str):
    import mujoco
    import mujoco.mjx as mjx

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return mjx.put_model(mj_model)


@pytest.mark.mjx
def test_mjx_dynamics_rejects_ball_joint():
    mjx_model = _put_model(_BALL_JOINT_XML)
    with pytest.raises(NotImplementedError, match="ball joints"):
        ox.MjxDynamics(mjx_model)


@pytest.mark.mjx
def test_mjx_dynamics_rejects_free_joint_after_hinge():
    mjx_model = _put_model(_FREE_AFTER_HINGE_XML)
    with pytest.raises(NotImplementedError, match="free joints to come before"):
        ox.MjxDynamics(mjx_model)


@pytest.mark.mjx
def test_mjx_dynamics_accepts_slide_plus_hinge(cartpole_mjx_model):
    # No exception — the cartpole has slide + hinge joints in that order.
    ox.MjxDynamics(cartpole_mjx_model)


@pytest.mark.mjx
def test_mjx_dynamics_accepts_pure_free_joint(free_body_mjx_model):
    # No exception — a single free joint is supported.
    ox.MjxDynamics(free_body_mjx_model)


# ===========================================================================
# Auto-populated bounds (jnt_range / actuator_ctrlrange)
# ===========================================================================


_LIMITED_SLIDER_XML = """
<mujoco model="limited_slider">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.5 4.0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="5" ctrlrange="-1.5 2.0" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

_UNLIMITED_ACTUATOR_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="h" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="h" gear="1"/>
  </actuator>
</mujoco>
"""


@pytest.mark.mjx
def test_mjx_dynamics_populates_qpos_bounds_from_jnt_range():
    """Slide joint with `limited="true" range="-2.5 4.0"` flows into qpos.min/max."""
    mjx_model = _put_model(_LIMITED_SLIDER_XML)
    dyn = ox.MjxDynamics(mjx_model)
    qpos, qvel = dyn.states

    # slider is joint 0 → qpos[0]; hinge has no range → ±inf
    assert qpos.min[0] == pytest.approx(-2.5)
    assert qpos.max[0] == pytest.approx(4.0)
    assert qpos.min[1] == -np.inf
    assert qpos.max[1] == np.inf


@pytest.mark.mjx
def test_mjx_dynamics_populates_ctrl_bounds_from_actuator_ctrlrange():
    """Actuator with `ctrlrange="-1.5 2.0" ctrllimited="true"` flows into ctrl.min/max."""
    mjx_model = _put_model(_LIMITED_SLIDER_XML)
    dyn = ox.MjxDynamics(mjx_model)
    (ctrl,) = dyn.controls

    assert ctrl.min[0] == pytest.approx(-1.5)
    assert ctrl.max[0] == pytest.approx(2.0)


@pytest.mark.mjx
def test_mjx_dynamics_unlimited_actuator_defaults_to_inf():
    """Actuator without ctrllimited gets ±inf ctrl bounds."""
    mjx_model = _put_model(_UNLIMITED_ACTUATOR_XML)
    dyn = ox.MjxDynamics(mjx_model)
    (ctrl,) = dyn.controls
    assert ctrl.min[0] == -np.inf
    assert ctrl.max[0] == np.inf


@pytest.mark.mjx
def test_mjx_dynamics_qvel_defaults_to_inf(cartpole_mjx_model):
    """qvel always defaults to ±inf since MuJoCo has no per-joint velocity limits."""
    dyn = ox.MjxDynamics(cartpole_mjx_model)
    _, qvel = dyn.states
    assert np.all(np.isneginf(qvel.min))
    assert np.all(np.isposinf(qvel.max))


@pytest.mark.mjx
def test_mjx_dynamics_free_joint_qpos_bounds_are_inf(free_body_mjx_model):
    """Free joints have jnt_limited=False, so all 7 qpos slots stay at ±inf."""
    dyn = ox.MjxDynamics(free_body_mjx_model)
    qpos, _ = dyn.states
    assert np.all(np.isneginf(qpos.min))
    assert np.all(np.isposinf(qpos.max))


@pytest.mark.mjx
def test_mjx_dynamics_user_can_override_auto_bounds():
    """Users can replace the auto-populated bounds after construction."""
    mjx_model = _put_model(_LIMITED_SLIDER_XML)
    dyn = ox.MjxDynamics(mjx_model)
    qpos, _ = dyn.states

    new_min = np.array([-10.0, -5.0])
    qpos.min = new_min
    assert qpos.min[0] == -10.0
    assert qpos.min[1] == -5.0
