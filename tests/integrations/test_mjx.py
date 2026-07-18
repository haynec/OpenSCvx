"""Unit tests for the OpenSCvx <-> MJX integration interface.

Coverage
--------
* ``_resolve_slice`` — slice extraction from State/slice/invalid types.
* ``mjx_dynamics``   — callable contract, output shape, return_component
                       validation, lazy slice resolution, extra_postprocess hook.
* ``_free_joint_qpos_dynamics`` — callable/output-shape contract and argument
                       handling (without re-testing quaternion math internals).

The high-level `MjxDynamics` adapter is covered in ``test_mjx_dynamics.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / minimal MuJoCo XML
# ---------------------------------------------------------------------------

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

# ===========================================================================
# _resolve_slice
# ===========================================================================


class TestResolveSlice:
    """Tests for the private ``_resolve_slice`` helper."""

    def _import(self):
        from openscvx.integrations.mjx import _resolve_slice

        return _resolve_slice

    def test_plain_slice_passthrough(self):
        _resolve_slice = self._import()
        s = slice(0, 4)
        assert _resolve_slice(s, "qpos") is s

    def test_state_with_slice_returns_slice(self):
        _resolve_slice = self._import()
        import openscvx as ox

        state = ox.State("q", shape=(3,))
        state._slice = slice(0, 3)  # inject manually
        result = _resolve_slice(state, "q")
        assert result == slice(0, 3)

    def test_state_without_slice_raises(self):
        _resolve_slice = self._import()
        import openscvx as ox

        state = ox.State("q", shape=(3,))
        # .slice is None until Problem assigns it
        with pytest.raises(ValueError, match="no .slice yet"):
            _resolve_slice(state, "q")

    def test_invalid_type_raises_type_error(self):
        _resolve_slice = self._import()
        with pytest.raises(TypeError, match="must be a State/Control or slice"):
            _resolve_slice(42, "qpos")

    def test_invalid_type_list_raises_type_error(self):
        _resolve_slice = self._import()
        with pytest.raises(TypeError):
            _resolve_slice([0, 1, 2], "qpos")


# ===========================================================================
# mjx_dynamics
# ===========================================================================


@pytest.fixture(scope="module")
def cartpole_mjx_model():
    """Build and return a minimal cartpole MJX model (contacts disabled)."""
    import mujoco
    import mujoco.mjx as mjx

    mj_model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return mjx.put_model(mj_model)


@pytest.mark.mjx
class TestMjxDynamics:
    """Tests for ``mjx_dynamics``."""

    def test_returns_callable(self, cartpole_mjx_model):
        from openscvx.integrations.mjx import mjx_dynamics

        f = mjx_dynamics(
            cartpole_mjx_model,
            qpos=slice(0, 2),
            qvel=slice(2, 4),
            ctrl=slice(0, 1),
        )
        assert callable(f)

    def test_invalid_return_component_raises(self, cartpole_mjx_model):
        from openscvx.integrations.mjx import mjx_dynamics

        with pytest.raises(ValueError, match="return_component"):
            mjx_dynamics(
                cartpole_mjx_model,
                qpos=slice(0, 2),
                qvel=slice(2, 4),
                ctrl=slice(0, 1),
                return_component="acceleration",
            )

    def test_qacc_output_shape(self, cartpole_mjx_model):
        """Default (return_component='qacc') should return shape (nv,)."""
        from openscvx.integrations.mjx import mjx_dynamics

        nq, nv, nu = 2, 2, 1
        f = mjx_dynamics(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
        )
        x = jnp.zeros(nq + nv)
        u = jnp.zeros(nu)
        out = f(x, u, 0, {})
        assert out.shape == (nv,)

    def test_qvel_return_component_shape(self, cartpole_mjx_model):
        """return_component='qvel' should return shape (nv,)."""
        from openscvx.integrations.mjx import mjx_dynamics

        nq, nv, nu = 2, 2, 1
        f = mjx_dynamics(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
            return_component="qvel",
        )
        x = jnp.zeros(nq + nv)
        u = jnp.zeros(nu)
        out = f(x, u, 0, {})
        assert out.shape == (nv,)

    def test_extra_postprocess_called(self, cartpole_mjx_model):
        """extra_postprocess should be applied to the MJX data object."""
        from openscvx.integrations.mjx import mjx_dynamics

        nq, nv, nu = 2, 2, 1
        postprocess_called = []

        def record(data):
            postprocess_called.append(True)
            return data

        f = mjx_dynamics(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
            extra_postprocess=record,
        )
        x = jnp.zeros(nq + nv)
        u = jnp.zeros(nu)
        f(x, u, 0, {})
        assert postprocess_called, "extra_postprocess was never called"

    def test_lazy_slice_resolution_raises_before_slice_set(self):
        """mjx_dynamics with an unresolved State should raise ValueError on first call."""
        import mujoco
        import mujoco.mjx as mjx

        import openscvx as ox
        from openscvx.integrations.mjx import mjx_dynamics

        mj_model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mjx_m = mjx.put_model(mj_model)

        qpos = ox.State("qpos", shape=(2,))  # .slice is None
        qvel = ox.State("qvel", shape=(2,))
        ctrl = ox.Control("ctrl", shape=(1,))

        f = mjx_dynamics(mjx_m, qpos=qpos, qvel=qvel, ctrl=ctrl)
        with pytest.raises(ValueError, match="no .slice yet"):
            f(jnp.zeros(4), jnp.zeros(1), 0, {})

    def test_node_and_params_are_ignored(self, cartpole_mjx_model):
        """node and params arguments must be accepted but have no effect on output."""
        from openscvx.integrations.mjx import mjx_dynamics

        nq, nv, nu = 2, 2, 1
        f = mjx_dynamics(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
        )
        x = jnp.zeros(nq + nv)
        u = jnp.zeros(nu)
        out1 = f(x, u, 0, {})
        out2 = f(x, u, 99, {"dummy": 1})
        np.testing.assert_allclose(np.array(out1), np.array(out2))


# ===========================================================================
# _free_joint_qpos_dynamics
# ===========================================================================


class TestFreeJointQposDynamics:
    """Tests for ``_free_joint_qpos_dynamics`` (pure JAX, no MuJoCo required)."""

    def _make(self, n_free=1, extra_joints=0):
        from openscvx.integrations.mjx import _free_joint_qpos_dynamics

        nq = 7 * n_free + extra_joints
        nv = 6 * n_free + extra_joints
        f = _free_joint_qpos_dynamics(
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            n_free_joints=n_free,
        )
        return f, nq, nv

    def test_returns_callable(self):
        from openscvx.integrations.mjx import _free_joint_qpos_dynamics

        f = _free_joint_qpos_dynamics(qpos=slice(0, 7), qvel=slice(7, 13))
        assert callable(f)

    def test_output_shape_single_free_joint(self):
        f, nq, nv = self._make(n_free=1)
        x = jnp.zeros(nq + nv)
        out = f(x, jnp.zeros(1), 0, {})
        assert out.shape == (nq,), f"expected ({nq},), got {out.shape}"

    def test_output_shape_two_free_joints(self):
        f, nq, nv = self._make(n_free=2)
        x = jnp.zeros(nq + nv)
        out = f(x, jnp.zeros(1), 0, {})
        assert out.shape == (nq,)

    def test_output_shape_free_plus_revolute(self):
        """1 free joint + 2 extra revolute joints → nq=9, nv=8."""
        f, nq, nv = self._make(n_free=1, extra_joints=2)
        x = jnp.zeros(nq + nv)
        out = f(x, jnp.zeros(1), 0, {})
        assert out.shape == (nq,)

    def test_u_node_params_ignored(self):
        """u, node, and params must not affect the output."""
        f, nq, nv = self._make(n_free=1)
        x = jnp.concatenate(
            [
                jnp.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
                jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ]
        )
        out1 = f(x, jnp.zeros(5), 0, {})
        out2 = f(x, jnp.ones(5) * 999, 42, {"key": "val"})
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-9)
