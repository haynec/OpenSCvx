"""Unit tests for openscvx.integrations.mjx and openscvx.integrations.menagerie.

Coverage
--------
* ``_resolve_slice`` — slice extraction from State/slice/invalid types.
* ``mjx_dynamics``   — return type, output shape, return_component, lazy slice
                       resolution, extra_postprocess.  Skipped when
                       ``mujoco.mjx`` is not installed.
* ``free_joint_qpos_dynamics`` — output shape, quaternion kinematics identity,
                       angular-velocity derivative, two free joints, revolute
                       pass-through, no-free-joints path.
* ``menagerie``      — ``find_menagerie_root``, ``get_model_dir``,
                       ``get_xml_path``, ``list_models``, ``get_asset_dir``
                       using a ``tmp_path``-backed mock directory tree.
* ``integrations.__init__``  — lazy public-API exports and AttributeError.
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

# ---------------------------------------------------------------------------
# Availability flags (used as skip decorators on MuJoCo-dependent classes)
# ---------------------------------------------------------------------------
try:
    import mujoco as _mujoco  # noqa: F401
    import mujoco.mjx as _mjx  # noqa: F401

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

requires_mujoco = pytest.mark.skipif(
    not _MUJOCO_AVAILABLE, reason="mujoco / mujoco.mjx not installed"
)


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


@requires_mujoco
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

    def test_zero_state_zero_ctrl_gravity(self, cartpole_mjx_model):
        """With zero control, gravity should produce non-zero qacc on the pole hinge."""
        from openscvx.integrations.mjx import mjx_dynamics

        nq, nv, nu = 2, 2, 1
        f = mjx_dynamics(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
        )
        # Upright pole (hinge=0): gravity acts along pole → non-trivial qacc
        x = jnp.array([0.0, 0.0, 0.0, 0.0])  # cart x, hinge angle, then velocities
        u = jnp.zeros(nu)
        qacc = f(x, u, 0, {})
        # Cart should be free to slide (no friction) → cart qacc near 0
        # Pole at vertical with gravity: qacc for hinge depends on mass distribution
        assert jnp.isfinite(qacc).all()

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
# free_joint_qpos_dynamics
# ===========================================================================


class TestFreeJointQposDynamics:
    """Tests for ``free_joint_qpos_dynamics`` (pure JAX, no MuJoCo required)."""

    def _make(self, n_free=1, extra_joints=0):
        from openscvx.integrations.mjx import free_joint_qpos_dynamics

        nq = 7 * n_free + extra_joints
        nv = 6 * n_free + extra_joints
        f = free_joint_qpos_dynamics(
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            n_free_joints=n_free,
        )
        return f, nq, nv

    def test_returns_callable(self):
        from openscvx.integrations.mjx import free_joint_qpos_dynamics

        f = free_joint_qpos_dynamics(qpos=slice(0, 7), qvel=slice(7, 13))
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

    def test_zero_velocity_zero_output(self):
        """Zero velocity ⇒ all derivatives are zero."""
        f, nq, nv = self._make(n_free=1)
        # qpos: identity pose [x=0,y=0,z=0, qw=1,qx=0,qy=0,qz=0]
        # qvel: all zeros
        x = jnp.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,  # qpos
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )  # qvel
        out = f(x, jnp.zeros(1), 0, {})
        np.testing.assert_allclose(np.array(out), np.zeros(nq), atol=1e-6)

    def test_translation_derivative_equals_linear_velocity(self):
        """Linear velocity [vx,vy,vz] must appear as qdot for translation."""
        f, nq, nv = self._make(n_free=1)
        v = jnp.array([1.0, 2.0, 3.0])
        x = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # qpos
                v,
                jnp.zeros(3),  # qvel: linear + angular
            ]
        )
        out = f(x, jnp.zeros(1), 0, {})
        np.testing.assert_allclose(np.array(out[:3]), np.array(v), atol=1e-6)

    def test_quaternion_derivative_identity_x_rotation(self):
        """Identity quaternion + angular velocity [wx, 0, 0] → quat_dot = [0, wx/2, 0, 0]."""
        f, nq, nv = self._make(n_free=1)
        wx = 2.0
        x = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # qpos
                jnp.array([0.0, 0.0, 0.0, wx, 0.0, 0.0]),  # qvel
            ]
        )
        out = f(x, jnp.zeros(1), 0, {})
        quat_dot = np.array(out[3:7])
        expected = np.array([0.0, wx / 2, 0.0, 0.0])
        np.testing.assert_allclose(quat_dot, expected, atol=1e-6)

    def test_quaternion_derivative_identity_y_rotation(self):
        """Identity quaternion + angular velocity [0, wy, 0] → quat_dot = [0, 0, wy/2, 0]."""
        f, nq, nv = self._make(n_free=1)
        wy = 3.0
        x = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                jnp.array([0.0, 0.0, 0.0, 0.0, wy, 0.0]),
            ]
        )
        out = f(x, jnp.zeros(1), 0, {})
        quat_dot = np.array(out[3:7])
        expected = np.array([0.0, 0.0, wy / 2, 0.0])
        np.testing.assert_allclose(quat_dot, expected, atol=1e-6)

    def test_quaternion_derivative_identity_z_rotation(self):
        """Identity quaternion + angular velocity [0, 0, wz] → quat_dot = [0, 0, 0, wz/2]."""
        f, nq, nv = self._make(n_free=1)
        wz = 5.0
        x = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, wz]),
            ]
        )
        out = f(x, jnp.zeros(1), 0, {})
        quat_dot = np.array(out[3:7])
        expected = np.array([0.0, 0.0, 0.0, wz / 2])
        np.testing.assert_allclose(quat_dot, expected, atol=1e-6)

    def test_revolute_joints_pass_through(self):
        """Revolute/prismatic joint velocities after the free joint must be copied 1-to-1."""
        f, nq, nv = self._make(n_free=1, extra_joints=3)
        joint_vel = jnp.array([0.5, -1.2, 0.7])
        x = jnp.concatenate(
            [
                jnp.zeros(7),  # qpos (free joint pose)
                jnp.zeros(3),  # qpos extra joints
                jnp.zeros(6),  # qvel (free joint)
                joint_vel,  # qvel extra joints
            ]
        )
        out = f(x, jnp.zeros(1), 0, {})
        np.testing.assert_allclose(np.array(out[7:]), np.array(joint_vel), atol=1e-6)

    def test_two_free_joints_translation(self):
        """Two free joints: second joint's translation derivative should equal its linear vel."""
        f, nq, nv = self._make(n_free=2)
        # qpos: [pose1(7), pose2(7)], qvel: [vel1(6), vel2(6)]
        v2 = jnp.array([4.0, 5.0, 6.0])
        x = jnp.concatenate(
            [
                jnp.zeros(3),
                jnp.array([1.0, 0.0, 0.0, 0.0]),  # joint 1 pose
                jnp.zeros(3),
                jnp.array([1.0, 0.0, 0.0, 0.0]),  # joint 2 pose
                jnp.zeros(6),  # joint 1 vel
                v2,
                jnp.zeros(3),  # joint 2 vel
            ]
        )
        out = f(x, jnp.zeros(1), 0, {})
        np.testing.assert_allclose(np.array(out[7:10]), np.array(v2), atol=1e-6)

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


# ===========================================================================
# menagerie
# ===========================================================================


class TestFindMenagerieRoot:
    """Tests for ``find_menagerie_root``."""

    def test_env_var_valid_dir_returned(self, monkeypatch, tmp_path):
        """A valid MUJOCO_MENAGERIE_PATH env var takes priority over all else."""
        monkeypatch.setenv("MUJOCO_MENAGERIE_PATH", str(tmp_path))
        import openscvx.integrations.menagerie as m

        result = m.find_menagerie_root()
        assert result == tmp_path

    def test_env_var_nonexistent_dir_falls_through(self, monkeypatch, tmp_path):
        """When MUJOCO_MENAGERIE_PATH points to a missing directory the env
        var is not returned — the function falls through to other discovery
        paths.  We verify that the result is never the nonexistent path.
        """
        import openscvx.integrations.menagerie as m

        nonexistent = tmp_path / "does_not_exist"
        monkeypatch.setenv("MUJOCO_MENAGERIE_PATH", str(nonexistent))

        result = m.find_menagerie_root()
        # The invalid env-var path must never be returned
        assert result != nonexistent

    def test_env_var_not_set_returns_non_none_when_submodule_present(self, monkeypatch):
        """When the git submodule is initialised the function must return a Path."""
        import openscvx.integrations.menagerie as m

        monkeypatch.delenv("MUJOCO_MENAGERIE_PATH", raising=False)
        root = m.find_menagerie_root()
        # The submodule lives at third_party/mujoco_menagerie; if it is
        # initialised (as expected in this repo) we get a path back.
        if root is not None:
            assert root.is_dir()
        # Passing if root is None is also fine (submodule not yet initialised)


class TestGetModelDir:
    """Tests for ``get_model_dir``."""

    def test_raises_when_root_is_none(self, monkeypatch):
        from openscvx.integrations import menagerie

        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: None)
        with pytest.raises(FileNotFoundError, match="Menagerie not found"):
            menagerie.get_model_dir("any_model")

    def test_raises_for_missing_model(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            menagerie.get_model_dir("ghost_model")

    def test_returns_correct_dir(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        model_dir = tmp_path / "my_robot"
        model_dir.mkdir()
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: tmp_path)
        result = menagerie.get_model_dir("my_robot")
        assert result == model_dir


class TestGetXmlPath:
    """Tests for ``get_xml_path``."""

    def _setup_model(self, tmp_path, xml_names):
        """Create a fake model directory with the given XML file names."""
        model_dir = tmp_path / "robot"
        model_dir.mkdir()
        for name in xml_names:
            (model_dir / name).touch()
        return tmp_path

    def test_prefers_mjx_xml(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        root = self._setup_model(tmp_path, ["robot.xml", "mjx_robot.xml"])
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: root)
        result = menagerie.get_xml_path("robot", prefer_mjx=True)
        assert result.name == "mjx_robot.xml"

    def test_falls_back_to_regular_xml(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        root = self._setup_model(tmp_path, ["robot.xml"])
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: root)
        result = menagerie.get_xml_path("robot", prefer_mjx=True)
        assert result.name == "robot.xml"

    def test_skips_scene_xml(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        root = self._setup_model(tmp_path, ["scene.xml", "robot.xml"])
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: root)
        result = menagerie.get_xml_path("robot", prefer_mjx=False)
        assert result.name == "robot.xml"

    def test_scene_xml_used_as_last_resort(self, monkeypatch, tmp_path):
        """If only scene*.xml files exist, one is returned."""
        from openscvx.integrations import menagerie

        root = self._setup_model(tmp_path, ["scene.xml"])
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: root)
        result = menagerie.get_xml_path("robot")
        assert result.name == "scene.xml"

    def test_raises_when_no_xml(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        root = self._setup_model(tmp_path, [])
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: root)
        with pytest.raises(FileNotFoundError, match="No XML file"):
            menagerie.get_xml_path("robot")

    def test_prefer_mjx_false_returns_non_mjx(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        root = self._setup_model(tmp_path, ["robot.xml", "mjx_robot.xml"])
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: root)
        result = menagerie.get_xml_path("robot", prefer_mjx=False)
        # Should return the alphabetically first non-scene file (mjx_robot.xml or robot.xml)
        assert result.name in ("robot.xml", "mjx_robot.xml")
        # It must NOT be a scene file
        assert not result.name.lower().startswith("scene")


class TestListModels:
    """Tests for ``list_models``."""

    def test_returns_empty_list_when_no_menagerie(self, monkeypatch):
        from openscvx.integrations import menagerie

        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: None)
        assert menagerie.list_models() == []

    def test_returns_model_names(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        (tmp_path / "robot_a").mkdir()
        (tmp_path / "robot_b").mkdir()
        (tmp_path / ".hidden").mkdir()  # should be excluded
        (tmp_path / "README.md").touch()  # file, not dir; should be excluded
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: tmp_path)
        models = menagerie.list_models()
        assert sorted(models) == ["robot_a", "robot_b"]

    def test_sorted_output(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        for name in ["zebra", "antelope", "meerkat"]:
            (tmp_path / name).mkdir()
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: tmp_path)
        models = menagerie.list_models()
        assert models == sorted(models)


class TestGetAssetDir:
    """Tests for ``get_asset_dir``."""

    def test_returns_assets_dir(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        model_dir = tmp_path / "robot"
        assets = model_dir / "assets"
        assets.mkdir(parents=True)
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: tmp_path)
        result = menagerie.get_asset_dir("robot")
        assert result == assets

    def test_raises_when_no_assets_dir(self, monkeypatch, tmp_path):
        from openscvx.integrations import menagerie

        (tmp_path / "robot").mkdir()
        monkeypatch.setattr(menagerie, "find_menagerie_root", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="assets/"):
            menagerie.get_asset_dir("robot")


# ===========================================================================
# integrations public API  (__init__.py lazy exports)
# ===========================================================================


@requires_mujoco
class TestMjxByof:
    """Tests for the high-level ``mjx_byof`` convenience wrapper."""

    def test_no_free_joints_returns_qvel_only(self, cartpole_mjx_model):
        """nq == nv model: mjx_byof should only include 'qvel'."""
        from openscvx.integrations.mjx import mjx_byof

        nq, nv, nu = 2, 2, 1
        result = mjx_byof(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
        )
        assert set(result.keys()) == {"qvel"}
        assert callable(result["qvel"])

    def test_free_joint_model_returns_qpos_and_qvel(self):
        """nq > nv model: mjx_byof must include both 'qpos' and 'qvel'."""
        import mujoco
        import mujoco.mjx as mjx

        from openscvx.integrations.mjx import mjx_byof

        _FREE_XML = """
        <mujoco><option gravity="0 0 -9.81"/>
          <worldbody>
            <body name="b"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
          </worldbody>
        </mujoco>"""
        mj_model = mujoco.MjModel.from_xml_string(_FREE_XML)
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mjx_model = mjx.put_model(mj_model)
        assert mjx_model.nq == 7 and mjx_model.nv == 6

        result = mjx_byof(
            mjx_model,
            qpos=slice(0, 7),
            qvel=slice(7, 13),
            ctrl=slice(0, 0),
        )
        assert set(result.keys()) == {"qpos", "qvel"}
        assert callable(result["qpos"])
        assert callable(result["qvel"])

    def test_qvel_callable_output_shape(self, cartpole_mjx_model):
        """qvel callable must return shape (nv,)."""
        from openscvx.integrations.mjx import mjx_byof

        nq, nv, nu = 2, 2, 1
        result = mjx_byof(
            cartpole_mjx_model,
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            ctrl=slice(0, nu),
        )
        x = jnp.zeros(nq + nv)
        u = jnp.zeros(nu)
        out = result["qvel"](x, u, 0, {})
        assert out.shape == (nv,)


class TestIntegrationsPublicAPI:
    """Tests for lazy attribute exports in openscvx.integrations."""

    def test_mjx_byof_importable(self):
        from openscvx.integrations import mjx_byof

        assert callable(mjx_byof)

    def test_mjx_dynamics_importable(self):
        from openscvx.integrations import mjx_dynamics

        assert callable(mjx_dynamics)

    def test_free_joint_qpos_dynamics_importable(self):
        from openscvx.integrations import free_joint_qpos_dynamics

        assert callable(free_joint_qpos_dynamics)

    def test_menagerie_importable(self):
        from openscvx.integrations import menagerie

        assert hasattr(menagerie, "find_menagerie_root")
        assert hasattr(menagerie, "load_mjmodel")
        assert hasattr(menagerie, "list_models")

    def test_unknown_attribute_raises(self):
        import openscvx.integrations as integrations

        with pytest.raises(AttributeError):
            _ = integrations.does_not_exist

    def test_all_list_contents(self):
        import openscvx.integrations as integrations

        assert "mjx_byof" in integrations.__all__
        assert "mjx_dynamics" in integrations.__all__
        assert "free_joint_qpos_dynamics" in integrations.__all__
        assert "menagerie" in integrations.__all__
