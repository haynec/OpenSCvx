"""
Unit tests for plotting functions.

Tests the plotting functions:
- plot_states: Plot state trajectories in subplot grid
- plot_controls: Plot control trajectories in subplot grid
- plot_state_component: Plot single state component
- plot_control_component: Plot single control component
- viser scene primitives: per-node naming and colormap helpers
"""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from openscvx.algorithms import OptimizationResults
from openscvx.plotting.plotting import (
    plot_control_component,
    plot_controls,
    plot_state_component,
    plot_states,
)
from openscvx.plotting.publication import (
    PublicationFigure,
    apply_publication_plotly_layout,
    parse_var_spec,
    publication_dark_colors,
)
from openscvx.plotting.viser import (
    add_animated_trail,
    add_attitude_frame,
    add_ghost_trajectory,
    add_glideslope_cone,
    add_position_marker,
    add_thrust_plume,
    add_thrust_vector,
    add_viewcone,
    compute_velocity_colors,
    create_server,
)
from openscvx.plotting.viser import server as viser_server_module


class TestPlotStatesFunction:
    """Test suite for plot_states function."""

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        # Mock nodes dictionary
        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 3),
        }

        # Mock trajectory
        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "state_x": np.random.randn(100, 3),
        }

        # Mock states
        state1 = Mock()
        state1.name = "state_x"
        state1._slice = slice(0, 3)
        state1.min = None
        state1.max = None
        result._states = [state1]
        result._controls = []

        return result

    def test_plot_states_returns_figure(self, mock_result_basic):
        """Test that plot_states returns a valid Plotly figure."""
        fig = plot_states(mock_result_basic)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "State Trajectories"

    def test_plot_states_with_multiple_states(self):
        """Test plot_states with multiple state variables."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "position": np.random.randn(10, 2),
            "velocity": np.random.randn(10, 2),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "position": np.random.randn(100, 2),
            "velocity": np.random.randn(100, 2),
        }

        pos_state = Mock()
        pos_state.name = "position"
        pos_state._slice = slice(0, 2)
        pos_state.min = None
        pos_state.max = None

        vel_state = Mock()
        vel_state.name = "velocity"
        vel_state._slice = slice(2, 4)
        vel_state.min = None
        vel_state.max = None

        result._states = [pos_state, vel_state]
        result._controls = []

        fig = plot_states(result)

        # Should have subplots for each state component (4 total)
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_states_with_state_names_filter(self):
        """Test plot_states with specific state names."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "position": np.random.randn(10, 2),
            "velocity": np.random.randn(10, 2),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "position": np.random.randn(100, 2),
            "velocity": np.random.randn(100, 2),
        }

        pos_state = Mock()
        pos_state.name = "position"
        pos_state._slice = slice(0, 2)
        pos_state.min = None
        pos_state.max = None

        vel_state = Mock()
        vel_state.name = "velocity"
        vel_state._slice = slice(2, 4)
        vel_state.min = None
        vel_state.max = None

        result._states = [pos_state, vel_state]
        result._controls = []

        # Only plot position
        fig = plot_states(result, ["position"])

        assert fig is not None
        # Should only have traces for position (2 components * 2 traces each = 4)
        # Each component gets trajectory + nodes trace
        assert len(fig.data) == 4

    def test_plot_states_single_component_spec(self):
        """Test plot_states with a single multidimensional component."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "position": np.random.randn(10, 3),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "position": np.random.randn(100, 3),
        }

        pos_state = Mock()
        pos_state.name = "position"
        pos_state._slice = slice(0, 3)
        pos_state.min = None
        pos_state.max = None

        result._states = [pos_state]
        result._controls = []

        fig = plot_states(result, ["position:2"])
        assert fig is not None
        assert len(fig.data) == 2

        fig_tuple = plot_states(result, [("position", 1)])
        assert len(fig_tuple.data) == 2

    def test_plot_states_publication_style(self, mock_result_basic, tmp_path):
        """Test publication style returns wrapper and saves PDF."""
        pdf_path = tmp_path / "states.pdf"
        fig = plot_states(
            mock_result_basic,
            style="publication",
            pdf_path=pdf_path,
        )
        assert isinstance(fig, PublicationFigure)
        assert pdf_path.is_file()
        # 3 panels in one row: 3 * 320 + margins
        assert fig.layout.width == 1088
        assert fig.layout.height == 400
        assert fig.layout.autosize is False

    def test_plot_states_with_empty_trajectory(self):
        """Test plot_states when trajectory is empty."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 3),
        }

        result.trajectory = {}  # Empty trajectory

        state = Mock()
        state.name = "state_x"
        state._slice = slice(0, 3)
        state.min = None
        state.max = None
        result._states = [state]
        result._controls = []

        fig = plot_states(result)

        assert fig is not None
        # Should still plot node markers even without full trajectory

    def test_plot_states_filters_private_states(self):
        """Test that plot_states filters out private states (starting with _)."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 2),
            "_ctcs_aug_0": np.random.randn(10, 1),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "state_x": np.random.randn(100, 2),
            "_ctcs_aug_0": np.random.randn(100, 1),
        }

        state = Mock()
        state.name = "state_x"
        state._slice = slice(0, 2)
        state.min = None
        state.max = None

        aug_state = Mock()
        aug_state.name = "_ctcs_aug_0"
        aug_state._slice = slice(2, 3)
        aug_state.min = None
        aug_state.max = None

        result._states = [state, aug_state]
        result._controls = []

        fig = plot_states(result)

        assert fig is not None
        # Private states should be filtered out, so we should only see state_x

    def test_plot_states_include_private(self):
        """Test plot_states with include_private=True."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 2),
            "_ctcs_aug_0": np.random.randn(10, 1),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "state_x": np.random.randn(100, 2),
            "_ctcs_aug_0": np.random.randn(100, 1),
        }

        state = Mock()
        state.name = "state_x"
        state._slice = slice(0, 2)
        state.min = None
        state.max = None

        aug_state = Mock()
        aug_state.name = "_ctcs_aug_0"
        aug_state._slice = slice(2, 3)
        aug_state.min = None
        aug_state.max = None

        result._states = [state, aug_state]
        result._controls = []

        fig = plot_states(result, include_private=True)

        assert fig is not None
        # Should include all 3 components (2 from state_x + 1 from _ctcs_aug_0)
        # Each gets 2 traces (trajectory + nodes)
        assert len(fig.data) == 6


class TestPlotStateComponentFunction:
    """Test suite for plot_state_component function."""

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "position": np.random.randn(10, 3),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "position": np.random.randn(100, 3),
        }

        state = Mock()
        state.name = "position"
        state._slice = slice(0, 3)
        state.min = None
        state.max = None
        result._states = [state]
        result._controls = []

        return result

    def test_plot_state_component_returns_figure(self, mock_result_basic):
        """Test that plot_state_component returns a valid Plotly figure."""
        fig = plot_state_component(mock_result_basic, "position", 0)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "position_0"

    def test_plot_state_component_different_components(self, mock_result_basic):
        """Test plotting different components."""
        for i in range(3):
            fig = plot_state_component(mock_result_basic, "position", i)
            assert fig is not None
            assert fig.layout.title.text == f"position_{i}"

    def test_plot_state_component_invalid_component(self, mock_result_basic):
        """Test that invalid component index raises error."""
        with pytest.raises(ValueError, match="out of range"):
            plot_state_component(mock_result_basic, "position", 5)

    def test_plot_state_component_invalid_state(self, mock_result_basic):
        """Test that invalid state name raises error."""
        with pytest.raises(ValueError, match="not found"):
            plot_state_component(mock_result_basic, "nonexistent", 0)


class TestPlotControlsFunction:
    """Test suite for plot_controls function."""

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "control_u": np.random.randn(10, 2),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "control_u": np.random.randn(100, 2),
        }

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 2)
        control.min = None
        control.max = None
        result._controls = [control]
        result._states = []

        return result

    def test_plot_controls_returns_figure(self, mock_result_basic):
        """Test that plot_controls returns a valid Plotly figure."""
        fig = plot_controls(mock_result_basic)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "Control Trajectories"

    def test_plot_controls_with_multiple_controls(self):
        """Test plot_controls with multiple control variables."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "thrust": np.random.randn(10, 2),
            "torque": np.random.randn(10, 1),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "thrust": np.random.randn(100, 2),
            "torque": np.random.randn(100, 1),
        }

        thrust_control = Mock()
        thrust_control.name = "thrust"
        thrust_control._slice = slice(0, 2)
        thrust_control.min = None
        thrust_control.max = None

        torque_control = Mock()
        torque_control.name = "torque"
        torque_control._slice = slice(2, 3)
        torque_control.min = None
        torque_control.max = None

        result._controls = [thrust_control, torque_control]
        result._states = []

        fig = plot_controls(result)

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_controls_with_control_names_filter(self):
        """Test plot_controls with specific control names."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "thrust": np.random.randn(10, 2),
            "torque": np.random.randn(10, 1),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "thrust": np.random.randn(100, 2),
            "torque": np.random.randn(100, 1),
        }

        thrust_control = Mock()
        thrust_control.name = "thrust"
        thrust_control._slice = slice(0, 2)
        thrust_control.min = None
        thrust_control.max = None

        torque_control = Mock()
        torque_control.name = "torque"
        torque_control._slice = slice(2, 3)
        torque_control.min = None
        torque_control.max = None

        result._controls = [thrust_control, torque_control]
        result._states = []

        # Only plot thrust
        fig = plot_controls(result, ["thrust"])

        assert fig is not None
        # Should only have traces for thrust (2 components * 2 traces each = 4)
        assert len(fig.data) == 4

    def test_plot_controls_with_empty_trajectory(self):
        """Test plot_controls when trajectory is empty."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "control_u": np.random.randn(10, 2),
        }

        result.trajectory = {}  # Empty trajectory

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 2)
        control.min = None
        control.max = None
        result._controls = [control]
        result._states = []

        fig = plot_controls(result)

        assert fig is not None

    def test_plot_controls_legend_only_on_first_subplot(self, mock_result_basic):
        """Test that legend items only appear on first subplot."""
        fig = plot_controls(mock_result_basic)

        # Count how many traces have showlegend=True
        legend_traces = [trace for trace in fig.data if trace.showlegend]

        # Should have exactly 2 legend traces (Trajectory and Nodes)
        assert len(legend_traces) == 2

    def test_plot_controls_single_component_spec(self, mock_result_basic):
        """Test plot_controls with a single multidimensional component."""
        fig = plot_controls(mock_result_basic, ["control_u:1"])
        assert fig is not None
        assert len(fig.data) == 2


class TestVarSpecParsing:
    """Test variable spec parsing for component selection."""

    def test_parse_var_spec_plain_name(self):
        assert parse_var_spec("position") == ("position", None)

    def test_parse_var_spec_colon_syntax(self):
        assert parse_var_spec("position:2") == ("position", 2)

    def test_parse_var_spec_bracket_syntax(self):
        assert parse_var_spec("velocity[0]") == ("velocity", 0)

    def test_parse_var_spec_tuple(self):
        assert parse_var_spec(("thrust_force", 2)) == ("thrust_force", 2)


class TestPlotControlComponentFunction:
    """Test suite for plot_control_component function."""

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "thrust": np.random.randn(10, 3),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "thrust": np.random.randn(100, 3),
        }

        control = Mock()
        control.name = "thrust"
        control._slice = slice(0, 3)
        control.min = None
        control.max = None
        result._controls = [control]
        result._states = []

        return result

    def test_plot_control_component_returns_figure(self, mock_result_basic):
        """Test that plot_control_component returns a valid Plotly figure."""
        fig = plot_control_component(mock_result_basic, "thrust", 0)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "thrust_0"

    def test_plot_control_component_different_components(self, mock_result_basic):
        """Test plotting different components."""
        for i in range(3):
            fig = plot_control_component(mock_result_basic, "thrust", i)
            assert fig is not None
            assert fig.layout.title.text == f"thrust_{i}"

    def test_plot_control_component_invalid_component(self, mock_result_basic):
        """Test that invalid component index raises error."""
        with pytest.raises(ValueError, match="out of range"):
            plot_control_component(mock_result_basic, "thrust", 5)

    def test_plot_control_component_invalid_control(self, mock_result_basic):
        """Test that invalid control name raises error."""
        with pytest.raises(ValueError, match="not found"):
            plot_control_component(mock_result_basic, "nonexistent", 0)


class TestPublicationDarkStyle:
    """Test the dark publication style and the layout helper it shares."""

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 3),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "state_x": np.random.randn(100, 3),
        }

        state = Mock()
        state.name = "state_x"
        state._slice = slice(0, 3)
        state.min = None
        state.max = None
        result._states = [state]
        result._controls = []

        return result

    def test_publication_dark_keeps_geometry_and_darkens_surround(
        self, mock_result_basic, tmp_path
    ):
        """publication_dark shares publication geometry but swaps the palette."""
        dark = publication_dark_colors()
        fig = plot_states(
            mock_result_basic,
            style="publication_dark",
            pdf_path=tmp_path / "states.pdf",
        )

        assert isinstance(fig, PublicationFigure)
        assert fig.layout.template == pio.templates["plotly_dark"]
        assert fig.layout.paper_bgcolor == dark["background"]
        assert fig.layout.plot_bgcolor == dark["background"]
        assert fig.layout.font.color == dark["foreground"]
        # Same fixed panel geometry as style="publication"
        assert fig.layout.width == 1088
        assert fig.layout.height == 400

    def test_publication_style_stays_white(self, mock_result_basic, tmp_path):
        """The default publication style is unaffected by the dark palette."""
        fig = plot_states(
            mock_result_basic,
            style="publication",
            pdf_path=tmp_path / "states.pdf",
        )
        assert fig.layout.paper_bgcolor == "white"
        assert fig.layout.template == pio.templates["plotly_white"]

    def test_layout_size_overrides(self):
        """Aspect-locked figures can override the grid-derived size."""
        fig = go.Figure()
        apply_publication_plotly_layout(fig, n_rows=2, n_cols=3, width=420, height=420)
        assert (fig.layout.width, fig.layout.height) == (420, 420)


class TestCreateServerScene:
    """``create_server`` is the only path that brands a scene, so it must fit every scene."""

    @staticmethod
    def _scene_nodes(monkeypatch, **kwargs) -> dict[str, str]:
        """Build a server against a recording stand-in and return the nodes it added."""
        server = _RecordingServer()
        server.gui = Mock()
        monkeypatch.setattr(viser_server_module.viser, "ViserServer", lambda **_: server)
        create_server(None, **kwargs)
        return server.scene.nodes

    def test_default_scene_keeps_grid_and_origin(self, monkeypatch):
        """Callers that pass nothing get today's scene unchanged."""
        nodes = self._scene_nodes(monkeypatch)
        assert {"/grid", "/origin"} <= set(nodes)

    def test_origin_can_be_suppressed(self, monkeypatch):
        """A half-metre triad dwarfs a scale-model scene, so it has to be optional."""
        nodes = self._scene_nodes(monkeypatch, show_grid=False, show_origin=False)
        assert nodes == {}

    def test_port_is_forwarded(self, monkeypatch):
        """Examples that run several servers at once need to choose their ports."""
        seen = {}

        def fake_server(**kwargs):
            seen.update(kwargs)
            server = _RecordingServer()
            server.gui = Mock()
            return server

        monkeypatch.setattr(viser_server_module.viser, "ViserServer", fake_server)
        create_server(None, port=8129)
        assert seen == {"port": 8129}

    def test_port_omitted_when_not_requested(self, monkeypatch):
        """Passing port=None must not force a port on viser's default behaviour."""
        seen = {}

        def fake_server(**kwargs):
            seen.update(kwargs)
            server = _RecordingServer()
            server.gui = Mock()
            return server

        monkeypatch.setattr(viser_server_module.viser, "ViserServer", fake_server)
        create_server(None)
        assert seen == {}


class _RecordingScene:
    """Stand-in for ``server.scene`` that records the path of every node added."""

    def __init__(self) -> None:
        self.nodes: dict[str, str] = {}

    def __getattr__(self, attr: str):
        if not attr.startswith("add_"):
            raise AttributeError(attr)

        def add(name: str, **kwargs) -> Mock:
            self.nodes[name] = attr
            return Mock()

        return add


class _RecordingServer:
    """Stand-in for a ``viser.ViserServer`` with only a recording scene."""

    def __init__(self) -> None:
        self.scene = _RecordingScene()


def _named_primitives() -> list:
    """Parametrize cases pairing each node-creating primitive with its default scene path."""
    n = 6
    pos = np.linspace(0.0, 1.0, n)[:, None] * np.ones(3)
    colors = np.tile(np.array([10, 20, 30], dtype=np.uint8), (n, 1))
    attitude = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    thrust = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))

    def case(add, default, label):
        return pytest.param(add, default, id=label)

    return [
        case(lambda s, **kw: add_animated_trail(s, pos, colors, **kw), "/trail", "trail"),
        case(lambda s, **kw: add_position_marker(s, pos, **kw), "/current_pos", "marker"),
        case(lambda s, **kw: add_attitude_frame(s, pos, attitude, **kw), "/body_frame", "frame"),
        case(lambda s, **kw: add_thrust_vector(s, pos, thrust, **kw), "/thrust_vector", "thrust"),
        case(lambda s, **kw: add_thrust_plume(s, pos, thrust, **kw), "/thrust_plume", "plume"),
        case(
            lambda s, **kw: add_viewcone(s, pos, attitude, 0.3, **kw),
            "/viewcone_sensor",
            "viewcone",
        ),
        case(
            lambda s, **kw: add_glideslope_cone(s, **kw),
            "/constraints/glideslope_cone",
            "glideslope",
        ),
        case(lambda s, **kw: add_ghost_trajectory(s, pos, colors, **kw), "/ghost_traj", "ghost"),
    ]


_NAMED_PRIMITIVES = _named_primitives()


class TestViserSceneNames:
    """Every primitive that creates a scene node must let the caller name it."""

    @pytest.mark.parametrize(("add", "default"), _NAMED_PRIMITIVES)
    def test_default_scene_path(self, add, default):
        """Omitting name= reproduces the historical single-vehicle scene path."""
        server = _RecordingServer()
        add(server)
        assert default in server.scene.nodes

    @pytest.mark.parametrize(("add", "default"), _NAMED_PRIMITIVES)
    def test_distinct_names_produce_distinct_nodes(self, add, default):
        """Two vehicles in one scene must not clobber each other."""
        server = _RecordingServer()
        add(server, name="/vehicle_a")
        add(server, name="/vehicle_b")

        assert {"/vehicle_a", "/vehicle_b"} <= set(server.scene.nodes)
        assert default not in server.scene.nodes
        # Any child geometry is parented under the caller's name, not the default.
        assert all(path.startswith(("/vehicle_a", "/vehicle_b")) for path in server.scene.nodes)


class TestViewconeSensorRotation:
    """``add_viewcone`` accepts a fixed or a gimballed sensor mounting."""

    @staticmethod
    def _pose(n: int) -> tuple[np.ndarray, np.ndarray]:
        pos = np.zeros((n, 3))
        attitude = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        return pos, attitude

    @staticmethod
    def _rotations_about_z(angles: np.ndarray) -> np.ndarray:
        c, s = np.cos(angles), np.sin(angles)
        zero, one = np.zeros_like(c), np.ones_like(c)
        return np.stack(
            [
                np.stack([c, -s, zero], axis=-1),
                np.stack([s, c, zero], axis=-1),
                np.stack([zero, zero, one], axis=-1),
            ],
            axis=-2,
        )

    def test_per_frame_rotation_moves_the_sensor_frame(self):
        """A gimballed mount reorients the cone even when the body pose is fixed."""
        pos, attitude = self._pose(4)
        R_sb = self._rotations_about_z(np.linspace(0.0, np.pi / 2, 4))

        frame, update = add_viewcone(_RecordingServer(), pos, attitude, 0.3, R_sb=R_sb)
        update(0)
        wxyz_first = frame.wxyz
        update(3)

        assert not np.allclose(wxyz_first, frame.wxyz)

    def test_constant_per_frame_rotation_matches_static(self):
        """An (N, 3, 3) mount that never changes equals the (3, 3) mount."""
        pos, attitude = self._pose(4)
        R_static = self._rotations_about_z(np.array([0.4]))[0]

        frame_static, update_static = add_viewcone(
            _RecordingServer(), pos, attitude, 0.3, R_sb=R_static
        )
        frame_series, update_series = add_viewcone(
            _RecordingServer(), pos, attitude, 0.3, R_sb=np.tile(R_static, (4, 1, 1))
        )
        update_static(2)
        update_series(2)

        assert np.allclose(frame_static.wxyz, frame_series.wxyz)

    def test_wrong_frame_count_raises(self):
        """A per-frame mount that does not cover the trajectory is a user error."""
        pos, attitude = self._pose(4)
        R_sb = self._rotations_about_z(np.zeros(3))

        with pytest.raises(ValueError, match="one rotation per frame"):
            add_viewcone(_RecordingServer(), pos, attitude, 0.3, R_sb=R_sb)

    def test_wrong_rotation_shape_raises(self):
        """Anything that is not (3, 3) or (N, 3, 3) is rejected up front."""
        pos, attitude = self._pose(4)

        with pytest.raises(ValueError, match=r"\(3, 3\) or \(N, 3, 3\)"):
            add_viewcone(_RecordingServer(), pos, attitude, 0.3, R_sb=np.zeros((4, 2)))


class TestComputeVelocityColors:
    """Velocity colormapping is unchanged by passing a preloaded colormap."""

    def test_preloaded_cmap_matches_lookup_by_name(self):
        vel = np.linspace(0.0, 4.0, 12)[:, None] * np.ones(3)

        by_name = compute_velocity_colors(vel, cmap_name="viridis")
        preloaded = compute_velocity_colors(vel, cmap=plt.get_cmap("viridis"))

        assert np.array_equal(by_name, preloaded)

    def test_colors_span_the_colormap(self):
        vel = np.linspace(0.0, 4.0, 12)[:, None] * np.ones(3)
        cmap = plt.get_cmap("viridis")

        colors = compute_velocity_colors(vel, cmap=cmap)

        assert colors.shape == (12, 3)
        assert np.array_equal(colors[0], (np.array(cmap(0.0)[:3]) * 255).astype(int))
        assert np.array_equal(colors[-1], (np.array(cmap(1.0)[:3]) * 255).astype(int))

    def test_fallback_length_used_when_velocity_missing(self):
        colors = compute_velocity_colors(None, fallback_length=5)

        assert colors.shape == (5, 3)
        assert len(np.unique(colors, axis=0)) == 1
