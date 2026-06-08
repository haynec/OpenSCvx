"""
Unit tests for plotting functions.

Tests the plotting functions:
- plot_states: Plot state trajectories in subplot grid
- plot_controls: Plot control trajectories in subplot grid
- plot_state_component: Plot single state component
- plot_control_component: Plot single control component
"""

from unittest.mock import Mock

import numpy as np
import pytest

from openscvx.algorithms import OptimizationResults
from openscvx.plotting.plotting import (
    plot_control_component,
    plot_controls,
    plot_state_component,
    plot_states,
)
from openscvx.plotting.publication import PublicationFigure, parse_var_spec


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
