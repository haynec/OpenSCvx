from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class OptimizationResults:
    """
    Structured container for optimization results from the Successive Convexification (SCP) solver.

    This class provides a type-safe and organized way to store and access optimization results,
    replacing the previous dictionary-based approach. It includes core optimization data,
    iteration history for convergence analysis, post-processing results, and flexible
    storage for plotting and application-specific data.

    Attributes:
        converged (bool): Whether the optimization successfully converged.
        t_final (float): Final time of the optimized trajectory.
        x_guess (np.ndarray): Optimized state trajectory at discretization nodes,
            shape (N, n_states).
        u_guess (np.ndarray): Optimized control trajectory at discretization nodes,
            shape (N, n_controls).
        nodes (dict[str, np.ndarray]): Dictionary mapping state/control names to arrays
            at optimization nodes. Includes both user-defined and augmented variables.
        trajectory (dict[str, np.ndarray]): Dictionary mapping state/control names to arrays
            along the propagated trajectory. Added by post_process().
        x_history (list[np.ndarray]): State trajectories from each SCP iteration.
        u_history (list[np.ndarray]): Control trajectories from each SCP iteration.
        discretization_history (list[np.ndarray]): Time discretization from each iteration.
        J_tr_history (list[np.ndarray]): Trust region cost history.
        J_vb_history (list[np.ndarray]): Virtual buffer cost history.
        J_vc_history (list[np.ndarray]): Virtual control cost history.
        t_full (Optional[np.ndarray]): Full time grid for interpolated trajectory.
            Added by propagate_trajectory_results.
        x_full (Optional[np.ndarray]): Interpolated state trajectory on full time grid.
            Added by propagate_trajectory_results.
        u_full (Optional[np.ndarray]): Interpolated control trajectory on full time grid.
            Added by propagate_trajectory_results.
        cost (Optional[float]): Total cost of the optimized trajectory.
            Added by propagate_trajectory_results.
        ctcs_violation (Optional[np.ndarray]): Continuous-time constraint violations.
            Added by propagate_trajectory_results.
        plotting_data (dict[str, Any]): Flexible storage for plotting and application data.
    """

    # Core optimization results
    converged: bool
    t_final: float

    # Dictionary-based access to states and controls
    nodes: dict[str, np.ndarray] = field(default_factory=dict)
    trajectory: dict[str, np.ndarray] = field(default_factory=dict)

    # Internal metadata for dictionary construction
    _states: list = field(default_factory=list, repr=False)
    _controls: list = field(default_factory=list, repr=False)

    # History of SCP iterations (single source of truth)
    X: list[np.ndarray] = field(default_factory=list)
    U: list[np.ndarray] = field(default_factory=list)
    discretization_history: list[np.ndarray] = field(default_factory=list)
    J_tr_history: list[np.ndarray] = field(default_factory=list)
    J_vb_history: list[np.ndarray] = field(default_factory=list)
    J_vc_history: list[np.ndarray] = field(default_factory=list)
    TR_history: list[np.ndarray] = field(default_factory=list)
    VC_history: list[np.ndarray] = field(default_factory=list)

    # Convergence histories
    lam_prox_history: list[float] = field(default_factory=list)
    actual_reduction_history: list[float] = field(default_factory=list)
    pred_reduction_history: list[float] = field(default_factory=list)
    acceptance_ratio_history: list[float] = field(default_factory=list)

    @property
    def x(self) -> np.ndarray:
        """Optimal state trajectory at discretization nodes.

        Returns the final converged solution from the SCP iteration history.

        Returns:
            State trajectory array, shape (N, n_states)
        """
        return self.X[-1]

    @property
    def u(self) -> np.ndarray:
        """Optimal control trajectory at discretization nodes.

        Returns the final converged solution from the SCP iteration history.

        Returns:
            Control trajectory array, shape (N, n_controls)
        """
        return self.U[-1]

    # Post-processing results (added by propagate_trajectory_results)
    t_full: Optional[np.ndarray] = None
    x_full: Optional[np.ndarray] = None
    u_full: Optional[np.ndarray] = None
    cost: Optional[float] = None
    ctcs_violation: Optional[np.ndarray] = None

    # Additional plotting/application data (added by user)
    plotting_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the results object."""
        pass

    def update_plotting_data(self, **kwargs: Any) -> None:
        """
        Update the plotting data with additional information.

        Args:
            **kwargs: Key-value pairs to add to plotting_data
        """
        self.plotting_data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the results, similar to dict.get().

        Args:
            key: The key to look up
            default: Default value if key is not found

        Returns:
            The value associated with the key, or default if not found
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key)

        # Check if it's in plotting_data
        if key in self.plotting_data:
            return self.plotting_data[key]

        return default

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to results.

        Args:
            key: The key to look up

        Returns:
            The value associated with the key

        Raises:
            KeyError: If key is not found
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key)

        # Check if it's in plotting_data
        if key in self.plotting_data:
            return self.plotting_data[key]

        raise KeyError(f"Key '{key}' not found in results")

    def __setitem__(self, key: str, value: Any):
        """
        Allow dictionary-style assignment to results.

        Args:
            key: The key to set
            value: The value to assign
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            # Store in plotting_data
            self.plotting_data[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the results.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        return hasattr(self, key) or key in self.plotting_data

    def update(self, other: dict[str, Any]):
        """
        Update the results with additional data from a dictionary.

        Args:
            other: Dictionary containing additional data
        """
        for key, value in other.items():
            self[key] = value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the results to a dictionary for backward compatibility.

        Returns:
            Dictionary representation of the results
        """
        result_dict = {}

        # Add all direct attributes
        for attr_name in self.__dataclass_fields__:
            if attr_name != "plotting_data":
                result_dict[attr_name] = getattr(self, attr_name)

        # Add plotting data
        result_dict.update(self.plotting_data)

        return result_dict
