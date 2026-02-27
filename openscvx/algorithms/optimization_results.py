from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

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


    !!! note "For Developers"
        The ``metadata={"npz": ...}`` parameter on each field below is a built-in feature of
        `dataclasses.field`.  It attaches a read-only mapping to the **field definition**,
        *not* to instances. Instances still only have the normal attributes (``result.X``,
        ``result.converged``, etc.).

        The ``save()`` / ``load()`` methods iterate over field metadata to
        determine how to serialize each field, so adding a new field only
        requires tagging it here, no separate field-name lists to maintain.

        Fields *without* an ``"npz"`` metadata entry are skipped during
        serialization (e.g. ``_states``, ``_controls``).
    """

    _SCALAR = "scalar"
    _ARRAY_LIST = "array_list"
    _FLOAT_LIST = "float_list"
    _DICT = "dict"
    _OPT_ARRAY = "optional_array"
    _OPT_SCALAR = "optional_scalar"

    # Core optimization results
    converged: bool = field(metadata={"npz": "scalar"})
    t_final: float = field(metadata={"npz": "scalar"})

    # Dictionary-based access to states and controls
    nodes: dict[str, np.ndarray] = field(default_factory=dict, metadata={"npz": "dict"})
    trajectory: dict[str, np.ndarray] = field(default_factory=dict, metadata={"npz": "dict"})

    # Internal metadata for dictionary construction (not serialized)
    _states: list = field(default_factory=list, repr=False)
    _controls: list = field(default_factory=list, repr=False)

    # History of SCP iterations (single source of truth)
    X: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    U: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    discretization_history: list[np.ndarray] = field(
        default_factory=list, metadata={"npz": "array_list"}
    )
    J_tr_history: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    J_vb_history: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    J_vc_history: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    TR_history: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    VC_history: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})

    # Convergence histories
    lam_prox_history: list[float] = field(default_factory=list, metadata={"npz": "float_list"})
    actual_reduction_history: list[float] = field(
        default_factory=list, metadata={"npz": "float_list"}
    )
    pred_reduction_history: list[float] = field(
        default_factory=list, metadata={"npz": "float_list"}
    )
    acceptance_ratio_history: list[float] = field(
        default_factory=list, metadata={"npz": "float_list"}
    )

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
    t_full: Optional[np.ndarray] = field(default=None, metadata={"npz": "optional_array"})
    x_full: Optional[np.ndarray] = field(default=None, metadata={"npz": "optional_array"})
    u_full: Optional[np.ndarray] = field(default=None, metadata={"npz": "optional_array"})
    cost: Optional[float] = field(default=None, metadata={"npz": "optional_scalar"})
    ctcs_violation: Optional[np.ndarray] = field(default=None, metadata={"npz": "optional_array"})

    # Additional plotting/application data (added by user)
    plotting_data: dict[str, Any] = field(default_factory=dict, metadata={"npz": "dict"})

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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save results to a compressed ``.npz`` file.

        Serialization behaviour is driven by the ``"npz"`` key in each
        dataclass field's ``metadata``.  Fields without that key are
        skipped.

        Args:
            path: Output file path.  ``.npz`` is appended automatically
                by numpy if not already present.
        """
        data: dict[str, np.ndarray] = {}

        for name, f in self.__dataclass_fields__.items():
            tag = f.metadata.get("npz")
            if tag is None:
                continue
            val = getattr(self, name)

            if tag == self._SCALAR:
                data[name] = np.array(val)

            elif tag == self._ARRAY_LIST:
                data[name] = np.stack([np.asarray(a) for a in val]) if val else np.array([])

            elif tag == self._FLOAT_LIST:
                data[name] = np.array(val, dtype=float)

            elif tag == self._DICT:
                for k, v in val.items():
                    try:
                        arr = np.asarray(v)
                        if arr.dtype.kind != "O":
                            data[f"{name}/{k}"] = arr
                    except (TypeError, ValueError):
                        pass

            elif tag == self._OPT_ARRAY:
                if val is not None:
                    data[name] = np.asarray(val)

            elif tag == self._OPT_SCALAR:
                if val is not None:
                    data[name] = np.array(val)

        np.savez_compressed(str(path), **data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OptimizationResults":
        """Load results from a ``.npz`` file previously created by :meth:`save`.

        Args:
            path: Path to the ``.npz`` file.  If the path has no suffix,
                ``.npz`` is appended automatically.

        Returns:
            Reconstructed :class:`OptimizationResults`.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")

        archive = np.load(str(path), allow_pickle=False)
        keys = set(archive.files)

        def _unstack(name: str) -> list:
            if name not in keys:
                return []
            arr = archive[name]
            return [arr[i] for i in range(arr.shape[0])] if arr.size else []

        def _prefixed_dict(prefix: str) -> dict:
            p = prefix + "/"
            return {k[len(p) :]: archive[k] for k in keys if k.startswith(p)}

        kwargs: dict[str, Any] = {}
        deferred: dict[str, tuple] = {}  # fields to set after construction

        for name, f in cls.__dataclass_fields__.items():
            tag = f.metadata.get("npz")
            if tag is None:
                continue

            if tag == cls._SCALAR:
                kwargs[name] = archive[name].item()

            elif tag == cls._ARRAY_LIST:
                kwargs[name] = _unstack(name)

            elif tag == cls._FLOAT_LIST:
                kwargs[name] = archive[name].tolist() if name in keys else []

            elif tag == cls._DICT:
                kwargs[name] = _prefixed_dict(name)

            elif tag == cls._OPT_ARRAY:
                deferred[name] = archive[name] if name in keys else None

            elif tag == cls._OPT_SCALAR:
                deferred[name] = float(archive[name]) if name in keys else None

        result = cls(**kwargs)
        for name, val in deferred.items():
            setattr(result, name, val)
        return result
