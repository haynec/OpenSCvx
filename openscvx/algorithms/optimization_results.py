from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from openscvx.algorithms.history import AlgorithmHistory
    from openscvx.algorithms.multishot import MultishotPropagation
    from openscvx.algorithms.state import AlgorithmState
    from openscvx.problem import Problem


@jax.tree_util.register_pytree_node_class
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

    !!! note "JAX pytree registration"
        ``OptimizationResults`` is a registered JAX pytree, so the output of
        :meth:`~openscvx.problem.Problem.solve_jax` composes with
        ``jax.vmap`` / ``jax.jit`` / ``jax.grad``. The pytree *children* are
        ``converged``, ``t_final``, ``nodes``, ``trajectory``, ``X``, ``U``,
        and every ``*_history`` field — the data the user actually consumes.
        Post-process fields (``t_full``, ``x_full``, ``u_full``, ``cost``,
        ``ctcs_violation``), ``plotting_data``, and the internal ``_states`` /
        ``_controls`` metadata are stashed in the treedef's *aux* instead, so
        a batched ``solve_jax`` doesn't force callers to handle ``None``
        leaves and ``post_process()`` outputs stay outside the vmap surface
        (post-process per element after the batched solve).

        The two construction paths produce different histories:

        * :meth:`from_history` (``solve()``) populates the per-iteration
          ``X`` / ``U`` / ``*_history`` lists from the Python loop's
          :class:`~openscvx.algorithms.history.AlgorithmHistory`.
        * :meth:`from_final_state` (``solve_jax()``) wraps only the final
          iterate: ``X = [state.x]`` and ``U = [state.u]`` (single-element
          lists so ``result.x`` / ``result.u`` continue to return the final
          iterate), and every ``*_history`` field is empty — per-iteration
          history requires Python-side list growth that can't run inside
          ``lax.while_loop``.

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

    # Pytree children: fields whose contents flow through ``jax.vmap`` /
    # ``jax.jit`` / ``jax.grad``. Excludes ``_states`` / ``_controls`` (Python
    # symbolic metadata), ``plotting_data`` (user scratch), and the
    # post-process fields (``None`` until ``post_process()`` runs and not part
    # of the ``solve_jax`` vmap surface — see the class docstring).
    _PYTREE_CHILDREN = (
        "converged",
        "t_final",
        "nodes",
        "trajectory",
        "X",
        "U",
        "discretization_history",
        "J_tr_history",
        "J_vb_history",
        "J_vc_history",
        "TR_history",
        "VC_history",
        "lam_prox_history",
        "actual_reduction_history",
        "pred_reduction_history",
        "acceptance_ratio_history",
    )

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
    lam_prox_history: list[np.ndarray] = field(default_factory=list, metadata={"npz": "array_list"})
    actual_reduction_history: list[float] = field(
        default_factory=list, metadata={"npz": "float_list"}
    )
    pred_reduction_history: list[float] = field(
        default_factory=list, metadata={"npz": "float_list"}
    )
    acceptance_ratio_history: list[float] = field(
        default_factory=list, metadata={"npz": "float_list"}
    )

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

    def multishot_propagation(
        self,
        iteration: int = -1,
        *,
        t_nodes: Optional[np.ndarray] = None,
    ) -> Optional["MultishotPropagation"]:
        """Return unpacked multishot data for SCP iteration ``iteration`` (default: final).

        Returns ``None`` when ``discretization_history`` is empty (e.g. ``solve_jax``).

        Attaches ``self._states`` so ``.state("q")`` works without passing slices manually.
        """
        from openscvx.algorithms.multishot import unpack_multishot_V

        dh = self.discretization_history
        if not dh:
            return None
        V = np.asarray(dh[iteration], dtype=np.float64)
        n_x, n_u = self.x.shape[1], self.u.shape[1]
        if t_nodes is None:
            t_nodes_raw = self.nodes.get("time")
            if t_nodes_raw is None:
                t_final = float(self.t_final)
                t_nodes = np.linspace(0.0, t_final, self.x.shape[0])
            else:
                t_nodes = np.asarray(t_nodes_raw, dtype=np.float64).ravel()
        return unpack_multishot_V(
            V, n_x=n_x, n_u=n_u, t_nodes=t_nodes, states=tuple(self._states or ())
        )

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

    # ------------------------------------------------------------------
    # JAX pytree registration
    # ------------------------------------------------------------------

    def tree_flatten(self):
        """Split into JAX-traceable children and host-side aux.

        See :attr:`_PYTREE_CHILDREN` for the children list. Aux carries the
        symbolic metadata, scratch dict, and post-process fields verbatim so
        the round-trip ``tree_unflatten(*tree_flatten(self))`` preserves the
        instance.
        """
        children = tuple(getattr(self, name) for name in self._PYTREE_CHILDREN)
        aux = (
            self._states,
            self._controls,
            self.t_full,
            self.x_full,
            self.u_full,
            self.cost,
            self.ctcs_violation,
            self.plotting_data,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(cls._PYTREE_CHILDREN, children))
        (
            _states,
            _controls,
            t_full,
            x_full,
            u_full,
            cost,
            ctcs_violation,
            plotting_data,
        ) = aux
        return cls(
            **kwargs,
            _states=_states,
            _controls=_controls,
            t_full=t_full,
            x_full=x_full,
            u_full=u_full,
            cost=cost,
            ctcs_violation=ctcs_violation,
            plotting_data=plotting_data,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_final_state(
        cls,
        state: "AlgorithmState",
        *,
        problem: "Problem",
    ) -> "OptimizationResults":
        """JAX-pure construction from the final SCP iterate.

        Used by :meth:`~openscvx.problem.Problem.solve_jax`. Every leaf is a
        ``jnp.ndarray`` derived from ``state`` via static slicing, so the
        result composes with ``jax.vmap`` / ``jax.jit`` / ``jax.grad`` over a
        batched ``state``. ``X = [state.x]`` and ``U = [state.u]`` are
        single-element lists so ``result.x`` / ``result.u`` continue to
        return the final iterate; every ``*_history`` field is empty — the
        Python-loop ``.solve()`` is the path that populates them.
        """
        settings = problem.settings
        symbolic = problem.symbolic

        has_impulsive_controls = any(
            control.parameterization == "impulsive" for control in symbolic.controls
        )

        nodes_dict: dict[str, jnp.ndarray] = {}
        for sym_state in symbolic.states:
            state_nodes = state.x[..., sym_state._slice]
            if has_impulsive_controls:
                bc = jnp.asarray(settings.sim.x.initial[sym_state._slice])
                state_nodes = state_nodes.at[..., 0, :].set(bc)
            nodes_dict[sym_state.name] = state_nodes

        for control in symbolic.controls:
            nodes_dict[control.name] = state.u[..., control._slice]

        # Match the host-path ``t_final`` shape: a 1-vector along the time
        # slice of the last node (vmap'd to ``(B, 1)``).
        t_final = state.x[..., -1, :][..., settings.sim.time_slice]

        # Convergence flag through the algorithm's own predicate, so this JAX
        # results path agrees with ``solve()`` / the ``lax.while_loop``. The
        # per-element thresholds ride the state pytree, so a batched solve with
        # a tolerance sweep reports convergence against each element's ep_*.
        converged = problem.algorithm.converged(state)

        return cls(
            converged=converged,
            t_final=t_final,
            nodes=nodes_dict,
            trajectory={},
            _states=symbolic.states_prop,
            _controls=symbolic.controls,
            X=[state.x],
            U=[state.u],
            discretization_history=[],
            J_tr_history=[],
            J_vb_history=[],
            J_vc_history=[],
            TR_history=[],
            VC_history=[],
            lam_prox_history=[],
            actual_reduction_history=[],
            pred_reduction_history=[],
            acceptance_ratio_history=[],
        )

    @classmethod
    def from_history(
        cls,
        history: "AlgorithmHistory",
        final_state: "AlgorithmState",
        *,
        problem: "Problem",
        converged: bool,
    ) -> "OptimizationResults":
        """Host-side construction from the Python-loop iteration history.

        Used by :meth:`~openscvx.problem.Problem.solve` to package the full
        per-iteration log produced by the Python ``while`` loop. Symmetric
        with :meth:`from_final_state` — same return type, populated history.
        """
        settings = problem.settings
        symbolic = problem.symbolic

        has_impulsive_controls = any(
            control.parameterization == "impulsive" for control in symbolic.controls
        )

        x_arr = np.asarray(final_state.x)
        u_arr = np.asarray(final_state.u)

        nodes_dict: dict[str, np.ndarray] = {}
        for sym_state in symbolic.states:
            state_nodes = x_arr[:, sym_state._slice].copy()
            if has_impulsive_controls:
                state_nodes[0] = settings.sim.x.initial[sym_state._slice]
            nodes_dict[sym_state.name] = state_nodes

        for control in symbolic.controls:
            nodes_dict[control.name] = u_arr[:, control._slice]

        return cls(
            converged=converged,
            t_final=x_arr[:, settings.sim.time_slice][-1],
            nodes=nodes_dict,
            trajectory={},
            _states=symbolic.states_prop,
            _controls=symbolic.controls,
            X=history.X,
            U=history.U,
            discretization_history=history.V_history,
            J_tr_history=float(final_state.J_tr),
            J_vb_history=float(final_state.J_vb),
            J_vc_history=float(final_state.J_vc),
            TR_history=history.TR,
            VC_history=history.VC,
            lam_prox_history=history.lam_prox.copy(),
            actual_reduction_history=history.actual_reduction.copy(),
            pred_reduction_history=history.pred_reduction.copy(),
            acceptance_ratio_history=history.acceptance_ratio.copy(),
        )

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
