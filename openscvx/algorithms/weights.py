from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


def _expand_lam_cost_dict(
    lam_cost_dict: Dict[str, Union[float, list, np.ndarray]],
    states: List["State"],
) -> np.ndarray:
    """Expand a ``{state_name: weight}`` dict to a per-state weight array.

    Maps user-provided per-state cost weights to a dense array of shape
    ``(n_states,)`` using each state's ``_slice``.  States without a
    minimize/maximize objective receive weight 0.  States **with** a
    minimize/maximize objective **must** appear in the dict.

    Values may be scalars (broadcast to every component of that state) or
    arrays matching the state's shape for per-component weighting, e.g.
    ``{"position": [0, 0, 1e-6]}``.

    Args:
        lam_cost_dict: Mapping from state names to cost weights (scalar or
            array matching the state's shape).
        states: List of State objects (must already have ``_slice`` assigned).

    Returns:
        np.ndarray of shape ``(n_states,)`` with per-index weights.

    Raises:
        ValueError: If the dict contains unknown state names or is missing
            entries for states that have minimize/maximize objectives.
    """
    n_states = sum(s.shape[0] if len(s.shape) > 0 else 1 for s in states)
    lam_arr = np.zeros(n_states)

    valid_names = {s.name for s in states}

    # Check for unknown keys
    unknown = set(lam_cost_dict.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"lam_cost dict contains unknown state name(s): {unknown}. "
            f"Valid state names: {sorted(valid_names)}"
        )

    # Identify states that have minimize/maximize objectives.
    # initial_type/final_type are set on symbolic State objects during property
    # assignment (e.g. state.initial = ...), so they are available before lowering.
    cost_states: Set[str] = set()
    for state in states:
        if state.initial_type is not None:
            for t in state.initial_type:
                if t in ("Minimize", "Maximize"):
                    cost_states.add(state.name)
                    break
        if state.final_type is not None:
            for t in state.final_type:
                if t in ("Minimize", "Maximize"):
                    cost_states.add(state.name)
                    break

    # Check that all cost states are in the dict
    missing = cost_states - set(lam_cost_dict.keys())
    if missing:
        raise ValueError(
            f"lam_cost dict is missing weight(s) for state(s) with "
            f"minimize/maximize objectives: {missing}. All states with "
            f"cost terms must have a weight in the dict."
        )

    # Fill the array.  _slice is assigned by preprocess_symbolic_problem
    # (via collect_and_assign_slices), which runs before algorithm construction.
    for state in states:
        if state.name in lam_cost_dict:
            val = np.asarray(lam_cost_dict[state.name], dtype=float)
            n_components = state.shape[0] if len(state.shape) > 0 else 1
            if val.ndim > 0 and val.shape[0] != n_components:
                raise ValueError(
                    f"lam_cost['{state.name}'] has length {val.shape[0]}, "
                    f"expected scalar or length {n_components}"
                )
            lam_arr[state._slice] = val

    return lam_arr


def _expand_lam_vc_dict(
    lam_vc_dict: Dict[str, Union[float, list, np.ndarray]],
    states: List["State"],
) -> np.ndarray:
    """Expand a ``{state_name: weight}`` dict to a per-state VC weight array.

    Maps user-provided per-state virtual control weights to a dense array
    using each state's ``_slice``.  States not present in the dict default
    to ``1.0`` (neutral weight — dynamics violations are still penalised at
    the base rate).

    Per-state values may be:

    * **scalar** — broadcast to every component and every node.
    * **1-D array** of length ``n_components`` — per-component weight,
      same for every node.
    * **2-D array** of shape ``(K, n_components)`` — per-node-per-component.
      When any state supplies a 2-D value the output is 2-D with first
      dimension *K* (all 2-D entries must agree on *K*).

    Args:
        lam_vc_dict: Mapping from state names to VC weights.
        states: List of State objects (must already have ``_slice`` assigned).

    Returns:
        np.ndarray of shape ``(n_states,)`` when all entries are scalar/1-D,
        or ``(K, n_states)`` when any entry is 2-D.

    Raises:
        ValueError: If the dict contains unknown state names or 2-D entries
            disagree on the number of nodes.
    """
    n_states = sum(s.shape[0] if len(s.shape) > 0 else 1 for s in states)

    valid_names = {s.name for s in states}
    unknown = set(lam_vc_dict.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"lam_vc dict contains unknown state name(s): {unknown}. "
            f"Valid state names: {sorted(valid_names)}"
        )

    # First pass: determine if any entry is 2-D and infer K.
    n_nodes: Optional[int] = None
    for state in states:
        if state.name not in lam_vc_dict:
            continue
        val = np.asarray(lam_vc_dict[state.name], dtype=float)
        if val.ndim == 2:
            if n_nodes is None:
                n_nodes = val.shape[0]
            elif val.shape[0] != n_nodes:
                raise ValueError(
                    f"lam_vc['{state.name}'] has {val.shape[0]} rows, but a "
                    f"previous entry had {n_nodes} rows. All 2-D entries "
                    f"must have the same number of rows (n_nodes-1)."
                )

    # Build the output array.
    if n_nodes is not None:
        lam_arr = np.ones((n_nodes, n_states))
    else:
        lam_arr = np.ones(n_states)

    for state in states:
        if state.name not in lam_vc_dict:
            continue
        val = np.asarray(lam_vc_dict[state.name], dtype=float)
        n_components = state.shape[0] if len(state.shape) > 0 else 1

        if val.ndim == 0:
            # Scalar — broadcast to all components (and all nodes if 2-D).
            lam_arr[..., state._slice] = float(val)
        elif val.ndim == 1:
            if val.shape[0] != n_components:
                raise ValueError(
                    f"lam_vc['{state.name}'] has length {val.shape[0]}, "
                    f"expected scalar or length {n_components}"
                )
            lam_arr[..., state._slice] = val
        elif val.ndim == 2:
            if val.shape[1] != n_components:
                raise ValueError(
                    f"lam_vc['{state.name}'] has {val.shape[1]} columns, expected {n_components}"
                )
            lam_arr[:, state._slice] = val
        else:
            raise ValueError(
                f"lam_vc['{state.name}'] has {val.ndim} dimensions, expected scalar, 1-D, or 2-D"
            )

    return lam_arr


def _expand_lam_prox_dict(
    lam_prox_dict: Dict[str, Union[float, list, np.ndarray]],
    states: List["State"],
    controls: List["Control"],
) -> np.ndarray:
    """Expand a ``{name: weight}`` dict to a per-variable proximal weight array.

    Maps user-provided per-state/per-control trust region weights to a dense
    array using each variable's ``_slice``.  Variables not present in the dict
    default to ``1.0``.

    Per-variable values may be:

    * **scalar** — broadcast to every component and every node.
    * **1-D array** of length ``n_components`` — per-component weight,
      same for every node.
    * **2-D array** of shape ``(K, n_components)`` — per-node-per-component.
      When any variable supplies a 2-D value the output is 2-D with first
      dimension *K* (all 2-D entries must agree on *K*).

    Args:
        lam_prox_dict: Mapping from state/control names to proximal weights.
        states: List of State objects (must already have ``_slice`` assigned).
        controls: List of Control objects (must already have ``_slice`` assigned).

    Returns:
        np.ndarray of shape ``(n_states + n_controls,)`` when all entries are
        scalar/1-D, or ``(K, n_states + n_controls)`` when any entry is 2-D.

    Raises:
        ValueError: If the dict contains unknown names or 2-D entries disagree
            on the number of nodes.
    """
    n_states = sum(s.shape[0] if len(s.shape) > 0 else 1 for s in states)
    n_controls = sum(c.shape[0] if len(c.shape) > 0 else 1 for c in controls)
    n_total = n_states + n_controls

    valid_state_names = {s.name for s in states}
    valid_control_names = {c.name for c in controls}
    valid_names = valid_state_names | valid_control_names
    unknown = set(lam_prox_dict.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"lam_prox dict contains unknown name(s): {unknown}. Valid names: {sorted(valid_names)}"
        )

    # Build a unified list of (name, n_components, slice_in_output).
    # States occupy columns [0, n_states), controls occupy [n_states, n_total).
    variables: list = []
    for s in states:
        nc = s.shape[0] if len(s.shape) > 0 else 1
        variables.append((s.name, nc, s._slice))
    for c in controls:
        nc = c.shape[0] if len(c.shape) > 0 else 1
        out_slice = slice(n_states + c._slice.start, n_states + c._slice.stop)
        variables.append((c.name, nc, out_slice))

    # First pass: determine if any entry is 2-D and infer K.
    n_nodes: Optional[int] = None
    for name, n_comp, _ in variables:
        if name not in lam_prox_dict:
            continue
        val = np.asarray(lam_prox_dict[name], dtype=float)
        if val.ndim == 2:
            if n_nodes is None:
                n_nodes = val.shape[0]
            elif val.shape[0] != n_nodes:
                raise ValueError(
                    f"lam_prox['{name}'] has {val.shape[0]} rows, but a "
                    f"previous entry had {n_nodes} rows. All 2-D entries "
                    f"must have the same number of rows (n_nodes)."
                )

    # Build the output array.
    if n_nodes is not None:
        lam_arr = np.ones((n_nodes, n_total))
    else:
        lam_arr = np.ones(n_total)

    for name, n_comp, out_slice in variables:
        if name not in lam_prox_dict:
            continue
        val = np.asarray(lam_prox_dict[name], dtype=float)

        if val.ndim == 0:
            lam_arr[..., out_slice] = float(val)
        elif val.ndim == 1:
            if val.shape[0] != n_comp:
                raise ValueError(
                    f"lam_prox['{name}'] has length {val.shape[0]}, "
                    f"expected scalar or length {n_comp}"
                )
            lam_arr[..., out_slice] = val
        elif val.ndim == 2:
            if val.shape[1] != n_comp:
                raise ValueError(
                    f"lam_prox['{name}'] has {val.shape[1]} columns, expected {n_comp}"
                )
            lam_arr[:, out_slice] = val
        else:
            raise ValueError(
                f"lam_prox['{name}'] has {val.ndim} dimensions, expected scalar, 1-D, or 2-D"
            )

    return lam_arr


@dataclass
class Weights:
    """SCP weights used internally by the algorithm and autotuner.

    Users should read and write weights through the algorithm's properties
    (e.g. ``algorithm.lam_cost``). The autotuner may mutate these fields
    during SCP iteration; those mutations are reflected in the weight
    histories on :class:`AlgorithmState`.

    Use :meth:`build` to construct from user-friendly inputs (floats or
    ``{name: weight}`` dicts). Use :meth:`build_vb_arrays` to populate
    ``lam_vb_nodal`` / ``lam_vb_cross`` once symbolic constraints are
    available.

    Attributes:
        lam_prox: Trust region (proximal) weight. Scalar or
            array of shape ``(n_states + n_controls,)`` or
            ``(N, n_states + n_controls)`` for per-variable / per-node
            weighting.
        lam_vc: Virtual control penalty weight. Scalar or
            array of shape ``(n_states,)`` or ``(n_nodes-1, n_states)``
            for per-state / per-node weighting.
        lam_cost: Cost weight per state. Scalar or array of
            shape ``(n_states,)`` for per-state weighting.
        lam_vb: Global virtual buffer penalty weight. Scalar
            default applied to every constraint. Use ``.weight()`` on
            individual constraints for per-constraint or per-node overrides.
        lam_vb_nodal: Virtual buffer penalty weights for nodal constraints,
            shape ``(N, n_nodal)``. Set by :meth:`build_vb_arrays`.
        lam_vb_cross: Virtual buffer penalty weights for cross-node
            constraints, shape ``(n_cross,)``. Set by
            :meth:`build_vb_arrays`.
    """

    lam_prox: Union[float, np.ndarray] = 1e0
    lam_vc: Union[float, np.ndarray] = 1e1
    lam_cost: Union[float, np.ndarray] = 1e-1
    lam_vb: float = 0.0
    lam_vb_nodal: Optional[np.ndarray] = None
    lam_vb_cross: Optional[np.ndarray] = None

    def __post_init__(self):
        # Coerce lists/lists-of-lists to numpy arrays.
        if isinstance(self.lam_prox, (list, tuple)):
            self.lam_prox = np.asarray(self.lam_prox, dtype=float)
        if isinstance(self.lam_vc, (list, tuple)):
            self.lam_vc = np.asarray(self.lam_vc, dtype=float)
        if isinstance(self.lam_cost, (list, tuple)):
            self.lam_cost = np.asarray(self.lam_cost, dtype=float)

    @classmethod
    def build(
        cls,
        lam_prox: Union[float, Dict[str, Union[float, list, np.ndarray]]] = 1e0,
        lam_vc: Union[float, Dict[str, Union[float, list, np.ndarray]]] = 1e1,
        lam_cost: Union[float, Dict[str, float]] = 1e-1,
        lam_vb: float = 0.0,
        states: Optional[List["State"]] = None,
        controls: Optional[List["Control"]] = None,
    ) -> "Weights":
        """Construct Weights from user-friendly inputs.

        Accepts floats (applied uniformly) or dicts mapping state/control
        names to per-variable weights. Dict inputs are expanded to dense
        arrays via the ``_expand_lam_*_dict`` helpers.

        Args:
            lam_prox: Trust region weight. Float or ``{name: weight}`` dict.
                Dict requires *states* and *controls*.
            lam_vc: Virtual control weight. Float or ``{state_name: weight}``
                dict. Dict requires *states*.
            lam_cost: Cost weight. Float or ``{state_name: weight}`` dict.
                Dict requires *states*.
            lam_vb: Virtual buffer default weight (scalar).
            states: Symbolic State objects (required when any weight is a dict).
            controls: Symbolic Control objects (required when *lam_prox* is a dict).

        Returns:
            A new Weights instance with resolved numeric values.

        Raises:
            ValueError: If a dict weight is given without the required
                states/controls.
        """
        return cls(
            lam_prox=cls.resolve_lam_prox(lam_prox, states, controls),
            lam_vc=cls.resolve_lam_vc(lam_vc, states),
            lam_cost=cls.resolve_lam_cost(lam_cost, states),
            lam_vb=float(lam_vb),
        )

    @staticmethod
    def resolve_lam_prox(
        lam_prox: Union[float, Dict[str, Union[float, list, np.ndarray]]],
        states: Optional[List["State"]] = None,
        controls: Optional[List["Control"]] = None,
    ) -> Union[float, np.ndarray]:
        """Resolve a ``lam_prox`` spec to a numeric value.

        If *lam_prox* is a float it is returned as-is. If it is a dict,
        *states* and *controls* must be provided for expansion.
        """
        if isinstance(lam_prox, dict):
            if states is None or controls is None:
                raise ValueError(
                    "lam_prox was specified as a dict but states and/or "
                    "controls were not provided. Pass both so the dict can "
                    "be expanded to a per-variable weight array."
                )
            return _expand_lam_prox_dict(lam_prox, states, controls)
        return lam_prox

    @staticmethod
    def resolve_lam_vc(
        lam_vc: Union[float, Dict[str, Union[float, list, np.ndarray]]],
        states: Optional[List["State"]] = None,
    ) -> Union[float, np.ndarray]:
        """Resolve a ``lam_vc`` spec to a numeric value.

        If *lam_vc* is a float it is returned as-is. If it is a dict,
        *states* must be provided for expansion.
        """
        if isinstance(lam_vc, dict):
            if states is None:
                raise ValueError(
                    "lam_vc was specified as a dict but no states were "
                    "provided. Pass states so the dict can be expanded to "
                    "a per-state weight array."
                )
            return _expand_lam_vc_dict(lam_vc, states)
        return lam_vc

    @staticmethod
    def resolve_lam_cost(
        lam_cost: Union[float, Dict[str, float]],
        states: Optional[List["State"]] = None,
    ) -> Union[float, np.ndarray]:
        """Resolve a ``lam_cost`` spec to a numeric value.

        If *lam_cost* is a float it is returned as-is. If it is a dict,
        *states* must be provided for expansion.
        """
        if isinstance(lam_cost, dict):
            if states is None:
                raise ValueError(
                    "lam_cost was specified as a dict but no states were "
                    "provided. Pass states so the dict can be expanded to "
                    "a per-state weight array."
                )
            return _expand_lam_cost_dict(lam_cost, states)
        return lam_cost

    def build_vb_arrays(
        self,
        N: int,
        nodal_constraints: list,
        cross_node_constraints: list,
        n_byof_nodal: int = 0,
        n_byof_cross: int = 0,
    ) -> None:
        """Build per-constraint virtual buffer weight arrays.

        Inspects each symbolic constraint's shape (to account for vector
        decomposition) and ``.weight()`` overrides, then populates
        ``self.lam_vb_nodal`` and ``self.lam_vb_cross``.

        Args:
            N: Number of trajectory nodes.
            nodal_constraints: Symbolic ``NodalConstraint`` objects (post-
                preprocessing, pre-lowering).
            cross_node_constraints: Symbolic ``CrossNodeConstraint`` objects.
            n_byof_nodal: Number of byof nodal constraints (each adds one
                column with the default weight).
            n_byof_cross: Number of byof cross-node constraints (each adds
                one entry with the default weight).
        """
        default_vb = float(self.lam_vb)

        # Count decomposed nodal constraints (vector → multiple scalars).
        # Vector constraints are decomposed element-wise during lowering
        # (see decompose_vector_nodal_constraints), so each element gets its
        # own column.  We mirror that here via check_shape() to ensure the
        # array dimensions match the post-decomposition constraint count.
        n_nodal = 0
        for nc in nodal_constraints:
            shape = nc.constraint.lhs.check_shape()
            n_nodal += int(np.prod(shape)) if len(shape) > 0 else 1

        # Byof constraints are scalar (one column each), added after symbolic.
        n_nodal += n_byof_nodal
        n_cross = len(cross_node_constraints) + n_byof_cross

        # max(..., 1) avoids size-0 CVXPy parameters.
        n_nodal_param = max(n_nodal, 1)
        n_cross_param = max(n_cross, 1)

        lam_vb_nodal = np.full((N, n_nodal_param), default_vb)
        lam_vb_cross = np.full(n_cross_param, default_vb)

        # Apply per-constraint .weight() overrides for nodal constraints.
        col = 0
        for nc in nodal_constraints:
            shape = nc.constraint.lhs.check_shape()
            n_elem = int(np.prod(shape)) if len(shape) > 0 else 1

            w = nc._lam_vb
            if w is not None:
                nodes = nc.nodes if nc.nodes is not None else list(range(N))
                if isinstance(w, (int, float)):
                    lam_vb_nodal[nodes, col : col + n_elem] = float(w)
                elif isinstance(w, np.ndarray):
                    if w.ndim == 1:
                        # (n_elem,) — broadcast across nodes
                        for i in range(n_elem):
                            val = float(w[0]) if len(w) == 1 else float(w[i])
                            lam_vb_nodal[nodes, col + i] = val
                    elif w.ndim == 2:
                        # (n_nodes, n_elem) — per-node-per-element
                        for i in range(n_elem):
                            c_i = 0 if w.shape[1] == 1 else i
                            lam_vb_nodal[nodes, col + i] = w[:, c_i]

            col += n_elem

        # Apply per-constraint .weight() overrides for cross-node constraints.
        for idx, cc in enumerate(cross_node_constraints):
            if cc._lam_vb is not None:
                lam_vb_cross[idx] = float(cc._lam_vb)

        self.lam_vb_nodal = lam_vb_nodal
        self.lam_vb_cross = lam_vb_cross
