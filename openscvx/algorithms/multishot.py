"""Read-side helpers for SCP multishot integration matrices ``V``.

:class:`MultishotPropagation` complements
:class:`~openscvx.algorithms.history.DiscretizationResult`, which unpacks the same
packed layout for nodal linearization matrices. This module exposes the full
substep state trajectories stored during multishot discretization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.symbolic.expr.state import State

StateLike = Union[str, "State"]


def _resolve_state_slice(name_or_state: StateLike, states: Sequence) -> slice:
    from openscvx.symbolic.expr.state import State

    if isinstance(name_or_state, State):
        if name_or_state._slice is None:
            raise ValueError(f"State {name_or_state.name!r} has no slice assigned")
        return name_or_state._slice
    for state in states:
        if state.name == name_or_state:
            if state._slice is None:
                raise ValueError(f"State {name_or_state!r} has no slice assigned")
            return state._slice
    available = sorted({state.name for state in states})
    raise KeyError(f"Unknown state {name_or_state!r}. Available: {available}")


def segment_size(n_x: int, n_u: int) -> int:
    """Packed row count per multishot segment (state + STM + sensitivities)."""
    return n_x + n_x * n_x + 2 * n_x * n_u


@dataclass(frozen=True)
class MultishotPropagation:
    """Unpacked SCP multishot integration matrix ``V``.

    ``V`` shape: ``(n_segments * segment_size, n_substeps)`` where
    ``segment_size = n_x + n_x**2 + 2 * n_x * n_u``.
    """

    V: np.ndarray
    n_x: int
    n_u: int
    t_nodes: np.ndarray
    states: tuple = ()

    @property
    def segment_size(self) -> int:
        return segment_size(self.n_x, self.n_u)

    @property
    def n_segments(self) -> int:
        return int(self.V.shape[0] // self.segment_size)

    @property
    def n_substeps(self) -> int:
        return int(self.V.shape[1])

    def segment_states(self, seg_idx: int) -> np.ndarray:
        """Integrated states for one segment, shape ``(n_substeps, n_x)``."""
        seg_start = seg_idx * self.segment_size
        rows = self.V[seg_start : seg_start + self.n_x, :]
        return np.asarray(rows, dtype=np.float64).T

    def segments(self) -> list[np.ndarray]:
        """All segment state arrays, each shape ``(n_substeps, n_x)``."""
        return [self.segment_states(seg_idx) for seg_idx in range(self.n_segments)]

    def chronological(self) -> tuple[np.ndarray, np.ndarray]:
        """Time-ordered stitched full states and times.

        Skips duplicated segment-boundary samples (``j0 = 0`` on seg 0, else ``1``).

        Returns:
            ``states`` — ``(n_samples, n_x)``
            ``t`` — ``(n_samples,)``, linearly interpolated within each segment
        """
        t_nodes = np.asarray(self.t_nodes, dtype=np.float64).ravel()
        n_sub = self.n_substeps
        state_rows: list[np.ndarray] = []
        t_rows: list[float] = []
        for seg_idx in range(self.n_segments):
            t0, t1 = float(t_nodes[seg_idx]), float(t_nodes[seg_idx + 1])
            j0 = 0 if seg_idx == 0 else 1
            for j in range(j0, n_sub):
                alpha = j / (n_sub - 1) if n_sub > 1 else 0.0
                state_rows.append(self.segment_states(seg_idx)[j])
                t_rows.append((1.0 - alpha) * t0 + alpha * t1)
        if not state_rows:
            return np.empty((0, self.n_x), dtype=np.float64), np.empty(0, dtype=np.float64)
        return np.stack(state_rows, axis=0), np.asarray(t_rows, dtype=np.float64)

    def state(self, name_or_state: StateLike) -> tuple[np.ndarray, np.ndarray]:
        """Chronological trajectory for one symbolic state.

        ``name_or_state`` may be a :class:`~openscvx.symbolic.expr.state.State`
        instance or a string name matching ``State.name`` in ``self.states``.
        """
        state_slice = _resolve_state_slice(name_or_state, self.states)
        return self.slice_states(state_slice)

    def slice_states(self, state_slice: slice) -> tuple[np.ndarray, np.ndarray]:
        """``chronological()`` then apply ``state_slice`` to the state dimension."""
        states, t = self.chronological()
        return states[:, state_slice], t


def unpack_multishot_V(
    V: np.ndarray,
    *,
    n_x: int,
    n_u: int,
    t_nodes: np.ndarray,
    states: Sequence = (),
) -> MultishotPropagation:
    """Validate ``V`` layout and return :class:`MultishotPropagation`.

    Raises:
        ValueError: invalid ``V`` shape, empty matrix, or ``len(t_nodes) != n_segments + 1``.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.size == 0:
        raise ValueError("multishot V matrix is empty")
    seg_size = segment_size(n_x, n_u)
    if seg_size <= 0:
        raise ValueError(f"invalid multishot dimensions n_x={n_x}, n_u={n_u}")
    n_rows, n_sub = V.shape
    if n_sub < 1:
        raise ValueError("multishot V must have at least one substep column")
    if n_rows % seg_size != 0:
        raise ValueError(
            f"multishot V row count {n_rows} is not divisible by segment_size {seg_size}"
        )
    n_segments = n_rows // seg_size
    t_nodes_arr = np.asarray(t_nodes, dtype=np.float64).ravel()
    if len(t_nodes_arr) != n_segments + 1:
        raise ValueError(
            f"multishot t_nodes length {len(t_nodes_arr)} != n_segments + 1 "
            f"({n_segments + 1}); pass optimization node times from "
            "results.nodes['time'] or an explicit array."
        )
    return MultishotPropagation(
        V=V,
        n_x=n_x,
        n_u=n_u,
        t_nodes=t_nodes_arr,
        states=tuple(states),
    )
