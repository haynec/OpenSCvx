"""STM metadata surfaced to the discretizer/propagator.

Frozen dataclasses so instances are hashable — required when passed as
static args to JAX-compiled integrators (e.g. diffrax).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


def _slice_key(sl: Optional[slice]) -> Optional[Tuple[Optional[int], Optional[int], Optional[int]]]:
    return None if sl is None else (sl.start, sl.stop, sl.step)


@dataclass(frozen=True)
class StmSlot:
    """One STM augmented-state slot in the unified state vector.

    ``psi_slice`` indexes the slot's second-order sensitivity block Ψ inside
    the discretizer's internal Ψ region (not the unified state vector). It is
    set only for ``mode="exact"`` physical slots (impulse Ψ is trivially zero
    in continuous segments and handled separately at impulsive nodes).
    """

    name: str
    kind: str  # "physical" | "impulse"
    slice: slice
    n_phys: int
    control_slice: Optional[slice] = None
    control_name: Optional[str] = None
    mode: str = "approx"  # SCP Jacobian treatment: "approx" | "exact"
    psi_slice: Optional[slice] = None
    # When set on a "physical" slot, Φ is identity-injected ONLY at this node
    # (zero-initialized globally) and propagates continuously through every
    # subsequent segment without per-segment reset. ``None`` keeps the default
    # behavior of resetting to identity at every segment start.
    anchor_node: Optional[int] = None

    def __hash__(self) -> int:
        # Python's ``slice`` is not hashable; key on (start, stop, step) tuples.
        return hash(
            (
                self.name,
                self.kind,
                _slice_key(self.slice),
                self.n_phys,
                _slice_key(self.control_slice),
                self.control_name,
                self.mode,
                _slice_key(self.psi_slice),
                self.anchor_node,
            )
        )


@dataclass(frozen=True)
class StmMeta:
    """STM slot table; empty iff the problem declares no STM augmented states."""

    slots: Tuple[StmSlot, ...] = field(default_factory=tuple)
    n_phys: int = 0  # physical-state dimension; shared by every STM slot
    psi_size: int = 0  # total size of the Ψ region (sum over exact physical slots)

    @property
    def is_empty(self) -> bool:
        return len(self.slots) == 0

    @property
    def has_exact(self) -> bool:
        return any(slot.mode == "exact" for slot in self.slots)
