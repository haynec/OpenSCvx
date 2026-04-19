"""STM metadata surfaced to the discretizer/propagator."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StmSlot:
    """One STM augmented-state slot in the unified state vector."""

    name: str
    kind: str  # "physical" | "impulse"
    slice: slice
    n_phys: int
    control_slice: Optional[slice] = None
    control_name: Optional[str] = None


@dataclass
class StmMeta:
    """STM slot table; empty iff the problem declares no STM augmented states."""

    slots: List[StmSlot] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.slots) == 0
