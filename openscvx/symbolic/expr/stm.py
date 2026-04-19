"""State transition matrix (STM) symbolic handles.

Symbolic leaves that expose *running* sensitivities along the reference
trajectory to augmented-state RHS expressions (primarily CTCS penalties).
They are **integration-only** auxiliary states — the discretizer propagates
them alongside the physical dynamics but they are not true SCP decision
variables (see the STM pipeline design doc / conversation for rationale).

This module defines the types and detection/validation helpers used by
the augmentation layer to enforce the one-way dependence invariant
(physical-state RHS must not read an STM). Variational-equation emission,
SCP/integration partitioning, boundary resets, and impulse-direction
injection are added in later steps and NOT implemented here — instantiating
an STM handle without the later wiring leaves it as an inert symbolic leaf.

Key constraint (enforced below via ``assert_no_stm``): physical-state RHS
must not read an STM. Only augmented-state RHS (e.g. CTCS penalties) may.
This one-way dependence is what keeps the variational equation
``dΦ/dτ = A_phys · Φ`` self-closing.
"""

from typing import TYPE_CHECKING, Dict, Iterable, Optional

from .expr import Expr
from .state import State

if TYPE_CHECKING:
    from .constraint import CTCS
    from .control import Control


class STMDependencyError(ValueError):
    """Raised when an STM handle is referenced in a disallowed context.

    The STM pipeline enforces a one-way dependence: physical-state RHS
    expressions must not read any STM leaf (``STMPhysical`` or
    ``STMImpulse``). Only augmented-state RHS (e.g. CTCS penalties) may
    read them. This invariant keeps ``dΦ/dτ = A_phys · Φ`` self-closing
    and decouples the variational equation from the SCP Jacobian.
    """


class STMPhysical(State):
    """Running state transition matrix of the physical state block.

    Represents ``Φ(τ) = ∂x_phys(τ)/∂x_phys(0)`` as a symbolic handle usable
    inside augmented-state RHS expressions. Internally stored as a flat
    ``(n_phys * n_phys,)`` state slice (row-major); ``n_phys`` is the
    physical-state dimension, which the augmentation pass will use when
    emitting the variational block.

    Initial value is the identity matrix (flattened) at each propagation
    segment start (or at the horizon start, depending on the reset mode
    configured on the problem). Dynamics are
    ``Φ̇ = A_phys(x_phys, u) · Φ`` emitted automatically by the augmentation
    pass — users do not write them.

    This class inherits from :class:`State` so that the existing symbolic
    machinery (slice assignment, hashing, lowering) can treat it uniformly;
    the augmentation / discretization layers flag it as
    ``_is_integration_only`` so it does not enter the SCP Jacobian
    calculation as a true state.

    Attributes:
        n_phys: Physical-state dimension ``n`` (``shape == (n*n,)``).
        _is_stm: Always ``True``.
        _stm_kind: Always ``"physical"``.
        _is_integration_only: Always ``True``. Read by the discretizer to
            partition V into SCP states vs integration-only slots.

    Example:
        Declare and reference inside a CTCS penalty (wiring added in later
        steps — this example is illustrative of the final API)::

            phi = ox.STMPhysical("phi", n_phys=2)
            # ... later, inside a CTCS expression ...
            # robust_term = phi @ B_disturbance   # once MatMul support is wired
    """

    _is_stm = True
    _stm_kind = "physical"
    _is_integration_only = True

    def __init__(self, name: str, n_phys: int):
        if n_phys <= 0:
            raise ValueError(
                f"STMPhysical '{name}': n_phys must be positive, got {n_phys}"
            )
        super().__init__(name, shape=(n_phys * n_phys,))
        self.n_phys = int(n_phys)

    def __repr__(self) -> str:
        return f"STMPhysical('{self.name}', n_phys={self.n_phys})"


class STMImpulse(State):
    """Running sensitivity of the physical state to an impulsive control.

    Represents ``Φ_imp(τ) = ∂x_phys(τ)/∂u_imp`` for a specified impulsive
    control channel, as a symbolic handle usable inside augmented-state
    RHS expressions. Shape is ``(n_phys,)``.

    Initial value is zero at each propagation segment start and is injected
    with the unit direction of the impulsive control channel at the
    impulse node's discrete jump (handled by the propagation layer in a
    later step). Continuous-time dynamics are ``Φ̇_imp = A_phys · Φ_imp``,
    emitted automatically by the augmentation pass.

    Attributes:
        n_phys: Physical-state dimension ``n`` (``shape == (n,)``).
        control: Reference to the impulsive :class:`Control` this handle
            tracks sensitivity to.
        _is_stm: Always ``True``.
        _stm_kind: Always ``"impulse"``.
        _is_integration_only: Always ``True``.

    Example:
        Declare against an impulsive control (wiring added in later steps
        — illustrative)::

            delta_v = ox.Control("delta_v", shape=(1,),
                                 parameterization="impulsive", nodes=[0, N-1])
            phi_imp = ox.STMImpulse("phi_imp", n_phys=2, control=delta_v)
    """

    _is_stm = True
    _stm_kind = "impulse"
    _is_integration_only = True

    def __init__(self, name: str, n_phys: int, control: Optional["Control"] = None):
        if n_phys <= 0:
            raise ValueError(
                f"STMImpulse '{name}': n_phys must be positive, got {n_phys}"
            )
        super().__init__(name, shape=(n_phys,))
        self.n_phys = int(n_phys)
        self.control = control

    def __repr__(self) -> str:
        ctrl_name = self.control.name if self.control is not None else None
        return f"STMImpulse('{self.name}', n_phys={self.n_phys}, control={ctrl_name!r})"


# ---------------------------------------------------------------------------
# Dependency check (physical-RHS gate)
# ---------------------------------------------------------------------------


def _collect_stm_into(expr: Expr, out: Dict[str, State]) -> None:
    """Populate ``out`` with STM leaves reachable from ``expr`` (name-keyed, dedup)."""
    if getattr(expr, "_is_stm", False) is True and expr.name not in out:
        out[expr.name] = expr
    for child in expr.children():
        _collect_stm_into(child, out)


def assert_no_stm(expr: Expr, context: str) -> None:
    """Raise :class:`STMDependencyError` if ``expr`` references any STM leaf."""
    offending: Dict[str, State] = {}
    _collect_stm_into(expr, offending)
    if not offending:
        return
    names = ", ".join(sorted(offending.keys()))
    raise STMDependencyError(
        f"STM handle(s) {{{names}}} referenced in {context}. "
        f"STM handles (STMPhysical, STMImpulse) may only appear in augmented-state "
        f"RHS expressions (e.g. CTCS penalties), not in physical-state dynamics."
    )


def collect_stm_leaves_from_ctcs(
    constraints_ctcs: Iterable["CTCS"],
) -> Dict[str, State]:
    """Return STM leaves referenced by any CTCS constraint (name-keyed, dedup)."""
    out: Dict[str, State] = {}
    for ctcs in constraints_ctcs:
        constraint = getattr(ctcs, "constraint", None)
        if constraint is None:
            continue
        _collect_stm_into(constraint, out)
    return out
