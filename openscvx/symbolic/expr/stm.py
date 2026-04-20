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


_STM_MODES = ("approx", "exact")


def _validate_stm_mode(name: str, mode: str) -> str:
    if mode not in _STM_MODES:
        raise ValueError(
            f"STM '{name}': mode must be one of {_STM_MODES}, got {mode!r}"
        )
    return mode


class STMPhysical(State):
    """Running state transition matrix of the physical state block.

    Represents ``Φ(τ) = ∂x_phys(τ)/∂x_phys(0)`` as a flat ``(n_phys*n_phys,)``
    state slice (row-major). Identity-initialized; dynamics
    ``Φ̇ = A_phys · Φ`` emitted by the discretizer/propagator (not the user).

    The ``mode`` attribute selects the SCP Jacobian treatment:

    - ``"approx"`` (default): ``jax.lax.stop_gradient`` wraps STM reads so the
      SCP treats Φ as a frozen input — CTCS rows see only ``∂/∂x_phys``.
    - ``"exact"``: SCP sees CTCS sensitivity through Φ; requires second-order
      sensitivity Ψ to close the chain (not yet implemented — raises).
    """

    _is_stm = True
    _stm_kind = "physical"
    _is_integration_only = True

    def __init__(self, name: str, n_phys: int, mode: str = "approx"):
        if n_phys <= 0:
            raise ValueError(
                f"STMPhysical '{name}': n_phys must be positive, got {n_phys}"
            )
        super().__init__(name, shape=(n_phys * n_phys,))
        self.n_phys = int(n_phys)
        self.mode = _validate_stm_mode(name, mode)

    def __repr__(self) -> str:
        return f"STMPhysical('{self.name}', n_phys={self.n_phys}, mode={self.mode!r})"


class STMImpulse(State):
    """Running sensitivity of the physical state to an impulsive control.

    Represents ``Φ_imp(τ) = ∂x_phys(τ)/∂u_imp`` for a specific impulsive
    control channel; shape ``(n_phys,)``. Zero-initialized, with the unit
    direction of ``control`` injected at the impulse node by the propagator.
    Continuous dynamics ``Φ̇_imp = A_phys · Φ_imp`` emitted by the discretizer.

    ``mode`` mirrors :class:`STMPhysical` (``"approx"`` / ``"exact"``).
    """

    _is_stm = True
    _stm_kind = "impulse"
    _is_integration_only = True

    def __init__(
        self,
        name: str,
        n_phys: int,
        control: Optional["Control"] = None,
        mode: str = "approx",
    ):
        if n_phys <= 0:
            raise ValueError(
                f"STMImpulse '{name}': n_phys must be positive, got {n_phys}"
            )
        super().__init__(name, shape=(n_phys,))
        self.n_phys = int(n_phys)
        self.control = control
        self.mode = _validate_stm_mode(name, mode)

    def __repr__(self) -> str:
        ctrl_name = self.control.name if self.control is not None else None
        return (
            f"STMImpulse('{self.name}', n_phys={self.n_phys}, "
            f"control={ctrl_name!r}, mode={self.mode!r})"
        )


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
