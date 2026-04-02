"""Base class for dynamics linearization and discretization.

This module defines the abstract interface for discretizers — components that
convert continuous-time nonlinear dynamics into discrete-time linear
approximations around a reference trajectory.

Discretization and linearization are inherently coupled in trajectory
optimization. Different schemes may:

- **Linearize then discretize**: Compute continuous-time Jacobians (df/dx, df/du),
  then integrate the variational equations to obtain discrete-time matrices.
- **Discretize then linearize**: Integrate the nonlinear dynamics to form a
  discrete map, then differentiate through the integrator.
- **Analytical methods**: Use matrix exponentials, Euler approximations, etc.

Since the ordering of these operations changes the intermediate types, a single
base class handles both, keeping the input/output contract consistent regardless
of internal strategy.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered.dynamics import Dynamics

#: Accepted type for ``dis_type``: a single string applied to every control,
#: or a per-control sequence.
DisType = Union[str, Sequence[str]]


def _make_foh_mask(dis_type: DisType, n_u: int) -> np.ndarray:
    """Convert a ``dis_type`` specification into a boolean FOH mask.

    Args:
        dis_type: ``"FOH"`` / ``"ZOH"`` (applies to all controls), or a
            sequence of length ``n_u`` with one entry per control.
        n_u: Total number of controls (including any augmented controls
            such as time-dilation).

    Returns:
        Boolean array of shape ``(n_u,)`` — ``True`` for FOH controls,
        ``False`` for ZOH controls.
    """
    if isinstance(dis_type, str):
        if dis_type == "FOH":
            return np.ones(n_u, dtype=bool)
        if dis_type == "ZOH":
            return np.zeros(n_u, dtype=bool)
        raise ValueError(f"Unknown dis_type: {dis_type!r}; expected 'FOH' or 'ZOH'")

    if len(dis_type) != n_u:
        raise ValueError(
            f"dis_type has {len(dis_type)} entries but expected {n_u} (one per control)"
        )
    mask = np.empty(n_u, dtype=bool)
    for i, dt in enumerate(dis_type):
        if dt == "FOH":
            mask[i] = True
        elif dt == "ZOH":
            mask[i] = False
        else:
            raise ValueError(f"Unknown dis_type[{i}]: {dt!r}; expected 'FOH' or 'ZOH'")
    return mask


def _resolve_foh_mask(
    dis_type: DisType,
    n_u: int,
    u_foh_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the final per-control FOH mask.

    Merges the discretizer-level ``dis_type`` default with any per-control
    ``parameterization`` (``"FOH"`` / ``"ZOH"``) on individual :class:`Control`
    objects and aggregated into ``u_foh_mask`` during unification.

    Control-level settings take precedence over the discretizer default.

    Args:
        dis_type: Discretizer-level default (``"FOH"``, ``"ZOH"``, or a
            per-control sequence).  Used as a fallback for controls whose
            ``parameterization`` does not set FOH/ZOH.
        n_u: Total number of controls.
        u_foh_mask: Optional float array of shape ``(n_u,)`` from
            :attr:`UnifiedControl.foh_mask`.  Values are ``1.0`` (FOH),
            ``0.0`` (ZOH), or ``nan`` (unset — use ``dis_type``).
            ``None`` means no control set FOH/ZOH ``parameterization``.

    Returns:
        Float array of shape ``(n_u,)`` with values ``1.0`` (FOH) or
        ``0.0`` (ZOH).
    """
    base = _make_foh_mask(dis_type, n_u).astype(float)
    if u_foh_mask is None:
        return base
    unset = np.isnan(u_foh_mask)
    return np.where(unset, base, u_foh_mask)


class Discretizer(ABC):
    """Abstract base class for dynamics linearization and discretization.

    This class defines the interface for converting continuous-time nonlinear
    dynamics into discrete-time linear approximations suitable for convex
    subproblems in successive convexification.

    The lifecycle mirrors other OpenSCvx ABCs:

    **Setup (called once):**

    - get_solver: Build a callable that computes discrete-time matrices

    **Per-iteration (via the returned callable):**

    - The callable is invoked with a reference trajectory and parameters,
      returning discretized matrices (A_d, B_d, C_d, x_prop)

    Discretization parameters (hold type, integrator, tolerances) live on each
    concrete subclass as instance attributes.

    Example:
        Implementing a custom discretizer::

            class EulerDiscretizer(Discretizer):
                def get_solver(self, dynamics, settings):
                    def solver(x, u, params):
                        # Euler discretization of dynamics
                        ...
                        return A_d, B_d, C_d, x_prop, V
                    return solver

                def citation(self):
                    return []
    """

    #: Control hold type. A single ``"FOH"`` or ``"ZOH"`` string applies the
    #: same hold to every control.  A sequence (e.g.
    #: ``["FOH", "ZOH", "FOH"]``) sets the hold independently for each
    #: control, and is merged with any per-control ``Control.parameterization``
    #: (``"FOH"`` / ``"ZOH"``).
    #: Subclasses must set this in ``__init__``.
    dis_type: DisType

    #: ODE solver name used for integration (e.g., ``"Tsit5"``).  Subclasses
    #: must set this in ``__init__``.
    ode_solver: str

    @abstractmethod
    def get_solver(self, dynamics: "Dynamics", settings: "Config") -> callable:
        """Create a discretization solver callable.

        Called once during problem initialization. Returns a function that
        computes linearized discrete-time dynamics matrices around a reference
        trajectory. The returned callable will be JIT-compiled and cached by
        the framework.

        Implementations are responsible for computing any Jacobians they need.
        The ``dynamics`` object always provides ``dynamics.f`` (the continuous-
        time nonlinear dynamics). Implementations that linearize first may
        compute Jacobians via ``jax.jacfwd(dynamics.f, ...)``.

        Args:
            dynamics: System dynamics object. ``dynamics.f`` is the continuous-
                time nonlinear dynamics function with signature
                ``f(x, u, node, params) -> x_dot``.
            settings: Problem configuration (node count, scaling matrices, etc.).

        Returns:
            Callable with signature
            ``(x: ndarray, u: ndarray, params: dict) -> (A_d, B_d, C_d, x_prop, V)``
            where:

            - ``A_d``: (N-1, n_x, n_x) discretized state transition matrix
            - ``B_d``: (N-1, n_x, n_u) control influence matrix (current node)
            - ``C_d``: (N-1, n_x, n_u) control influence matrix (next node)
            - ``x_prop``: (N-1, n_x) propagated state
            - ``V``: raw integration data (implementation-specific, used for
                diagnostics and history tracking)
        """
        raise NotImplementedError

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this discretization method.

        Implementations should return a list of BibTeX entry strings for the
        papers that should be cited when using this discretization scheme.

        Returns:
            List of BibTeX citation strings.
        """
        raise NotImplementedError
