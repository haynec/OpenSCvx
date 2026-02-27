"""SymbolicProblem dataclass - container for symbolic problem specification.

This module provides the SymbolicProblem dataclass that represents a trajectory
optimization problem in symbolic form, before lowering to executable code.

The SymbolicProblem can represent two lifecycle stages:

1. **Before preprocessing**: Raw user input with unsorted constraints
2. **After preprocessing**: Augmented and validated, ready for lowering

Use `is_preprocessed` to check which stage the problem is in.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from openscvx.symbolic.constraint_set import ConstraintSet

if TYPE_CHECKING:
    from openscvx.symbolic.expr import Expr
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


@dataclass
class SymbolicProblem:
    """Container for symbolic problem specification.

    This dataclass holds a trajectory optimization problem in symbolic form,
    either as raw user input or after preprocessing/augmentation. It provides
    a typed interface for the preprocessing and lowering pipeline.

    Lifecycle Stages:
        1. **Before preprocessing**: User creates with raw dynamics, states,
           controls, and unsorted constraints. Propagation fields are None.
        2. **After preprocessing**: Dynamics and states are augmented (CTCS,
           time dilation), constraints are categorized, propagation fields
           are populated.

    Use `is_preprocessed` to check whether preprocessing has completed.

    Attributes:
        dynamics: Symbolic dynamics expression (dx/dt = f(x, u)).
            After preprocessing, includes CTCS augmented state dynamics.
        states: List of State objects. After preprocessing, includes
            time state and CTCS augmented states.
        controls: List of Control objects. After preprocessing, includes
            time dilation control.
        constraints: ConstraintSet holding all constraints. Before preprocessing,
            raw constraints live in `constraints.unsorted`. After preprocessing,
            constraints are categorized into ctcs, nodal, nodal_convex, etc.
        parameters: Dictionary mapping parameter names to numpy arrays.
        N: Number of discretization nodes.
        node_intervals: List of (start, end) tuples for CTCS constraint intervals.
            Populated during preprocessing when CTCS constraints are sorted.

        dynamics_prop: Propagation dynamics (may include extra states).
            None before preprocessing, populated after.
        states_prop: Propagation states (may include extra states).
            None before preprocessing, populated after.
        controls_prop: Propagation controls (typically same as controls).
            None before preprocessing, populated after.
        algebraic_prop: Algebraic outputs computed during propagation (no integration).
            None before preprocessing, populated after.

    Example:
        Before preprocessing::

            problem = SymbolicProblem(
                dynamics=dynamics_expr,
                states=[x, v],
                controls=[u],
                constraints=ConstraintSet(unsorted=[c1, c2, c3]),
                parameters={"mass": 1.0},
                N=50,
            )
            assert not problem.is_preprocessed

        After preprocessing::

            processed = preprocess_symbolic_problem(problem, time=time_config)
            assert processed.is_preprocessed
            assert processed.constraints.is_categorized
            # Now ready for lowering
            lowered = lower_symbolic_problem(processed)
    """

    # Core problem specification
    dynamics: "Expr"
    states: List["State"]
    controls: List["Control"]
    constraints: ConstraintSet
    parameters: Dict[str, any]
    N: int

    # Discrete dynamics
    dynamics_discrete: Optional["Expr"] = None

    # CTCS node intervals (populated during preprocessing)
    node_intervals: List[Tuple[int, int]] = field(default_factory=list)

    # Propagation (None before preprocessing, populated after)
    dynamics_prop: Optional["Expr"] = None
    states_prop: Optional[List["State"]] = None
    controls_prop: Optional[List["Control"]] = None

    # Algebraic outputs computed during propagation (no integration)
    # Maps output names to symbolic expressions
    algebraic_prop: Optional[Dict[str, "Expr"]] = None

    @property
    def is_preprocessed(self) -> bool:
        """True if the problem has been preprocessed and is ready for lowering.

        A problem is considered preprocessed when:
        1. All constraints have been categorized (unsorted is empty)
        2. Propagation dynamics have been set up
        """
        return self.constraints.is_categorized and self.dynamics_prop is not None
