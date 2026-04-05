"""YAML / JSON problem loader.

Reads a structured problem definition from a YAML or JSON file (or a plain
Python dict) and returns the keyword arguments needed to construct a
:class:`openscvx.problem.Problem`.

Expected schema
---------------
.. code-block:: yaml

    N: 50

    time:
      initial: 0.0                 # or [minimize, 10.0]
      final: [minimize, 10.0]
      min: 0.0
      max: 20.0

    states:
      - name: pos
        shape: [3]
        min: [-10, -10, 0]
        max: [10, 10, 100]
        initial: [0, 0, 50]
        final: [10, [free, 5], 0]

    controls:
      - name: thrust
        shape: [3]
        # optional: parameterization: FOH | ZOH | impulsive (use ``nodes`` with impulsive)
        parameterization: ZOH
        min: [-10, -10, 0]
        max: [10, 10, 50]

    parameters:
      - name: gravity
        shape: [3]
        value: [0, 0, -9.81]

    dynamics:
      pos: "vel"
      vel: "thrust + gravity"

    constraints:
      - "Norm(pos[:2] - obs_center) >= 2.0"
      - "(vel[0] <= 3.0).at(0, 10, 20)"

    algorithm:                       # optional
      lam_cost: 5.0e-1
      ep_tr: 1.0e-3
      autotuner:
        type: RampProximalWeight
        ramp_factor: 1.04
        lam_prox_max: 100.0

    discretizer:                     # optional (integrator / tolerances)
      ode_solver: Dopri8

    solver:                          # optional (convex subproblem solver)
      cvx_solver: QOCO
      solver_args: {abstol: 1.0e-6, reltol: 1.0e-9}

    settings:                        # optional, applied after Problem()
      dev:
        printing: true
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from openscvx.symbolic.expr.control import ControlSpec
from openscvx.symbolic.expr.expr import Expr
from openscvx.symbolic.expr.parameter import ParameterSpec
from openscvx.symbolic.expr.state import StateSpec
from openscvx.symbolic.expr.time import TimeSpec
from openscvx.symbolic.parser._registry import _PARSE_FUNCTIONS
from openscvx.symbolic.parser.parser import ExprParser

# =============================================================================
# Top-level problem spec
# =============================================================================


class ProblemSpec(BaseModel):
    """Validates the entire YAML/JSON problem structure."""

    N: int
    time: TimeSpec
    states: List[StateSpec]
    controls: List[ControlSpec]
    parameters: List[ParameterSpec] = []
    dynamics: Dict[str, Any] = {}
    constraints: List[str] = []
    algorithm: Optional[Dict[str, Any]] = None
    discretizer: Optional[Dict[str, Any]] = None
    solver: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    states_prop: Optional[List[StateSpec]] = None
    dynamics_prop: Optional[Dict[str, Any]] = None
    algebraic_prop: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Public API
# =============================================================================


def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML problem definition and return ``Problem`` constructor kwargs.

    Args:
        path: Path to the YAML file.

    Returns:
        Dict of keyword arguments suitable for ``Problem(**result)``.
    """
    import yaml  # optional dependency

    with open(path) as f:
        data = yaml.safe_load(f)
    return load_dict(data)


def load_json(path: Union[str, Path]) -> dict:
    """Load a JSON problem definition and return ``Problem`` constructor kwargs.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict of keyword arguments suitable for ``Problem(**result)``.
    """
    import json

    with open(path) as f:
        data = json.load(f)
    return load_dict(data)


def load_dict(data: dict) -> dict:
    """Convert a parsed dict into ``Problem`` constructor keyword arguments.

    This is the core routine called by :func:`load_yaml` and
    :func:`load_json`.  It can also be called directly with an
    already-parsed Python dict.

    Args:
        data: Problem definition dictionary (see module docstring for schema).

    Returns:
        Dict with keys ``dynamics``, ``constraints``, ``states``,
        ``controls``, ``N``, ``time``, and optionally ``algorithm``,
        ``discretizer``, ``solver``, ``dynamics_prop``, ``states_prop``,
        ``algebraic_prop``, and ``settings`` (a raw dict to be applied
        via :meth:`Config.apply_dict() <openscvx.config.Config.apply_dict>`
        after construction).
    """
    spec = ProblemSpec.model_validate(data)

    # ---- Build symbolic objects from validated specs -----------------------
    states = [s.to_state() for s in spec.states]
    controls = [c.to_control() for c in spec.controls]
    parameters = [p.to_parameter() for p in spec.parameters]
    time = spec.time.to_time()

    # ---- symbol table ------------------------------------------------------
    symbols: Dict[str, Expr] = {}
    for s in states:
        symbols[s.name] = s
    for c in controls:
        symbols[c.name] = c
    for p in parameters:
        symbols[p.name] = p

    # Validate no symbol names collide with built-in function names
    for name in symbols:
        if name.lower() in _PARSE_FUNCTIONS:
            raise ValueError(
                f"Symbol name {name!r} conflicts with built-in function "
                f"{name.lower()!r}; please rename it"
            )

    parser = ExprParser(symbols)

    # ---- dynamics ----------------------------------------------------------
    dynamics: Dict[str, Expr] = {}
    for state_name, expr_str in spec.dynamics.items():
        dynamics[state_name] = parser.parse(str(expr_str))

    # ---- constraints -------------------------------------------------------
    constraints: list = []
    for constraint_str in spec.constraints:
        constraints.append(parser.parse(str(constraint_str)))

    result: Dict[str, Any] = {
        "dynamics": dynamics,
        "constraints": constraints,
        "states": states,
        "controls": controls,
        "N": spec.N,
        "time": time,
    }

    # ---- algorithm / discretizer / solver (optional) -----------------------
    for key in ("algorithm", "discretizer", "solver"):
        val = getattr(spec, key)
        if val is not None:
            result[key] = val

    # ---- optional: propagation states --------------------------------------
    if spec.states_prop is not None:
        states_prop = [s.to_state() for s in spec.states_prop]
        for s in states_prop:
            symbols[s.name] = s
        result["states_prop"] = states_prop

    # ---- optional: propagation dynamics ------------------------------------
    if spec.dynamics_prop is not None:
        dynamics_prop: Dict[str, Expr] = {}
        for state_name, expr_str in spec.dynamics_prop.items():
            dynamics_prop[state_name] = parser.parse(str(expr_str))
        result["dynamics_prop"] = dynamics_prop

    # ---- optional: algebraic propagation outputs ---------------------------
    if spec.algebraic_prop is not None:
        algebraic_prop: Dict[str, Expr] = {}
        for name, expr_str in spec.algebraic_prop.items():
            algebraic_prop[name] = parser.parse(str(expr_str))
        result["algebraic_prop"] = algebraic_prop

    # ---- optional: settings (applied after Problem construction) -----------
    if spec.settings is not None:
        result["settings"] = spec.settings

    return result
