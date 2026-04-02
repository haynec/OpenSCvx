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
        parameterization: ZOH     # optional: FOH, ZOH, or impulsive (with optional nodes)
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
from typing import Any, Dict, List, Union

import numpy as np

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.expr import Expr, Parameter
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.parser._registry import _PARSE_FUNCTIONS
from openscvx.symbolic.parser.parser import ExprParser

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    # ---- states --------------------------------------------------------
    states: List[State] = []
    for s in data.get("states", []):
        state = State(s["name"], shape=tuple(s["shape"]))
        if "min" in s:
            state.min = np.asarray(s["min"], dtype=float)
        if "max" in s:
            state.max = np.asarray(s["max"], dtype=float)
        if "initial" in s:
            state.initial = _parse_boundary(s["initial"])
        if "final" in s:
            state.final = _parse_boundary(s["final"])
        if "guess" in s:
            state.guess = np.asarray(s["guess"], dtype=float)
        if "scaling_min" in s:
            state.scaling_min = np.asarray(s["scaling_min"], dtype=float)
        if "scaling_max" in s:
            state.scaling_max = np.asarray(s["scaling_max"], dtype=float)
        states.append(state)

    # ---- controls ------------------------------------------------------
    controls: List[Control] = []
    for c in data.get("controls", []):
        param = c.get("parameterization")
        if param is None and c.get("hold") is not None:
            param = c["hold"]
        elif param is None and c.get("impulsive"):
            param = "impulsive"
        nodes = c.get("nodes")
        control = Control(
            c["name"],
            shape=tuple(c["shape"]),
            parameterization=param,
            nodes=nodes,
        )
        if "min" in c:
            control.min = np.asarray(c["min"], dtype=float)
        if "max" in c:
            control.max = np.asarray(c["max"], dtype=float)
        if "guess" in c:
            control.guess = np.asarray(c["guess"], dtype=float)
        if "scaling_min" in c:
            control.scaling_min = np.asarray(c["scaling_min"], dtype=float)
        if "scaling_max" in c:
            control.scaling_max = np.asarray(c["scaling_max"], dtype=float)
        controls.append(control)

    # ---- parameters ----------------------------------------------------
    parameters: List[Parameter] = []
    for p in data.get("parameters", []):
        param = Parameter(
            p["name"],
            shape=tuple(p["shape"]),
            value=np.asarray(p["value"], dtype=float),
        )
        parameters.append(param)

    # ---- time ----------------------------------------------------------
    time_data = data["time"]
    time = Time(
        initial=_parse_time_boundary(time_data["initial"]),
        final=_parse_time_boundary(time_data["final"]),
        min=float(time_data["min"]),
        max=float(time_data["max"]),
        uniform_time_grid=bool(time_data.get("uniform_time_grid", False)),
    )

    # ---- N -------------------------------------------------------------
    N = int(data["N"])

    # ---- symbol table --------------------------------------------------
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

    # ---- dynamics ------------------------------------------------------
    dynamics: Dict[str, Expr] = {}
    for state_name, expr_str in data.get("dynamics", {}).items():
        dynamics[state_name] = parser.parse(str(expr_str))

    # ---- constraints ---------------------------------------------------
    constraints: list = []
    for constraint_str in data.get("constraints", []):
        constraints.append(parser.parse(str(constraint_str)))

    result: Dict[str, Any] = {
        "dynamics": dynamics,
        "constraints": constraints,
        "states": states,
        "controls": controls,
        "N": N,
        "time": time,
    }

    # ---- algorithm / discretizer / solver (optional) ---------------------
    # Pass raw values (dict or instance) through to Problem, which owns
    # resolution via _resolve_algorithm / _resolve_discretizer / _resolve_solver.
    for key in ("algorithm", "discretizer", "solver"):
        if key in data:
            result[key] = data[key]

    # ---- optional: propagation states ----------------------------------
    if "states_prop" in data:
        states_prop: List[State] = []
        for s in data["states_prop"]:
            state = State(s["name"], shape=tuple(s["shape"]))
            if "min" in s:
                state.min = np.asarray(s["min"], dtype=float)
            if "max" in s:
                state.max = np.asarray(s["max"], dtype=float)
            if "initial" in s:
                state.initial = _parse_boundary(s["initial"])
            states_prop.append(state)
            symbols[state.name] = state  # add to symbol table for expressions
        result["states_prop"] = states_prop

    # ---- optional: propagation dynamics --------------------------------
    if "dynamics_prop" in data:
        dynamics_prop: Dict[str, Expr] = {}
        for state_name, expr_str in data["dynamics_prop"].items():
            dynamics_prop[state_name] = parser.parse(str(expr_str))
        result["dynamics_prop"] = dynamics_prop

    # ---- optional: algebraic propagation outputs -----------------------
    if "algebraic_prop" in data:
        algebraic_prop: Dict[str, Expr] = {}
        for name, expr_str in data["algebraic_prop"].items():
            algebraic_prop[name] = parser.parse(str(expr_str))
        result["algebraic_prop"] = algebraic_prop

    # ---- optional: settings (applied after Problem construction) -------
    if "settings" in data:
        result["settings"] = data["settings"]

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_boundary(arr: list) -> list:
    """Convert YAML boundary condition arrays to the State setter format.

    Plain numbers stay as-is (→ fixed).  Two-element lists like
    ``[free, 5.0]`` are converted to tuples ``("free", 5.0)``.

    A bare ``[tag, value]`` pair (e.g. ``[free, 5.0]`` for a shape-[1]
    state) is auto-wrapped so that both ``[free, 5.0]`` and
    ``[[free, 5.0]]`` produce the same result.  A bare string is never
    a valid boundary element, so this detection is unambiguous.
    """
    # Bare [tag, value] pair — wrap so the element-wise loop handles it
    if len(arr) == 2 and isinstance(arr[0], str) and not isinstance(arr[1], list):
        arr = [arr]

    result: list = []
    for item in arr:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
            result.append((str(item[0]), float(item[1])))
        else:
            result.append(item)
    return result


def _parse_time_boundary(val: Any) -> Any:
    """Convert a YAML time boundary to the Time constructor format."""
    if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
        return (str(val[0]), float(val[1]))
    return val
