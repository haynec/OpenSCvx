"""YAML / JSON problem loader.

Reads a structured problem definition from a YAML or JSON file (or a plain
Python dict) and returns the keyword arguments needed to construct a
`openscvx.problem.Problem`.

!!! tip "YAML Schema & Editor Autocomplete"
    The accepted YAML structure is defined by `ProblemSpec` (a pydantic
    model). You can generate a JSON Schema for editor autocomplete and
    validation with:

        openscvx schema                 # prints to stdout
        openscvx schema -o schema.json  # writes to file

    Then enable it in your editor:

    - **VS Code** (with the ``redhat.vscode-yaml`` extension) --
    add a modeline as the first line of your YAML file:
    ```yaml
    # yaml-language-server: $schema=./schema.json
    ```
    Or configure globally in ``settings.json``:
    ```json
    "yaml.schemas": { "./schema.json": "*.ox.yaml" }
    ```
    - **JetBrains** -- *Languages & Frameworks > Schemas and DTDs >
    JSON Schema Mappings*, point at the generated file and associate
    a file pattern.

    This gives you field-name autocomplete, hover docs, and red squiggles
    on typos or wrong types -- all derived from the pydantic models, so the
    schema always matches the version of openscvx you have installed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from openscvx.algorithms import PenalizedTrustRegionConfig
from openscvx.config import SettingsSpec
from openscvx.discretization import DiscretizerSpec
from openscvx.solvers import SolverSpec
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
    algorithm: Optional[PenalizedTrustRegionConfig] = None
    discretizer: Optional[DiscretizerSpec] = None
    solver: Optional[SolverSpec] = None
    settings: Optional[SettingsSpec] = None
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
    # Convert to dict for Config.apply_dict(); only include sections that
    # were explicitly provided so that unset sections keep their defaults.
    if spec.settings is not None:
        result["settings"] = spec.settings.model_dump(exclude_unset=True)

    return result
