"""Assembly of BYOF convex cost terms into CVXPyPTRSolver objectives."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import cvxpy as cp

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.expert.byof import ConvexCostSpec
    from openscvx.lowered.cvxpy_variables import CVXPyVariables

__all__ = ["assemble_byof_convex_costs"]


def assemble_byof_convex_costs(
    specs: list["ConvexCostSpec"],
    ocp_vars: "CVXPyVariables",
    settings: "Config",
    params: dict[str, Any],
) -> cp.Expression:
    """Sum user BYOF convex cost terms into a single CVXPy expression."""
    total = 0
    n_nodes = settings.sim.n
    for spec in specs:
        fn = spec.cost_fn
        n_params = len(inspect.signature(fn).parameters)
        if n_params == 1:
            total += fn(ocp_vars)
        elif n_params == 5:
            nodes = spec.nodes if spec.nodes is not None else range(n_nodes)
            for node in nodes:
                k = node if node >= 0 else n_nodes + node
                x = ocp_vars.x_nonscaled[k]
                u = ocp_vars.u_nonscaled[k]
                total += fn(x, u, k, params, ocp_vars)
        else:
            raise RuntimeError(
                f"byof convex_costs entry has {n_params} parameters; expected 1 "
                "(ocp_vars) or 5 (x, u, node, params, ocp_vars)."
            )
    return total
