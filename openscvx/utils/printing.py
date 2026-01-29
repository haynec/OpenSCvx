import queue
import sys
import time
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Callable, Optional

import jax
import numpy as np
from termcolor import colored

if TYPE_CHECKING:
    from openscvx.algorithms import OptimizationResults

warnings.filterwarnings("ignore")


# Define colors for printing
col_main = "blue"
col_pos = "green"
col_neg = "red"


class Verbosity(IntEnum):
    """Verbosity levels for iteration table output."""

    MINIMAL = 1  # Core metrics only (iter, cost, status)
    STANDARD = 2  # + timing, penalty terms
    FULL = 3  # + autotuning diagnostics (J_nonlin, reductions, etc.)


@dataclass
class Column:
    """Specification for a single column in the iteration table."""

    key: str  # Key in emission data dict
    header: str  # Column header text
    width: int  # Column width
    fmt: str  # Format string for values
    color_fn: Optional[Callable[[Any, Any, dict], Optional[str]]] = None
    min_verbosity: int = field(default=Verbosity.MINIMAL)  # Minimum verbosity to show


def color_J_tr(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color J_tr green if within tolerance, red otherwise."""
    if params is None:
        return None
    return col_pos if value <= params.scp.ep_tr else col_neg


def color_J_vb(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color J_vb green if within tolerance, red otherwise."""
    if params is None:
        return None
    return col_pos if value <= params.scp.ep_vb else col_neg


def color_J_vc(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color J_vc green if within tolerance, red otherwise."""
    if params is None:
        return None
    return col_pos if value <= params.scp.ep_vc else col_neg


def color_prob_stat(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color solver status green if optimal, red otherwise."""
    return col_pos if value == "optimal" else col_neg


def color_adaptive_state(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color adaptive state green if acceptable, red otherwise."""
    acceptable_states = ["Accept Constant", "Accept Higher", "Accept Lower", "Initial"]
    return col_pos if value in acceptable_states else col_neg


def color_acceptance_ratio(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color acceptance ratio based on success level.

    <= 0.1: red (unsuccessful)
    0.1 < ratio <= 0.8: somewhat successful (green)
    0.8 < ratio <= 1.5: very successful (blue)
    > 1.5: overly successful (magenta)
    """
    if value <= 0.1:
        return "red"
    elif value <= 0.8:
        return "green"
    elif value <= 1.5:
        return "blue"
    else:
        return "magenta"


def color_J_nonlin(value: Any, params: Any, data: dict) -> Optional[str]:
    """Color J_nonlin green if positive, red otherwise."""
    return "green" if value > 0 else "red"


def build_separator(columns: list[Column]) -> str:
    """Generate separator line matching the total width of active columns."""
    total_width = sum(col.width for col in columns) + 3 * (len(columns) - 1)
    return "─" * total_width


def build_header_format(columns: list[Column]) -> str:
    """Generate header format string from active columns."""
    return " │ ".join(f"{{:^{col.width}}}" for col in columns)


def format_value(col: Column, value: Any, params: Any, data: dict) -> str:
    """Format a single value and apply coloring if needed."""
    # Handle None values
    if value is None:
        value = 0.0

    # Format the value
    formatted = col.fmt.format(value)

    # Pad to column width BEFORE applying color (ANSI codes break alignment)
    formatted = f"{formatted:^{col.width}}"

    # Apply coloring if a color function is defined
    if col.color_fn is not None:
        color = col.color_fn(value, params, data)
        if color is not None:
            formatted = colored(formatted, color)

    return formatted


def header(columns: list[Column]) -> None:
    """Print the table header for the given columns."""
    separator = build_separator(columns)
    header_fmt = build_header_format(columns)

    print(colored(separator))
    print(header_fmt.format(*[col.header for col in columns]))
    print(colored(separator))


def print_row(columns: list[Column], data: dict, params: Any) -> None:
    """Print a single data row for the given columns."""
    values = [format_value(col, data.get(col.key), params, data) for col in columns]
    print(" │ ".join(values))


def get_version() -> str:
    try:
        return version("openscvx")
    except PackageNotFoundError:
        return "0.0.0"


def print_summary_box(lines, title="Summary"):
    """
    Print a centered summary box with the given lines.

    Args:
        lines (list): List of strings to display in the box
        title (str): Title for the box (default: "Summary")
    """
    # Find the longest line (excluding the title which will be handled separately)
    content_lines = lines[1:] if len(lines) > 1 else []
    max_content_width = max(len(line) for line in content_lines) if content_lines else 0
    title_width = len(title)

    # Box width should accommodate both title and content
    box_width = max(max_content_width, title_width) + 4  # Add padding for the box borders

    # Center with respect to the 89-character horizontal lines in io.py
    total_width = 89
    if box_width <= total_width:
        indent = (total_width - box_width) // 2
    else:
        # If box is wider than 89 chars, use a smaller fixed indentation
        indent = 2

    # Print the box with dynamic width and centering
    print(f"\n{' ' * indent}╭{'─' * box_width}╮")
    print(f"{' ' * indent}│ {title:^{box_width - 2}} │")
    print(f"{' ' * indent}├{'─' * box_width}┤")
    for line in content_lines:
        print(f"{' ' * indent}│ {line:<{box_width - 2}} │")
    print(f"{' ' * indent}╰{'─' * box_width}╯\n")


def print_problem_summary(settings: Any, lowered: Any, solver: Any) -> None:
    """
    Print the problem summary box.

    Args:
        settings: Configuration settings containing problem information
        lowered: LoweredProblem from lower_symbolic_problem()
        solver: Initialized ConvexSolver with built problem
    """
    n_nodal_convex = len(lowered.cvxpy_constraints.constraints)
    n_nodal_nonconvex = len(lowered.jax_constraints.nodal)
    n_ctcs = len(lowered.jax_constraints.ctcs)
    n_augmented = settings.sim.n_states - settings.sim.true_state_slice.stop

    # Get solver statistics
    stats = solver.get_stats()
    n_cvx_variables = stats["n_variables"]
    n_cvx_parameters = stats["n_parameters"]
    n_cvx_constraints = stats["n_constraints"]

    # Get JAX backend information
    jax_backend = jax.devices()[0].platform.upper()
    jax_version = jax.__version__

    # Build weights string conditionally
    if isinstance(settings.scp.lam_vc, np.ndarray):
        lam_vc_str = f"λ_vc=matrix({settings.scp.lam_vc.shape})"
    else:
        lam_vc_str = f"λ_vc={settings.scp.lam_vc:4.1f}"
    weights_parts = [
        f"λ_cost={settings.scp.lam_cost:4.1f}",
        f"λ_tr={settings.scp.lam_prox:4.1f}",
        lam_vc_str,
    ]

    # Add λ_vb only if there are nodal nonconvex constraints
    if n_nodal_nonconvex > 0:
        weights_parts.append(f"λ_vb={settings.scp.lam_vb:4.1f}")

    weights_str = ", ".join(weights_parts)

    lines = [
        "Problem Summary",
        (
            f"Dimensions: {settings.sim.n_states} states ({n_augmented} aug),"
            f" {settings.sim.n_controls} controls, {settings.scp.n} nodes"
        ),
        f"Constraints: {n_nodal_convex} conv, {n_nodal_nonconvex} nonconv, {n_ctcs} ctcs",
        (
            f"Subproblem: {n_cvx_variables} vars, {n_cvx_parameters} params,"
            f" {n_cvx_constraints} constraints"
        ),
        f"Weights: {weights_str}",
        f"CVX Solver: {settings.cvx.solver}, Discretization Solver: {settings.dis.solver}",
        f"JAX Backend: {jax_backend} (v{jax_version})",
    ]

    print_summary_box(lines, "Problem Summary")


def print_results_summary(result: "OptimizationResults", timing_post, timing_init, timing_solve):
    """
    Print the results summary box.

    Args:
        result (OptimizationResults): Optimization results object
        timing_post (float): Post-processing time
        timing_init (float): Initialization time
        timing_solve (float): Solve time
    """
    cost = result.get("cost", 0.0)
    ctcs_violation = result.get("ctcs_violation", 0.0)

    # Convert numpy arrays to scalars for formatting
    if hasattr(cost, "item"):
        cost = cost.item()

    # Handle CTCS violation - display as 1D array
    if hasattr(ctcs_violation, "size"):
        if ctcs_violation.size == 1:
            ctcs_violation_str = f"[{ctcs_violation.item():.2e}]"
        else:
            # Display as 1D array
            ctcs_violation_str = f"[{', '.join([f'{v:.2e}' for v in ctcs_violation])}]"
    else:
        ctcs_violation_str = f"[{ctcs_violation:.2e}]"

    # Calculate total computation time
    total_time = (timing_init or 0.0) + (timing_solve or 0.0) + timing_post

    lines = [
        "Results Summary",
        f"Cost: {cost:.6f}",
        f"CTCS Constraint Violation: {ctcs_violation_str}",
        f"Preprocessing Time: {timing_init or 0.0:.3f}s",
        f"Main Solve Time: {timing_solve or 0.0:.3f}s",
        f"Post-processing Time: {timing_post:.3f}s",
        f"Total Computation Time: {total_time:.3f}s",
    ]

    print_summary_box(lines, "Results Summary")


def intro():
    # Silence syntax warnings
    warnings.filterwarnings("ignore")
    # fmt: off
    ascii_art = rf"""

                            ____                    _____  _____
                           / __ \                  / ____|/ ____|
                          | |  | |_ __   ___ _ __ | (___ | |  __   ____  __
                          | |  | | '_ \ / _ \ '_ \ \___ \| |  \ \ / /\ \/ /
                          | |__| | |_) |  __/ | | |____) | |___\ V /  >  <
                           \____/| .__/ \___|_| |_|_____/ \_____\_/  /_/\_\
                                 | |
                                 |_|
─────────────────────────────────────────────────────────────────────────────────────────────────────────
                                Author: Chris Hayner and Griffin Norris
                                    Autonomous Controls Laboratory
                                       University of Washington
                                         Version: {get_version()}
─────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
    # fmt: on
    print(ascii_art)


def intermediate(print_queue: queue.Queue, params: Any, columns: list[Column]) -> None:
    """Process and print iteration data from the queue.

    This function runs in a loop, reading data from the print queue and
    displaying formatted iteration rows.

    Args:
        print_queue: Queue containing iteration data dicts
        params: Settings object (used for color threshold comparisons)
        columns: List of Column specs defining the table structure
    """
    hz = 30.0
    separator = build_separator(columns)

    while True:
        t_start = time.time()
        try:
            data = print_queue.get(timeout=1.0 / hz)

            # Truncate prob_stat if it's a longer string (e.g., "infeasible" -> "i")
            prob_stat = data.get("prob_stat", "")
            if prob_stat.startswith("inf"):
                data["prob_stat"] = "i"

            # Remove bottom separator and header (2 lines)
            if data["iter"] != 1:
                sys.stdout.write("\x1b[1A\x1b[2K\x1b[1A\x1b[2K")

            # Print the data row
            print_row(columns, data, params)

            # Print separator and header (footer() will add closing separator)
            print(colored(separator))
            header_fmt = build_header_format(columns)
            print(header_fmt.format(*[col.header for col in columns]))

        except queue.Empty:
            pass
        time.sleep(max(0.0, 1.0 / hz - (time.time() - t_start)))


def footer(columns: list[Column]) -> None:
    """Print the table footer.

    Args:
        columns: List of Column specs defining the table structure
    """
    separator = build_separator(columns)
    print(colored(separator))
