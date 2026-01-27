import queue
import sys
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import jax
import numpy as np
from termcolor import colored

from openscvx.algorithms import OptimizationResults

warnings.filterwarnings("ignore")


# Define colors for printing
col_main = "blue"
col_pos = "green"
col_neg = "red"


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
        f"λ_tr={settings.scp.w_tr:4.1f}",
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


def print_results_summary(result: OptimizationResults, timing_post, timing_init, timing_solve):
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


def header():
    print(
        colored(
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────"
        )
    )
    header_format = (
        "{:^4} │ {:^13} │ {:^14} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ "
        "{:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ "
        "{:^14} │ {:^16}"
    )
    print(
        header_format.format(
            "Iter",
            "Dis Time (ms)",
            "Solve Time (ms)",
            "J_total",
            "J_tr",
            "J_vb",
            "J_vc",
            "J_nonlin",
            "J_lin",
            "pred_red",
            "act_red",
            "acc_ratio",
            "w_tr",
            "Cost",
            "Solver Status",
            "Adaptive State",
        )
    )
    print(
        colored(
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────"
        )
    )


def intermediate(print_queue, params):
    hz = 30.0
    while True:
        t_start = time.time()
        try:
            data = print_queue.get(timeout=1.0 / hz)
            # remove bottom labels and line
            if data["iter"] != 1:
                sys.stdout.write("\x1b[1A\x1b[2K\x1b[1A\x1b[2K")
            if data["prob_stat"][3] == "f":
                # Only show the first element of the string
                data["prob_stat"] = data["prob_stat"][0]

            iter_colored = colored("{:4d}".format(data["iter"]))
            J_tot_colored = colored("{: .1e}".format(data["J_total"]))
            J_tr_colored = colored(
                "{: .1e}".format(data["J_tr"]),
                col_pos if data["J_tr"] <= params.scp.ep_tr else col_neg,
            )
            J_vb_colored = colored(
                "{: .1e}".format(data["J_vb"]),
                col_pos if data["J_vb"] <= params.scp.ep_vb else col_neg,
            )
            J_vc_colored = colored(
                "{: .1e}".format(data["J_vc"]),
                col_pos if data["J_vc"] <= params.scp.ep_vc else col_neg,
            )
            J_nonlin_colored = colored("{: .1e}".format(data.get("J_nonlin", 0.0)))
            J_lin_colored = colored("{: .1e}".format(data.get("J_lin", 0.0)))
            pred_reduction_colored = colored("{: .1e}".format(data.get("pred_reduction", 0.0)))
            actual_reduction_colored = colored("{: .1e}".format(data.get("actual_reduction", 0.0)))
            acceptance_ratio_colored = colored("{: .2e}".format(data.get("acceptance_ratio", 0.0)))
            w_tr_colored = colored("{: .1e}".format(data.get("w_tr", 0.0)))
            cost_colored = colored("{: .1e}".format(data["cost"]))
            prob_stat_colored = colored(
                data["prob_stat"], col_pos if data["prob_stat"] == "optimal" else col_neg
            )
            adaptive_state = data.get("adaptive_state", "")
            is_acceptable = adaptive_state in [
                "Accept Constant",
                "Accept Higher",
                "Accept Lower",
                "Initial",
            ]
            adaptive_state_colored = colored(
                data.get("adaptive_state", "N/A"),
                col_pos if is_acceptable else col_neg,
            )

            row_format = (
                "{:^4} │     {:^6.2f}    │      {:^6.2F}     │ {:^8} │ "
                "{:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ "
                "{:^8} │ {:^8} │ {:^8} │ {:^14} │ {:^16}"
            )
            print(
                row_format.format(
                    iter_colored,
                    data["dis_time"],
                    data["subprop_time"],
                    J_tot_colored,
                    J_tr_colored,
                    J_vb_colored,
                    J_vc_colored,
                    J_nonlin_colored,
                    J_lin_colored,
                    pred_reduction_colored,
                    actual_reduction_colored,
                    acceptance_ratio_colored,
                    w_tr_colored,
                    cost_colored,
                    prob_stat_colored,
                    adaptive_state_colored,
                )
            )

            print(
                colored(
                    "─────────────────────────────────────────────────────────────────────────────────────────────────────────"
                )
            )
            header_format = (
                "{:^4} │ {:^13} │ {:^14} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ "
                "{:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ {:^8} │ "
                "{:^14} │ {:^16}"
            )
            print(
                header_format.format(
                    "Iter",
                    "Dis Time (ms)",
                    "Solve Time (ms)",
                    "J_total",
                    "J_tr",
                    "J_vb",
                    "J_vc",
                    "J_nonlin",
                    "J_lin",
                    "pred_red",
                    "act_red",
                    "acc_ratio",
                    "w_tr",
                    "Cost",
                    "Solver Status",
                    "Adaptive State",
                )
            )
        except queue.Empty:
            pass
        time.sleep(max(0.0, 1.0 / hz - (time.time() - t_start)))


def footer():
    print(
        colored(
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────"
        )
    )
