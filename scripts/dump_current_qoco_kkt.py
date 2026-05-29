"""Dump QOCO canonical data and static KKT matrices for a 3DoF PDG run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path

import numpy as np
from scipy import sparse


def _stub_plotting_modules() -> None:
    plotting_viser = types.ModuleType("examples.plotting_viser")
    plotting_viser.create_pdg_animated_plotting_server = lambda *args, **kwargs: None
    plotting_viser.create_scp_animated_plotting_server = lambda *args, **kwargs: None
    sys.modules["examples.plotting_viser"] = plotting_viser

    plotting = types.ModuleType("openscvx.plotting")
    for name in ["plot_controls", "plot_projections_2d", "plot_states", "plot_vector_norm"]:
        setattr(plotting, name, lambda *args, **kwargs: None)
    sys.modules["openscvx.plotting"] = plotting


def _stub_optional_codegen_modules() -> None:
    cvxpygen = types.ModuleType("cvxpygen")
    cvxpygen.cpg = types.ModuleType("cvxpygen.cpg")
    sys.modules["cvxpygen"] = cvxpygen
    sys.modules["cvxpygen.cpg"] = cvxpygen.cpg

    pdaqp = types.ModuleType("pdaqp")
    pdaqp.MPQP = object
    sys.modules["pdaqp"] = pdaqp

    juliacall = types.ModuleType("juliacall")
    juliacall.Main = object()
    sys.modules["juliacall"] = juliacall


def _load_example(repo: Path, force_float64: bool):
    _stub_optional_codegen_modules()
    for name in list(sys.modules):
        if name == "openscvx" or name.startswith("openscvx."):
            del sys.modules[name]
    _stub_plotting_modules()
    sys.path.insert(0, str(repo))

    if force_float64:
        import openscvx as ox

        original_problem = ox.Problem

        def problem_float64(*args, **kwargs):
            kwargs["float_dtype"] = "float64"
            return original_problem(*args, **kwargs)

        ox.Problem = problem_float64

    example_path = repo / "examples" / "rocket" / "3DoF_pdg.py"
    spec = importlib.util.spec_from_file_location("pdg_for_qoco_kkt", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _matrix(data: dict, key: str) -> sparse.csc_matrix:
    value = data.get(key)
    if value is None:
        return sparse.csc_matrix((0, 0))
    return sparse.csc_matrix(value)


def _dense(data: dict, key: str) -> np.ndarray:
    value = data.get(key)
    if value is None:
        return np.zeros(0)
    return np.asarray(value, dtype=np.float64)


def _stats_sparse(mat: sparse.spmatrix) -> dict:
    data = mat.data
    abs_data = np.abs(data)
    nonzero = abs_data[abs_data > 0]
    return {
        "shape": list(mat.shape),
        "nnz": int(mat.nnz),
        "max_abs": float(abs_data.max()) if abs_data.size else 0.0,
        "min_abs_nonzero": float(nonzero.min()) if nonzero.size else 0.0,
        "spread_nonzero": float(nonzero.max() / nonzero.min()) if nonzero.size else 0.0,
        "nonfinite": int((~np.isfinite(data)).sum()) if data.size else 0,
    }


def _stats_dense(vec: np.ndarray) -> dict:
    arr = np.asarray(vec)
    abs_arr = np.abs(arr.ravel())
    nonzero = abs_arr[abs_arr > 0]
    return {
        "shape": list(arr.shape),
        "max_abs": float(abs_arr.max()) if abs_arr.size else 0.0,
        "min_abs_nonzero": float(nonzero.min()) if nonzero.size else 0.0,
        "spread_nonzero": float(nonzero.max() / nonzero.min()) if nonzero.size else 0.0,
        "nonfinite": int((~np.isfinite(arr)).sum()) if arr.size else 0,
    }


def _condition_estimates(mat: sparse.spmatrix) -> dict:
    dense = mat.toarray()
    if dense.size == 0:
        return {"sigma_max": 0.0, "sigma_min_nonzero": 0.0, "rank": 0, "cond_nonzero": 0.0}
    singular_values = np.linalg.svd(dense, compute_uv=False)
    tol = max(dense.shape) * np.finfo(float).eps * singular_values[0]
    nonzero = singular_values[singular_values > tol]
    if nonzero.size == 0:
        return {
            "sigma_max": float(singular_values[0]),
            "sigma_min_nonzero": 0.0,
            "rank": 0,
            "tol": float(tol),
            "cond_nonzero": float("inf"),
        }
    return {
        "sigma_max": float(singular_values[0]),
        "sigma_min_nonzero": float(nonzero[-1]),
        "rank": int(nonzero.size),
        "tol": float(tol),
        "cond_nonzero": float(singular_values[0] / nonzero[-1]),
    }


def _static_kkt(P: sparse.spmatrix, A: sparse.spmatrix, G: sparse.spmatrix) -> sparse.csc_matrix:
    n = P.shape[0]
    p = A.shape[0]
    m = G.shape[0]
    z_pp = sparse.csc_matrix((p, p))
    z_pm = sparse.csc_matrix((p, m))
    z_mm = sparse.csc_matrix((m, m))
    return sparse.bmat(
        [
            [P, A.T, G.T],
            [A, z_pp, z_pm],
            [G, z_pm.T, z_mm],
        ],
        format="csc",
    )


def _dump_subproblem(problem, solver_name: str, outdir: Path, solve_call: int) -> dict:
    data, chain, inverse = problem.get_problem_data(solver=solver_name)
    del chain, inverse

    P = _matrix(data, "P")
    A = _matrix(data, "A")
    G = _matrix(data, "G")
    b = _dense(data, "b")
    c = _dense(data, "c")
    h = _dense(data, "h")
    kkt = _static_kkt(P, A, G)
    rhs = np.concatenate((-c, b, h))

    prefix = outdir / f"scp_{solve_call:03d}"
    sparse.save_npz(f"{prefix}_P.npz", P)
    sparse.save_npz(f"{prefix}_A.npz", A)
    sparse.save_npz(f"{prefix}_G.npz", G)
    sparse.save_npz(f"{prefix}_K_static.npz", kkt)
    np.savez_compressed(f"{prefix}_vectors.npz", b=b, c=c, h=h, rhs_static=rhs)

    return {
        "solve_call": solve_call,
        "P": _stats_sparse(P),
        "A": _stats_sparse(A),
        "G": _stats_sparse(G),
        "K_static": _stats_sparse(kkt),
        "b": _stats_dense(b),
        "c": _stats_dense(c),
        "h": _stats_dense(h),
        "rhs_static": _stats_dense(rhs),
        "cond_A": _condition_estimates(A),
        "cond_AG": _condition_estimates(sparse.vstack([A, G], format="csc")),
        "cond_K_static": _condition_estimates(kkt),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--outdir", default=".tmp_qoco_kkt_dumps/current_float64")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--force-float64", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    repo = Path(args.repo).resolve()
    outdir_arg = Path(args.outdir)
    outdir = outdir_arg if outdir_arg.is_absolute() else cwd / outdir_arg
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    module = _load_example(repo, args.force_float64)
    problem = module.problem
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False
    if hasattr(problem.settings, "sim"):
        problem.settings.sim.save_compiled = False

    problem.initialize()

    solver = problem._solver
    cvxpy_problem = solver._problem
    solver_name = solver.cvx_solver
    original_solve = solver._solve_fn
    solve_calls = {"n": 0}
    subproblems = []

    def wrapped_solve():
        solve_calls["n"] += 1
        subproblems.append(_dump_subproblem(cvxpy_problem, solver_name, outdir, solve_calls["n"]))
        return original_solve()

    solver._solve_fn = wrapped_solve

    summary = {
        "repo": str(repo),
        "example": str(repo / "examples" / "rocket" / "3DoF_pdg.py"),
        "outdir": str(outdir),
        "solver": solver_name,
        "float_dtype": getattr(problem, "_float_dtype", None),
        "force_float64": args.force_float64,
        "max_steps": args.max_steps,
        "iteration_results": [],
        "exception": None,
        "subproblems": subproblems,
    }

    try:
        for step in range(1, args.max_steps + 1):
            result = problem.step()
            summary["iteration_results"].append(
                {
                    "step": step,
                    "converged": bool(result.get("converged", False)),
                    "scp_k": int(result.get("scp_k", -1)),
                    "scp_J_tr": float(result.get("scp_J_tr", np.nan)),
                    "scp_J_vb": float(result.get("scp_J_vb", np.nan)),
                    "scp_J_vc": float(result.get("scp_J_vc", np.nan)),
                }
            )
            if result.get("converged", False):
                break
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
