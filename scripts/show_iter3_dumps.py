#!/usr/bin/env python3
"""Load and inspect SCP iteration-3 CVXPy dump artifacts.

This script reads the files produced by the debug dump workflow:

- ``<label>_iter3_cvxpy_param_dict.npz``
- ``<label>_iter3_cvxpy_solver_data.npz``
- ``<label>_iter3_stats.json``
- optional: ``<label>_fullsolve.json``

and prints compact summaries plus side-by-side diffs for two labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class DumpBundle:
    label: str
    param_npz_path: Path
    solver_npz_path: Path
    stats_json_path: Path
    fullsolve_json_path: Path | None
    param_npz: np.lib.npyio.NpzFile
    solver_npz: np.lib.npyio.NpzFile
    stats: dict
    fullsolve: dict | None


def _find_labels(dump_dir: Path) -> List[str]:
    labels = []
    for p in sorted(dump_dir.glob("*_iter3_stats.json")):
        labels.append(p.name.replace("_iter3_stats.json", ""))
    return labels


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bundle(dump_dir: Path, label: str) -> DumpBundle:
    param_npz_path = dump_dir / f"{label}_iter3_cvxpy_param_dict.npz"
    solver_npz_path = dump_dir / f"{label}_iter3_cvxpy_solver_data.npz"
    stats_json_path = dump_dir / f"{label}_iter3_stats.json"
    fullsolve_json_path = dump_dir / f"{label}_fullsolve.json"
    missing = [p for p in (param_npz_path, solver_npz_path, stats_json_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing dump files for label '{label}': {missing_str}")

    return DumpBundle(
        label=label,
        param_npz_path=param_npz_path,
        solver_npz_path=solver_npz_path,
        stats_json_path=stats_json_path,
        fullsolve_json_path=fullsolve_json_path if fullsolve_json_path.exists() else None,
        param_npz=np.load(param_npz_path),
        solver_npz=np.load(solver_npz_path),
        stats=_load_json(stats_json_path),
        fullsolve=_load_json(fullsolve_json_path) if fullsolve_json_path.exists() else None,
    )


def _numeric_stats(arr: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(arr)
    is_num = np.issubdtype(arr.dtype, np.number)
    out: Dict[str, object] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "numeric": bool(is_num),
    }
    if not is_num:
        return out
    finite = np.isfinite(arr)
    out["nonfinite"] = int((~finite).sum())
    if arr.size:
        out["min"] = float(np.nanmin(arr))
        out["max"] = float(np.nanmax(arr))
        out["maxabs"] = float(np.nanmax(np.abs(arr)))
    else:
        out["min"] = 0.0
        out["max"] = 0.0
        out["maxabs"] = 0.0
    return out


def _print_summary(bundle: DumpBundle) -> None:
    print(f"\n=== {bundle.label} ===")
    summary = bundle.stats.get("summary", {})
    print(f"repo: {summary.get('repo', 'n/a')}")
    print(f"solver: {summary.get('solver', 'n/a')}")
    print("iteration_results:")
    for rec in summary.get("iteration_results", []):
        print(
            "  "
            f"iter={rec.get('iter')} "
            f"converged={rec.get('converged')} "
            f"J_tr={rec.get('scp_J_tr')} "
            f"J_vb={rec.get('scp_J_vb')} "
            f"J_vc={rec.get('scp_J_vc')}"
        )
    exc = summary.get("exception")
    if exc is None:
        print("exception: None")
    else:
        print(f"exception: {exc.get('type')}: {exc.get('message')}")

    if bundle.fullsolve is not None:
        if bundle.fullsolve.get("solve_exception") is None:
            print(
                "fullsolve: converged="
                f"{bundle.fullsolve.get('converged')} "
                f"iters={bundle.fullsolve.get('iters')}"
            )
        else:
            s_exc = bundle.fullsolve.get("solve_exception", {})
            print(f"fullsolve: {s_exc.get('type')}: {s_exc.get('message')}")

    print(f"param keys ({len(bundle.param_npz.files)}): {sorted(bundle.param_npz.files)}")
    print(f"solver data keys ({len(bundle.solver_npz.files)}): {sorted(bundle.solver_npz.files)}")


def _print_sparse_matrix_stats(bundle: DumpBundle, name: str, top_k: int) -> None:
    k_data = f"{name}__data"
    k_row = f"{name}__row"
    k_col = f"{name}__col"
    k_shape = f"{name}__shape"
    if any(k not in bundle.solver_npz.files for k in (k_data, k_row, k_col, k_shape)):
        return

    data = np.asarray(bundle.solver_npz[k_data])
    row = np.asarray(bundle.solver_npz[k_row])
    col = np.asarray(bundle.solver_npz[k_col])
    shape = tuple(int(v) for v in np.asarray(bundle.solver_npz[k_shape]).tolist())
    nonfinite = int((~np.isfinite(data)).sum())
    maxabs = float(np.max(np.abs(data))) if data.size else 0.0
    print(
        f"{name}: shape={shape} nnz={data.size} nonfinite={nonfinite} maxabs={maxabs:.6e}"
    )
    if data.size and top_k > 0:
        idx = np.argsort(np.abs(data))[::-1][: min(top_k, data.size)]
        print(f"  top-{len(idx)} |{name}| entries:")
        for i in idx:
            print(f"    ({int(row[i])}, {int(col[i])}) -> {float(data[i]):.6e}")


def _print_dense_solver_vectors(bundle: DumpBundle) -> None:
    for key in ("b", "c", "h"):
        if key not in bundle.solver_npz.files:
            continue
        st = _numeric_stats(np.asarray(bundle.solver_npz[key]))
        print(
            f"{key}: shape={tuple(st['shape'])} dtype={st['dtype']} "
            f"nonfinite={st.get('nonfinite', 'n/a')} maxabs={st.get('maxabs', 'n/a')}"
        )


def _print_param_highlights(bundle: DumpBundle, keys: List[str]) -> None:
    for key in keys:
        if key not in bundle.param_npz.files:
            continue
        st = _numeric_stats(np.asarray(bundle.param_npz[key]))
        print(
            f"{key}: shape={tuple(st['shape'])} dtype={st['dtype']} "
            f"nonfinite={st.get('nonfinite', 'n/a')} maxabs={st.get('maxabs', 'n/a')}"
        )


def _compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_mismatch": True, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number):
        return {"shape_mismatch": False, "numeric": False}
    d = np.abs(a - b)
    return {
        "shape_mismatch": False,
        "numeric": True,
        "maxabs_a": float(np.max(np.abs(a))) if a.size else 0.0,
        "maxabs_b": float(np.max(np.abs(b))) if b.size else 0.0,
        "maxabs_diff": float(np.max(d)) if d.size else 0.0,
        "l2_diff": float(np.linalg.norm((a - b).reshape(-1))) if a.size else 0.0,
    }


def _print_bundle_diff(a: DumpBundle, b: DumpBundle) -> None:
    print(f"\n=== Diff: {a.label} vs {b.label} ===")
    keys_a = set(a.param_npz.files)
    keys_b = set(b.param_npz.files)
    print(f"param only {a.label}: {sorted(keys_a - keys_b)}")
    print(f"param only {b.label}: {sorted(keys_b - keys_a)}")

    common = sorted(keys_a & keys_b)
    print("common param diffs:")
    for k in common:
        cmp = _compare_arrays(np.asarray(a.param_npz[k]), np.asarray(b.param_npz[k]))
        if cmp.get("shape_mismatch"):
            print(f"  {k}: shape mismatch {cmp['shape_a']} vs {cmp['shape_b']}")
            continue
        if not cmp.get("numeric", True):
            print(f"  {k}: non-numeric array")
            continue
        print(
            f"  {k}: maxabs_a={cmp['maxabs_a']:.6e} "
            f"maxabs_b={cmp['maxabs_b']:.6e} "
            f"maxabs_diff={cmp['maxabs_diff']:.6e} "
            f"l2_diff={cmp['l2_diff']:.6e}"
        )

    s_keys_a = set(a.solver_npz.files)
    s_keys_b = set(b.solver_npz.files)
    print(f"solver data only {a.label}: {sorted(s_keys_a - s_keys_b)}")
    print(f"solver data only {b.label}: {sorted(s_keys_b - s_keys_a)}")


def _print_named_arrays(bundle: DumpBundle, names: List[str], precision: int) -> None:
    if not names:
        return
    print(f"\n=== Raw Arrays: {bundle.label} ===")
    np.set_printoptions(precision=precision, suppress=False, linewidth=140, threshold=1000)
    for name in names:
        if name in bundle.param_npz.files:
            arr = np.asarray(bundle.param_npz[name])
            print(f"\n[{name}] from param_dict npz")
            print(arr)
            continue
        if name in bundle.solver_npz.files:
            arr = np.asarray(bundle.solver_npz[name])
            print(f"\n[{name}] from solver_data npz")
            print(arr)
            continue
        sparse_triplet_keys = {f"{name}__data", f"{name}__row", f"{name}__col", f"{name}__shape"}
        if sparse_triplet_keys.issubset(set(bundle.solver_npz.files)):
            data = np.asarray(bundle.solver_npz[f"{name}__data"])
            row = np.asarray(bundle.solver_npz[f"{name}__row"])
            col = np.asarray(bundle.solver_npz[f"{name}__col"])
            shape = np.asarray(bundle.solver_npz[f"{name}__shape"])
            print(f"\n[{name}] sparse triplet")
            print(f"shape={tuple(shape.tolist())}, nnz={data.size}")
            print("row[:20] =", row[:20])
            print("col[:20] =", col[:20])
            print("data[:20] =", data[:20])
            continue
        print(f"\n[{name}] not found in {bundle.label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SCP iter-3 dump artifacts.")
    parser.add_argument(
        "--dump-dir",
        default=".tmp_iter3_dumps",
        help="Directory containing *_iter3_*.npz/json artifacts (default: .tmp_iter3_dumps).",
    )
    parser.add_argument(
        "--label-a",
        default="commit_506_3d695e76",
        help="First label prefix (default: commit_506_3d695e76).",
    )
    parser.add_argument(
        "--label-b",
        default="commit_mid_e47b6d19",
        help="Second label prefix (default: commit_mid_e47b6d19).",
    )
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="List available labels in dump-dir and exit.",
    )
    parser.add_argument(
        "--show-array",
        action="append",
        default=[],
        help="Show raw array by name (repeatable). Names can be param keys, solver keys, or A/G/P.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of largest-magnitude sparse entries to print for A/G/P (default: 5).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Numpy print precision for --show-array (default: 6).",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir).resolve()
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    labels = _find_labels(dump_dir)
    if args.list_labels:
        print("Available labels:")
        for label in labels:
            print(f"  {label}")
        return

    bundle_a = _load_bundle(dump_dir, args.label_a)
    bundle_b = _load_bundle(dump_dir, args.label_b)

    _print_summary(bundle_a)
    print("param highlights:")
    _print_param_highlights(
        bundle_a,
        keys=["A_d", "B_d", "C_d", "dyn_bias", "x_bar", "u_bar", "x_prop", "x_prop_plus", "E_d"],
    )
    print("solver matrix/vector highlights:")
    for name in ("P", "A", "G"):
        _print_sparse_matrix_stats(bundle_a, name=name, top_k=args.top_k)
    _print_dense_solver_vectors(bundle_a)

    _print_summary(bundle_b)
    print("param highlights:")
    _print_param_highlights(
        bundle_b,
        keys=["A_d", "B_d", "C_d", "dyn_bias", "x_bar", "u_bar", "x_prop", "x_prop_plus", "E_d"],
    )
    print("solver matrix/vector highlights:")
    for name in ("P", "A", "G"):
        _print_sparse_matrix_stats(bundle_b, name=name, top_k=args.top_k)
    _print_dense_solver_vectors(bundle_b)

    _print_bundle_diff(bundle_a, bundle_b)
    _print_named_arrays(bundle_a, args.show_array, args.precision)
    _print_named_arrays(bundle_b, args.show_array, args.precision)


if __name__ == "__main__":
    main()
