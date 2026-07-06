"""Pytest session configuration for the OpenSCvx test suite.

Optional third-party dependencies are declared at test sites with a plain
per-package marker (``pytest.mark.qpax``, ``pytest.mark.mjx``, ...) whose name
matches the extra in ``pyproject.toml [project.optional-dependencies]``. The
collection hook below derives everything else: the umbrella ``extras`` marker
(so CI can select all optional-dependency tests at once) and a skip when the
package is not installed. Tests under ``tests/e2e/`` are auto-marked ``e2e``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_moreau() -> bool:
    """Return True iff moreau is installed *and* a valid license key is found.

    Runs a one-variable LP so that "installed but unlicensed" is detected here
    rather than surfacing as a confusing ``RuntimeError`` inside a test.
    """
    try:
        from moreau.jax import Cones, Solver

        P = sp.csr_matrix((1, 1))
        A = sp.eye(1, format="csr")
        s = Solver(
            n=1,
            m=1,
            P_row_offsets=P.indptr,
            P_col_indices=P.indices,
            A_row_offsets=A.indptr,
            A_col_indices=A.indices,
            cones=Cones(num_nonneg_cones=1),
            jit=False,
        )
        s.solve(P.data, np.array([1.0]), A.data, np.array([1.0]))
        return True
    except ImportError:
        return False
    except RuntimeError as exc:
        if "No license key found" in str(exc):
            return False
        raise


# Marker name == extra name in pyproject.toml [project.optional-dependencies].
OPTIONAL_DEPS = {
    "mjx": lambda: _installed("mujoco") and _installed("mujoco.mjx"),
    "qpax": lambda: _installed("qpax"),
    "cvxpygen": lambda: _installed("cvxpygen") and _installed("qocogen"),
    "lie": lambda: _installed("jaxlie"),
    "moreau": _check_moreau,
}

_E2E_DIR = Path(__file__).parent / "e2e"


def pytest_collection_modifyitems(config, items):
    available = {dep: probe() for dep, probe in OPTIONAL_DEPS.items()}
    for item in items:
        if _E2E_DIR in item.path.parents:
            item.add_marker(pytest.mark.e2e)
        for dep, ok in available.items():
            if item.get_closest_marker(dep):
                item.add_marker(pytest.mark.extras)
                if not ok:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"requires optional dependency '{dep}' "
                            f"(pip install openscvx[{dep}])"
                        )
                    )
