"""Shared pytest skip marks for optional solver backends.

Importing from this module (not from ``conftest.py``) is the correct pattern
because ``conftest.py`` is loaded via pytest's internal plugin mechanism and is
*not* registered in ``sys.modules`` under the ``tests.conftest`` key, making
direct imports of ``tests.conftest`` unreliable across environments.

Usage in a test module::

    from tests._marks import requires_moreau, _MOREAU_OK

    pytestmark = requires_moreau          # skip whole module
    # or inline:
    if not _MOREAU_OK:
        pytest.skip("moreau not installed or license key not found")
"""

import numpy as np
import pytest
import scipy.sparse as sp


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


_MOREAU_OK: bool = _check_moreau()

requires_moreau = pytest.mark.skipif(
    not _MOREAU_OK,
    reason="moreau not installed or license key not found (pip install openscvx[moreau])",
)
