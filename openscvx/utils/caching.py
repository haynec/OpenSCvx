import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import jax
import numpy as np
from jax import export

from openscvx.utils.cache import get_cache_dir

if TYPE_CHECKING:
    from openscvx.expert.byof import ByofSpec
    from openscvx.symbolic.problem import SymbolicProblem


def _hash_byof(byof: Optional["ByofSpec"]) -> bytes:
    """Hash BYOF functions by their bytecode and constants.

    Args:
        byof: Optional ByofSpec containing raw JAX functions

    Returns:
        Concatenated bytecode and constants of all functions, or empty bytes if no byof
    """
    if not byof:
        return b""

    codes = []
    for f in byof.get("dynamics", {}).values():
        codes.append(f.__code__.co_code)
        codes.append(repr(f.__code__.co_consts).encode())
    for c in byof.get("nodal_constraints", []):
        codes.append(c["constraint_fn"].__code__.co_code)
        codes.append(repr(c["constraint_fn"].__code__.co_consts).encode())
    for f in byof.get("cross_nodal_constraints", []):
        codes.append(f.__code__.co_code)
        codes.append(repr(f.__code__.co_consts).encode())
    for c in byof.get("ctcs_constraints", []):
        codes.append(c["constraint_fn"].__code__.co_code)
        codes.append(repr(c["constraint_fn"].__code__.co_consts).encode())

    return b"".join(codes)


def get_solver_cache_paths(
    symbolic_problem: "SymbolicProblem",
    dt: float,
    total_time: float,
    cache_dir: Optional[Path] = None,
    byof: Optional["ByofSpec"] = None,
) -> Tuple[Path, Path]:
    """Generate cache file paths using symbolic AST hashing.

    This function computes a hash based on the symbolic structure of the problem,
    which is more stable than hashing lowered JAX code. Two problems with the same
    mathematical structure will produce the same hash, regardless of variable names.

    Args:
        symbolic_problem: The preprocessed SymbolicProblem
        dt: Time step for propagation
        total_time: Total simulation time
        cache_dir: Directory to store cached solvers. If None, uses the default
            cache directory (see :func:`openscvx.get_cache_dir`).
        byof: Optional ByofSpec containing raw JAX functions. If provided,
            function bytecode is included in the hash.

    Returns:
        Tuple of (discretization_solver_path, propagation_solver_path)
    """
    from openscvx.symbolic.hashing import hash_symbolic_problem

    # Get the structural hash of the symbolic problem
    problem_hash = hash_symbolic_problem(symbolic_problem)

    # Include runtime config in the hash
    final_hasher = hashlib.sha256()
    final_hasher.update(problem_hash.encode())
    final_hasher.update(f"dt:{dt}".encode())
    final_hasher.update(f"total_time:{total_time}".encode())

    # Include BYOF function bytecode in the hash
    final_hasher.update(_hash_byof(byof))

    final_hash = final_hasher.hexdigest()[:32]

    solver_dir = cache_dir if cache_dir is not None else get_cache_dir()
    solver_dir.mkdir(parents=True, exist_ok=True)

    dis_solver_file = solver_dir / f"compiled_discretization_solver_{final_hash}.jax"
    prop_solver_file = solver_dir / f"compiled_propagation_solver_{final_hash}.jax"

    return dis_solver_file, prop_solver_file


def load_or_compile_discretization_solver(
    discretization_solver: callable,
    cache_file: Path,
    params: Dict[str, Any],
    n_discretization_nodes: int,
    n_states: int,
    n_controls: int,
    save_compiled: bool = False,
    debug: bool = False,
) -> callable:
    """Load discretization solver from cache or compile and cache it.

    Args:
        discretization_solver: The solver function to compile
        cache_file: Path to cache file
        params: Parameters dictionary
        n_discretization_nodes: Number of discretization nodes
        n_states: Number of state variables
        n_controls: Number of control variables
        save_compiled: Whether to save/load compiled solvers
        debug: Whether in debug mode (skip compilation)

    Returns:
        Compiled discretization solver
    """
    if debug:
        return discretization_solver

    if save_compiled:
        try:
            with open(cache_file, "rb") as f:
                serial_dis = f.read()
            compiled_solver = export.deserialize(serial_dis)
            print("✓ Loaded existing discretization solver")
            return compiled_solver
        except FileNotFoundError:
            print("Compiling discretization solver...")

    else:
        print("Compiling discretization solver (not saving/loading from disk)...")

    # Pass parameters as a single dictionary
    compiled_solver = export.export(jax.jit(discretization_solver))(
        np.ones((n_discretization_nodes, n_states)),
        np.ones((n_discretization_nodes, n_controls)),
        params,
    )

    if save_compiled:
        with open(cache_file, "wb") as f:
            f.write(compiled_solver.serialize())
        print("✓ Discretization solver compiled and saved")

    return compiled_solver


def load_or_compile_propagation_solver(
    propagation_solver: callable,
    cache_file: Path,
    params: Dict[str, Any],
    n_states_prop: int,
    n_controls: int,
    max_tau_len: int,
    save_compiled: bool = False,
) -> callable:
    """Load propagation solver from cache or compile and cache it.

    Args:
        propagation_solver: The solver function to compile
        cache_file: Path to cache file
        params: Parameters dictionary
        n_states_prop: Number of propagation state variables
        n_controls: Number of control variables
        max_tau_len: Maximum tau length for propagation
        save_compiled: Whether to save/load compiled solvers

    Returns:
        Compiled propagation solver
    """
    if save_compiled:
        try:
            with open(cache_file, "rb") as f:
                serial_prop = f.read()
            compiled_solver = export.deserialize(serial_prop)
            print("✓ Loaded existing propagation solver")
            return compiled_solver
        except FileNotFoundError:
            print("Compiling propagation solver...")

    else:
        print("Compiling propagation solver (not saving/loading from disk)...")

    # Pass parameters as a single dictionary
    compiled_solver = export.export(jax.jit(propagation_solver))(
        np.ones(n_states_prop),  # x_0
        (0.0, 0.0),  # time span
        np.ones((1, n_controls)),  # controls_current
        np.ones((1, n_controls)),  # controls_next
        np.ones((1, 1)),  # tau_0
        np.ones((1, 1)).astype("int"),  # segment index
        0,  # idx_s_stop
        np.ones((max_tau_len,)),  # save_time (tau_cur_padded)
        np.ones((max_tau_len,), dtype=bool),  # mask_padded (boolean mask)
        params,  # additional parameters as dict
    )

    if save_compiled:
        with open(cache_file, "wb") as f:
            f.write(compiled_solver.serialize())
        print("✓ Propagation solver compiled and saved")

    return compiled_solver


def prime_propagation_solver(
    propagation_solver: callable, params: Dict[str, Any], settings: Any
) -> None:
    """Prime the propagation solver with a test call to ensure it works.

    Args:
        propagation_solver: Compiled propagation solver
        params: Parameters dictionary
        settings: Settings configuration object
    """
    try:
        x_0 = np.ones(settings.sim.x_prop.initial.shape, dtype=settings.sim.x_prop.initial.dtype)
        tau_grid = (0.0, 1.0)
        controls_current = np.ones((1, settings.sim.u.shape[0]), dtype=settings.sim.u.guess.dtype)
        controls_next = np.ones((1, settings.sim.u.shape[0]), dtype=settings.sim.u.guess.dtype)
        tau_init = np.array([[0.0]], dtype=np.float64)
        node = np.array([[0]], dtype=np.int64)
        idx_s_stop = settings.sim.time_dilation_slice.stop
        save_time = np.ones((settings.prp.max_tau_len,), dtype=np.float64)
        mask_padded = np.ones((settings.prp.max_tau_len,), dtype=bool)
        # Create dummy params dict with same structure
        dummy_params = {
            name: np.ones_like(value) if hasattr(value, "shape") else float(value)
            for name, value in params.items()
        }
        propagation_solver.call(
            x_0,
            tau_grid,
            controls_current,
            controls_next,
            tau_init,
            node,
            idx_s_stop,
            save_time,
            mask_padded,
            dummy_params,
        )
    except Exception as e:
        print(f"[Initialization] Priming propagation_solver.call failed: {e}")
