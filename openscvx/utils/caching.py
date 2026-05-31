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
    for f in byof.dynamics.values():
        codes.append(f.__code__.co_code)
        codes.append(repr(f.__code__.co_consts).encode())
    for f in byof.dynamics_discrete.values():
        codes.append(f.__code__.co_code)
        codes.append(repr(f.__code__.co_consts).encode())
    for c in byof.nodal_constraints:
        codes.append(c.constraint_fn.__code__.co_code)
        codes.append(repr(c.constraint_fn.__code__.co_consts).encode())
    for f in byof.cross_nodal_constraints:
        codes.append(f.__code__.co_code)
        codes.append(repr(f.__code__.co_consts).encode())
    for c in byof.ctcs_constraints:
        codes.append(c.constraint_fn.__code__.co_code)
        codes.append(repr(c.constraint_fn.__code__.co_consts).encode())

    return b"".join(codes)


def hash_value_into(hasher: "hashlib._Hash", value: Any) -> None:
    """Fold a scalar / array / ``None`` into ``hasher`` deterministically.

    Shared by the ``_hash_into`` methods of the runtime objects the
    ``solve_batched`` artifact closes over — the convex backend, the
    algorithm/autotuner, the discretizer — and by
    :func:`get_solve_batched_cache_path` for the scaling matrices. Mirrors the
    symbolic ``_hash_into`` protocol's byte-level updates so the same value
    always contributes the same bytes (arrays by shape + dtype + contents,
    everything else by ``repr``).
    """
    if value is None:
        hasher.update(b"None")
    elif isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        hasher.update(f"{arr.shape}:{arr.dtype}:".encode())
        hasher.update(arr.tobytes())
    else:
        hasher.update(repr(value).encode())


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


def get_solve_batched_cache_path(
    symbolic_problem: "SymbolicProblem",
    settings: Any,
    algorithm: Any,
    solver: Any,
    discretizer: Any,
    B: int,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Cache path for the exported :meth:`Problem.solve_batched` artifact.

    The vmapped SCP loop closes over far more than the symbolic problem does, so
    caching it on :func:`get_solver_cache_paths`' key would silently deserialize
    a **stale artifact** whenever any of the extra closed-over state changes — a
    wrong-answer bug, not a cache miss. This assembler extends
    :func:`hash_symbolic_problem` (dynamics, constraints, boundary-condition
    types, parameter shapes) with everything *additionally* baked into the loop:

    * the convex backend class and its ``solver_args`` (via ``solver._hash_into``),
    * the algorithm + autotuner + convergence thresholds + initial penalty
      weights (via ``algorithm._hash_into``),
    * the discretizer scheme (via ``discretizer._hash_into``),
    * the state/control scaling matrices ``inv_S_x`` / ``inv_S_u`` (settings-
      derived, not covered by any subsystem hash),
    * the fixed batch size ``B`` (the artifact is exported at one ``B``), and
    * the JAX version (exported artifacts are not guaranteed to survive an
      incompatible jax bump — the worst failure mode is deserializing one).

    Each runtime object contributes through its own ``_hash_into`` — the same
    two-tier split the symbolic layer uses — so a new field that changes the
    artifact is hashed next to where it lives, never reached into from here.

    Args:
        symbolic_problem: The preprocessed :class:`SymbolicProblem`.
        settings: Problem :class:`~openscvx.config.Config`; supplies the
            scaling matrices.
        algorithm: The SCP algorithm (and its autotuner / weights).
        solver: The convex subproblem backend.
        discretizer: The dynamics discretizer.
        B: Fixed batch size the artifact is exported at.
        cache_dir: Override for the cache directory. ``None`` uses
            :func:`openscvx.get_cache_dir`.

    Returns:
        Path to the ``compiled_solve_batched_<hash>.jax`` artifact.
    """
    from openscvx.symbolic.hashing import hash_symbolic_problem

    hasher = hashlib.sha256()
    hasher.update(hash_symbolic_problem(symbolic_problem).encode())
    solver._hash_into(hasher)
    algorithm._hash_into(hasher)
    discretizer._hash_into(hasher)
    hash_value_into(hasher, settings.sim.inv_S_x)
    hash_value_into(hasher, settings.sim.inv_S_u)
    hasher.update(f"B:{B}".encode())
    hasher.update(f"jax:{jax.__version__}".encode())

    final_hash = hasher.hexdigest()[:32]

    solver_dir = cache_dir if cache_dir is not None else get_cache_dir()
    solver_dir.mkdir(parents=True, exist_ok=True)

    return solver_dir / f"compiled_solve_batched_{final_hash}.jax"


def load_or_compile_discretization_solver(
    discretization_solver: callable,
    cache_file: Path,
    params: Dict[str, Any],
    n_discretization_nodes: int,
    n_states: int,
    n_controls: int,
    save_compiled: bool = False,
    name: str = "continuous",
    debug: bool = False,
) -> callable:
    """Load discretization solver from cache or compile and cache it.

    The no-disk path (``save_compiled=False``, the default) returns a plain
    ``jax.jit`` callable, which has a ``vmap`` batching rule and so composes
    with ``jax.vmap(problem.solve)`` in the JAX-pure solve work. The disk-cached
    path (``save_compiled=True``) returns a ``jax.export`` wrapper that
    serializes to disk but, because ``call_exported`` has no ``vmap`` rule,
    **cannot** be combined with ``jax.vmap(problem.solve)``.

    Args:
        discretization_solver: The solver function to compile
        cache_file: Path to cache file
        params: Parameters dictionary
        n_discretization_nodes: Number of discretization nodes
        n_states: Number of state variables
        n_controls: Number of control variables
        save_compiled: Whether to save/load compiled solvers to/from disk
            (``jax.export``); incompatible with ``jax.vmap(problem.solve)``.
        debug: Whether in debug mode (skip compilation)

    Returns:
        Compiled discretization solver — a ``jax.jit`` callable (no-disk) or a
        ``jax.export`` wrapper (disk-cached).
    """
    if debug:
        return discretization_solver

    if not save_compiled:
        print(f"Compiling {name} discretization solver (not saving/loading from disk)...")
        return jax.jit(discretization_solver)

    try:
        with open(cache_file, "rb") as f:
            serial_dis = f.read()
        compiled_solver = export.deserialize(serial_dis)
        print(f"✓ Loaded existing {name} discretization solver")
        return compiled_solver
    except FileNotFoundError:
        print(f"Compiling {name} discretization solver...")

    # Pass parameters as a single dictionary
    compiled_solver = export.export(jax.jit(discretization_solver))(
        np.ones((n_discretization_nodes, n_states)),
        np.ones((n_discretization_nodes, n_controls)),
        params,
    )

    with open(cache_file, "wb") as f:
        f.write(compiled_solver.serialize())
    print(f"✓ {name} discretization solver compiled and saved")

    return compiled_solver


def load_or_export_solve_batched(
    batched_fn: callable,
    cache_file: Path,
    sample_state: Any,
    sample_params: Dict[str, Any],
) -> Any:
    """Load the exported batched solve from disk, or export and cache it.

    Mirrors :func:`load_or_compile_discretization_solver` but at whole-loop
    granularity: the single artifact is the *entire* vmapped SCP loop exported
    at a fixed batch size, not a per-solver piece. On a cache hit the artifact
    is deserialized with no XLA compile; on a miss the loop is exported against
    the sample batched :class:`~openscvx.algorithms.base.AlgorithmState` and
    parameters, serialized, and returned.

    Unlike the per-solver solvers — whose ``call_exported`` has no ``vmap`` rule
    and so cannot be folded into an outer ``vmap`` — this artifact already has
    the batch baked in, so it is the one exportable form of a batched solve.

    Args:
        batched_fn: The ``jax.vmap``'d ``lax.while_loop`` to export — a pure
            ``(batched_state, params) -> batched_state`` over an exportable
            (QPAX / Moreau) backend.
        cache_file: Path from :func:`get_solve_batched_cache_path`.
        sample_state: A batched ``AlgorithmState`` with the artifact's leading
            axis ``B``; supplies shapes/dtypes for the export trace.
        sample_params: Problem parameters dict; supplies shapes/dtypes.

    Returns:
        A ``jax.export`` wrapper — call it via ``.call(state, params)``.
    """
    try:
        with open(cache_file, "rb") as f:
            wrapper = export.deserialize(f.read())
        print("✓ Loaded existing batched solve")
        return wrapper
    except FileNotFoundError:
        print("Exporting batched solve...")

    wrapper = export.export(jax.jit(batched_fn))(sample_state, sample_params)

    with open(cache_file, "wb") as f:
        f.write(wrapper.serialize())
    print("✓ Batched solve exported and saved")

    return wrapper


def load_or_compile_propagation_solver(
    propagation_solver: callable,
    cache_file: Path,
    params: Dict[str, Any],
    n_states_prop: int,
    n_controls: int,
    max_tau_len: int,
    save_compiled: bool = False,
    debug: bool = False,
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
    if debug:
        return propagation_solver

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
        td_slice = getattr(settings.sim, "time_dilation_slice", None)
        int(td_slice.start) if td_slice is not None else settings.sim.u.shape[0] - 1
        save_time = np.ones((settings.prp.max_tau_len,), dtype=np.float64)
        mask_padded = np.ones((settings.prp.max_tau_len,), dtype=bool)
        # Create dummy params dict with same structure
        dummy_params = {
            name: np.ones_like(value) if hasattr(value, "shape") else float(value)
            for name, value in params.items()
        }
        if hasattr(propagation_solver, "call"):
            propagation_solver.call(
                x_0,
                tau_grid,
                controls_current,
                controls_next,
                tau_init,
                node,
                save_time,
                mask_padded,
                dummy_params,
            )
        else:
            propagation_solver(
                x_0,
                tau_grid,
                controls_current,
                controls_next,
                tau_init,
                node,
                save_time,
                mask_padded,
                dummy_params,
            )
    except Exception as e:
        print(f"[Initialization] Priming propagation_solver.call failed: {e}")
