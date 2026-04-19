import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Cache management
# Core symbolic expressions - flat namespace for most common functions
import openscvx.init as init
import openscvx.symbolic.expr.lie as lie
import openscvx.symbolic.expr.linalg as linalg
import openscvx.symbolic.expr.spatial as spatial
import openscvx.symbolic.expr.stl as stl
import openscvx.symbolic.expr.stljax as stljax
from openscvx.algorithms import (
    AdaptiveProximalWeight,
    AugmentedLagrangian,
    ConstantProximalWeight,
    PenalizedTrustRegion,
    RampProximalWeight,
)
from openscvx.algorithms.optimization_results import OptimizationResults
from openscvx.discretization import (
    DiscretizeLinearizeVectorize,
    LinearizeDiscretize,
    LinearizeDiscretizeSparse,
    VectorizeDiscretizeLinearize,
)
from openscvx.expert import ByofSpec
from openscvx.integrations import DynamicsAdapter, MjxDynamics
from openscvx.loader import load_dict, load_json, load_yaml
from openscvx.problem import Problem
from openscvx.solvers import CVXPyPTRSolver, PTRSolver

# QPAXPTRSolver is exposed lazily via __getattr__ below to keep `import qpax`
# off the hot import path for users who don't install the optional extra.
from openscvx.symbolic.expr import (
    CTCS,
    Abs,
    Acos,
    Add,
    All,
    Any,
    Asin,
    Atan,
    Atan2,
    Bilerp,
    Block,
    Cinterp,
    Concat,
    Cond,
    Constant,
    Constraint,
    Control,
    Cos,
    Diag,
    Div,
    Equality,
    Exp,
    Expr,
    Fixed,
    Free,
    Hstack,
    Index,
    Inequality,
    Inv,
    Leaf,
    Linterp,
    Log,
    LogSumExp,
    MatMul,
    Max,
    Maximize,
    Min,
    Minimize,
    Mul,
    Neg,
    NodalConstraint,
    Parameter,
    Power,
    Sin,
    Sqrt,
    Stack,
    State,
    STMImpulse,
    STMPhysical,
    Sub,
    Sum,
    Tan,
    Variable,
    Vmap,
    Vstack,
    ctcs,
)
from openscvx.symbolic.expr.time import Time
from openscvx.utils.cache import clear_cache, get_cache_dir, get_cache_size

load_results = OptimizationResults.load


def __getattr__(name: str):
    """Lazy export for backends that depend on optional packages."""
    if name == "QPAXPTRSolver":
        from openscvx.solvers.qpax_ptr_solver import QPAXPTRSolver

        return QPAXPTRSolver
    if name == "MoreauPTRSolver":
        from openscvx.solvers.moreau_ptr_solver import MoreauPTRSolver

        return MoreauPTRSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main Trajectory Optimization Entrypoint
    "Problem",
    # Config file loading
    "load_yaml",
    "load_json",
    "load_dict",
    # Results I/O
    "OptimizationResults",
    "load_results",
    # Cache management
    "get_cache_dir",
    "clear_cache",
    "get_cache_size",
    # Time configuration
    "Time",
    # Core base classes
    "Expr",
    "Leaf",
    "Parameter",
    "Variable",
    "State",
    "Control",
    # STM symbolic handles
    "STMPhysical",
    "STMImpulse",
    # Boundary condition helpers
    "Free",
    "Fixed",
    "Minimize",
    "Maximize",
    # Basic arithmetic operations
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Neg",
    "Power",
    "Sum",
    # Array operations
    "Index",
    "Concat",
    "Stack",
    "Hstack",
    "Vstack",
    "Block",
    "Diag",
    "Inv",
    "Constant",
    # Mathematical functions
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Atan2",
    "Sqrt",
    "Abs",
    "Exp",
    "Log",
    "LogSumExp",
    "Max",
    "Min",
    "Linterp",
    "Cinterp",
    "Bilerp",
    # Logical/control flow operations
    "All",
    "Any",
    "Cond",
    # Constraints
    "Constraint",
    "Equality",
    "Inequality",
    "NodalConstraint",
    "CTCS",
    "ctcs",
    # Data parallelism
    "Vmap",
    # Submodules
    "init",
    "stl",
    "stljax",
    "spatial",
    "linalg",
    "lie",
    # Expert mode types
    "ByofSpec",
    # External-backend dynamics adapters
    "DynamicsAdapter",
    "MjxDynamics",
    # Discretization
    "DiscretizeLinearizeVectorize",
    "LinearizeDiscretize",
    "LinearizeDiscretizeSparse",
    "VectorizeDiscretizeLinearize",
    # Convex Solver
    "PTRSolver",
    "CVXPyPTRSolver",
    "QPAXPTRSolver",
    "MoreauPTRSolver",
    # Algorithm & Autotuning
    "PenalizedTrustRegion",
    "AugmentedLagrangian",
    "AdaptiveProximalWeight",
    "ConstantProximalWeight",
    "RampProximalWeight",
]
