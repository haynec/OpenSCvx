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
from openscvx.algorithms.autotuning import (
    AugmentedLagrangian,
    ConstantProximalWeight,
    RampProximalWeight,
)
from openscvx.expert import ByofSpec
from openscvx.problem import Problem
from openscvx.symbolic.expr import (
    CTCS,
    Abs,
    Add,
    All,
    Any,
    Bilerp,
    Block,
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
    Sub,
    Sum,
    Tan,
    Variable,
    Vmap,
    Vstack,
    ctcs,
)
from openscvx.symbolic.time import Time
from openscvx.utils.cache import clear_cache, get_cache_dir, get_cache_size

__all__ = [
    # Main Trajectory Optimization Entrypoint
    "Problem",
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
    "Sqrt",
    "Abs",
    "Exp",
    "Log",
    "LogSumExp",
    "Max",
    "Linterp",
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
    "spatial",
    "linalg",
    "lie",
    # Expert mode types
    "ByofSpec",
    # Autotuning
    "AugmentedLagrangian",
    "ConstantProximalWeight",
    "RampProximalWeight",
]
