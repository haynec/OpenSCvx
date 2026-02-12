"""Unit tests for autotuning functions in openscvx.algorithms."""

import numpy as np
import pytest

from openscvx.algorithms.AugmentedLagrangian import AugmentedLagrangian
from openscvx.algorithms.base import (
    AlgorithmState,
    AutotuningBase,
    CandidateIterate,
    DiscretizationResult,
)
from openscvx.algorithms.ConstantProximalWeight import ConstantProximalWeight
from openscvx.algorithms.RampProximalWeight import RampProximalWeight
from openscvx.config import (
    Config,
    ConvexSolverConfig,
    DevConfig,
    DiscretizationConfig,
    PropagationConfig,
    ScpConfig,
    SimConfig,
)
from openscvx.lowered.jax_constraints import (
    LoweredCrossNodeConstraint,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)
