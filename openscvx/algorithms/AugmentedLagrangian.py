"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from copy import deepcopy
from typing import TYPE_CHECKING, List

import numpy as np

from openscvx.config import Config
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_acceptance_ratio,
    color_adaptive_state,
    color_J_nonlin,
)

from .base import AutotuningBase

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from .base import AlgorithmState, CandidateIterate
