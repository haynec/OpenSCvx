"""Base class for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
must follow, along with the AlgorithmState dataclass that holds mutable state
during SCP iterations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from openscvx.utils.printing import Column

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver
