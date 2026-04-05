from typing import Any, List, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from .expr import Leaf


class Parameter(Leaf):
    """Parameter that can be changed at runtime without recompilation.

    Parameters are symbolic variables with initial values that can be updated
    through the problem's parameter dictionary. They allow for efficient
    parameter sweeps without needing to recompile the optimization problem.

    Example:
        obs_center = ox.Parameter("obs_center", shape=(3,), value=np.array([1.0, 0.0, 0.0]))
        # Later: problem.parameters["obs_center"] = new_value
    """

    def __init__(
        self,
        name: str,
        shape: tuple = (),
        value: Union[float, int, np.ndarray, None] = None,
    ):
        """Initialize a Parameter node.

        Args:
            name: Name identifier for the parameter
            shape: Shape of the parameter (default: scalar)
            value: Initial value for the parameter (required)
        """
        super().__init__(name, shape)
        if value is None:
            raise ValueError(f"Parameter '{name}' requires an initial value")
        self.value = np.asarray(value, dtype=float)

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash Parameter by its shape only (value-invariant).

        Parameters are hashed by shape only, not by value. This allows the same
        compiled solver to be reused across parameter sweeps - only the structure
        matters for compilation, not the actual values.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Parameter")
        hasher.update(str(self._shape).encode())


# =============================================================================
# Pydantic spec for YAML / JSON / dict validation
# =============================================================================


class ParameterSpec(BaseModel):
    """Validates Parameter configuration from YAML/JSON/dict input."""

    name: str
    shape: List[int]
    value: Any

    model_config = ConfigDict(extra="forbid")

    def to_parameter(self) -> "Parameter":
        return Parameter(
            self.name,
            shape=tuple(self.shape),
            value=np.asarray(self.value, dtype=float),
        )
