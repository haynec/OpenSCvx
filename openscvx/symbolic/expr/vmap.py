"""Vmap expression for data-parallel operations.

This module provides symbolic support for JAX's vmap (vectorized map) operation,
enabling efficient data-parallel computations over batched data within the
symbolic expression framework.

Vmap supports multiple modes based on the type of `batch`:

- **Constant/array**: Values baked into the compiled function at trace time, equivalent to
    closure-captured values in BYOF. Use for static data.
- **Parameter**: Values looked up from params dict at runtime, allowing updates between SCP
    iterations. Use for data that may change.
- **State**: Values extracted from the unified state vector at runtime, enabling vectorized
    operations over state elements (e.g., multi-agent).
- **Control**: Values extracted from the unified control vector at runtime, enabling vectorized
    operations over control elements.

Vmap also supports batching over multiple arguments by passing a list of
batch sources. Each batch source is mapped to a corresponding lambda argument.

See the :class:`Vmap` class documentation for usage examples.
"""

import uuid
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, Union

import numpy as np

from .control import Control
from .expr import Constant, Expr, Leaf
from .state import State

if TYPE_CHECKING:
    from .expr import Parameter

# Type alias for a single batch source
BatchSource = Union[np.ndarray, Constant, "Parameter", State, Control]


class _Placeholder(Leaf):
    """Placeholder variable for use inside Vmap expressions.

    Placeholder is a symbolic leaf node that represents a single element from
    a batched array during vmap execution. It is created automatically by
    Vmap.__init__ and should not be instantiated directly by users.

    During lowering, the Vmap visitor injects the current batch element into
    the params dict, and Placeholder retrieves it via params lookup.

    Attributes:
        name (str): Unique identifier for params lookup (auto-generated)
        _shape (tuple): Shape of a single element from the batched data

    Note:
        Users should not create Placeholder instances directly. Instead, use
        ox.Vmap with a lambda that receives the placeholder as an argument.
    """

    def __init__(self, shape: Tuple[int, ...]):
        """Initialize a Placeholder.

        Args:
            shape: Shape of a single element from the batched data.
                   For example, if vmapping over data with shape (10, 3),
                   the placeholder shape would be (3,).
        """
        # Generate unique name for params lookup
        name = f"_vmap_placeholder_{uuid.uuid4().hex[:8]}"
        super().__init__(name, shape)

    def _hash_into(self, hasher):
        """Hash Placeholder by its unique name.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Placeholder")
        hasher.update(self.name.encode())


class Vmap(Expr):
    """Vectorized map over batched data in symbolic expressions.

    Vmap enables data-parallel operations by applying a symbolic expression
    to each element of a batched array (or multiple arrays). This is the
    symbolic equivalent of JAX's jax.vmap, allowing efficient vectorized
    computation without explicit loops.

    The expression is defined via a lambda that receives one or more Placeholder
    arguments, each representing a single element from the corresponding batch.
    During lowering, this becomes a jax.vmap call.

    The behavior depends on the type of each `batch` element:

    - **numpy array or Constant**: Data is baked into the compiled function
      at trace time, equivalent to closure-captured values in BYOF.
    - **Parameter**: Data is looked up from the params dict at runtime,
      allowing the same compiled code to be reused with different values.
    - **State**: Data is extracted from the unified state vector at runtime,
      enabling vectorized operations over state elements (e.g., multi-agent).
    - **Control**: Data is extracted from the unified control vector at runtime,
      enabling vectorized operations over control elements.

    Attributes:
        _batches (tuple): Tuple of data sources (Constant, Parameter, State, or Control)
        _axis (int): The axis to vmap over (default: 0)
        _placeholders (tuple): Tuple of placeholders used in the expression
        _child (Expr): The expression tree built from the user's lambda
        _is_parameter (tuple): Tuple of bools indicating which batches are Parameters
        _is_state (tuple): Tuple of bools indicating which batches are States
        _is_control (tuple): Tuple of bools indicating which batches are Controls

    Example:
        Compute distances to multiple reference points (baked-in)::

            position = ox.State("position", shape=(3,))
            init_poses = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            distances = ox.Vmap(
                lambda pose: ox.linalg.Norm(position - pose),
                batch=init_poses
            )
            # distances has shape (3,)

        With runtime-updateable Parameter::

            refs = ox.Parameter("refs", shape=(10, 3), value=init_poses)
            dist_state = ox.State("dist_state", shape=(10,))

            dynamics["dist_state"] = ox.Vmap(
                lambda pose: ox.linalg.Norm(position - pose),
                batch=refs
            )

            # Later, change the parameter value without recompiling:
            problem.parameters["refs"] = new_poses

        With multiple batch arguments::

            obs_centers = ox.Parameter("obs_centers", shape=(100, 3))
            obs_radii = ox.Parameter("obs_radii", shape=(100,))

            constraints = ox.Vmap(
                lambda center, radius: radius <= ox.linalg.Norm(position - center),
                batch=[obs_centers, obs_radii]
            )
            # constraints has shape (100,)

        With State batching (multi-agent)::

            # State representing n_agents positions, each 3D
            agent_positions = ox.State("agent_positions", shape=(n_agents, 3))

            # Apply constraint to each agent's position
            constraints = ox.Vmap(
                lambda pos: ox.linalg.Norm(pos) <= max_distance,
                batch=agent_positions
            )
            # constraints has shape (n_agents,)

        With Control batching::

            # Control representing n_thrusters, each scalar
            thrusters = ox.Control("thrusters", shape=(n_thrusters,))

            # Apply constraint to each thruster
            constraints = ox.Vmap(
                lambda t: t <= max_thrust,
                batch=thrusters
            )
            # constraints has shape (n_thrusters,)

        Batching over a non-default axis::

            # Data shaped (features, n_samples) - batch over axis 1
            samples = ox.Parameter("samples", shape=(3, n_samples), value=data)
            results = ox.Vmap(
                lambda sample: ox.linalg.Norm(sample),
                batch=samples,
                axis=1  # batch over samples, not features
            )
            # results has shape (n_samples,)

    Note:
        - For static data that won't change, pass a numpy array or Constant
          to get closure-equivalent behavior (numerically identical to BYOF).
        - For data that needs to be updated between iterations, use Parameter.
        - For vectorized operations over state/control elements, pass State/Control.
        - When using multiple batches, all must have the same size along the
          vmap axis.

    !!! warning "Prefer Constants over Parameters"
        **Use a raw numpy array or Constant unless you specifically need to
        update the vmap data between solves without recompiling.**

        Using a Parameter (runtime lookup) may produce **different numerical
        results** compared to using a Constant (baked-in), even when the
        underlying data is identical. This can manifest as:

        - Different SCP iteration counts
        - Different convergence behavior
        - In unlucky cases, convergence to a different local solution

        This is likely due to JAX/XLA trace and compilation differences between
        the two code paths. When data is baked in, JAX sees concrete values at
        trace time. When data is looked up from a params dict at runtime, JAX
        traces through the dictionary access, potentially producing different
        XLA compilation or floating-point operation ordering.
    """

    def __init__(
        self,
        fn: Callable[..., Expr],
        batch: Union[BatchSource, Sequence[BatchSource]],
        axis: int = 0,
    ):
        """Initialize a Vmap expression.

        Args:
            fn: A callable (typically a lambda) that takes one or more Placeholder
                arguments and returns a symbolic expression. Each Placeholder
                represents a single element from the corresponding batched data.
            batch: The batched data to vmap over. Can be:
                  - A single batch source (numpy array, Constant, Parameter, State, or Control)
                  - A list/tuple of batch sources for multi-argument vmapping
                  Each batch source can be:
                  - numpy array: baked into compiled function (closure-equivalent)
                  - Constant: baked into compiled function (closure-equivalent)
                  - Parameter: looked up from params dict at runtime
                  - State: extracted from unified state vector at runtime
                  - Control: extracted from unified control vector at runtime
            axis: The axis to vmap over. Default is 0 (first axis).
                  Applied to all batch sources.

        Example:
            Single batch (baked-in data)::

                ox.Vmap(lambda x: ox.linalg.Norm(x), batch=points)

            Single batch with Parameter::

                refs = ox.Parameter("refs", shape=(10, 3), value=points)
                ox.Vmap(lambda ref: ox.linalg.Norm(position - ref), batch=refs)

            Multiple batches::

                centers = ox.Parameter("centers", shape=(100, 3))
                radii = ox.Parameter("radii", shape=(100,))
                ox.Vmap(
                    lambda c, r: r <= ox.linalg.Norm(position - c),
                    batch=[centers, radii]
                )

            State batching (multi-agent)::

                agent_positions = ox.State("positions", shape=(n_agents, 3))
                ox.Vmap(lambda pos: g(pos), batch=agent_positions)

            Non-default axis::

                # Batch over axis 1 instead of axis 0
                data = ox.Parameter("data", shape=(3, n_samples))
                ox.Vmap(lambda x: f(x), batch=data, axis=1)
        """
        from .expr import Parameter

        # Normalize input: convert single batch to list, then process each
        if isinstance(batch, (list, tuple)) and not isinstance(batch, np.ndarray):
            batch_list = list(batch)
        else:
            batch_list = [batch]

        # Normalize each batch source: wrap raw arrays in Constant
        # Keep State/Control/Parameter as-is
        normalized_batches = []
        is_parameter_flags = []
        is_state_flags = []
        is_control_flags = []
        for b in batch_list:
            if isinstance(b, np.ndarray):
                b = Constant(b)
            elif not isinstance(b, (Constant, Parameter, State, Control)):
                # Try to convert to array then Constant
                b = Constant(np.asarray(b))
            normalized_batches.append(b)
            is_parameter_flags.append(isinstance(b, Parameter))
            is_state_flags.append(isinstance(b, State))
            is_control_flags.append(isinstance(b, Control))

        self._batches = tuple(normalized_batches)
        self._axis = axis
        self._is_parameter = tuple(is_parameter_flags)
        self._is_state = tuple(is_state_flags)
        self._is_control = tuple(is_control_flags)

        # Get batch size from first batch and validate all batches match
        first_shape = Vmap._get_batch_shape(
            self._batches[0],
            self._is_parameter[0],
            self._is_state[0],
            self._is_control[0],
        )
        if axis < 0 or axis >= len(first_shape):
            raise ValueError(f"Vmap axis {axis} out of bounds for data with shape {first_shape}")
        batch_size = first_shape[axis]

        # Validate all batches have the same size along the vmap axis
        for i, (b, is_param, is_state, is_control) in enumerate(
            zip(self._batches, self._is_parameter, self._is_state, self._is_control)
        ):
            shape = Vmap._get_batch_shape(b, is_param, is_state, is_control)
            if axis >= len(shape):
                raise ValueError(f"Vmap axis {axis} out of bounds for batch {i} with shape {shape}")
            if shape[axis] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: batch 0 has size {batch_size} along axis {axis}, "
                    f"but batch {i} has size {shape[axis]}"
                )

        # Create placeholders for each batch
        placeholders = []
        for b, is_param, is_state, is_control in zip(
            self._batches, self._is_parameter, self._is_state, self._is_control
        ):
            shape = Vmap._get_batch_shape(b, is_param, is_state, is_control)
            # Compute per-element shape by removing the vmap axis
            per_elem_shape = tuple(s for i, s in enumerate(shape) if i != axis)
            placeholders.append(_Placeholder(shape=per_elem_shape))

        self._placeholders = tuple(placeholders)

        # Build expression tree by calling fn with all placeholders
        if len(self._placeholders) == 1:
            self._child = fn(self._placeholders[0])
        else:
            self._child = fn(*self._placeholders)

    @property
    def batches(self) -> Tuple[Union[Constant, "Parameter", State, Control], ...]:
        """Tuple of batched data sources being vmapped over."""
        return self._batches

    @property
    def batch(self) -> Union[Constant, "Parameter", State, Control]:
        """The first batched data source (for single-batch backward compatibility)."""
        return self._batches[0]

    @property
    def axis(self) -> int:
        """The axis being vmapped over."""
        return self._axis

    @property
    def placeholders(self) -> Tuple[_Placeholder, ...]:
        """Tuple of placeholders used in the inner expression."""
        return self._placeholders

    @property
    def placeholder(self) -> _Placeholder:
        """The first placeholder (for single-batch backward compatibility)."""
        return self._placeholders[0]

    @property
    def is_parameter(self) -> Tuple[bool, ...]:
        """Tuple of bools indicating which batches are Parameters (runtime lookup)."""
        return self._is_parameter

    @property
    def is_state(self) -> Tuple[bool, ...]:
        """Tuple of bools indicating which batches are States (state vector lookup)."""
        return self._is_state

    @property
    def is_control(self) -> Tuple[bool, ...]:
        """Tuple of bools indicating which batches are Controls (control vector lookup)."""
        return self._is_control

    @property
    def num_batches(self) -> int:
        """Number of batch arguments."""
        return len(self._batches)

    @staticmethod
    def _get_batch_shape(
        batch: Union[Constant, "Parameter", State, Control],
        is_param: bool,
        is_state: bool,
        is_control: bool,
    ) -> Tuple[int, ...]:
        """Get shape of a batch source.

        Parameter, State, and Control have .shape directly.
        Constant has .value.shape.
        """
        if is_param or is_state or is_control:
            return batch.shape
        return batch.value.shape

    def children(self) -> List["Expr"]:
        """Return child expressions.

        Returns:
            list: The vmapped expression and any Parameter/State/Control data sources.
                  These are included so traverse() finds them for parameter/variable
                  collection in preprocessing.
        """
        result = [self._child]
        # Include Parameter/State/Control batches so they are discovered during traversal
        for b, is_param, is_state, is_control in zip(
            self._batches, self._is_parameter, self._is_state, self._is_control
        ):
            if is_param or is_state or is_control:
                result.append(b)
        return result

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing the child expression.

        Returns:
            Vmap: A new Vmap with canonicalized child expression
        """
        canon_child = self._child.canonicalize()
        # Create new Vmap with the canonicalized child
        new_vmap = Vmap.__new__(Vmap)
        new_vmap._batches = self._batches
        new_vmap._axis = self._axis
        new_vmap._placeholders = self._placeholders
        new_vmap._child = canon_child
        new_vmap._is_parameter = self._is_parameter
        new_vmap._is_state = self._is_state
        new_vmap._is_control = self._is_control
        return new_vmap

    def check_shape(self) -> Tuple[int, ...]:
        """Compute the output shape of the vmapped expression.

        The output shape is (batch_size,) + inner_shape, where batch_size
        is the size of the vmap axis and inner_shape is the shape of the
        child expression.

        Returns:
            tuple: Output shape after vmapping

        Example:
            If data has shape (10, 3) and the inner expression produces a
            scalar (shape ()), the output shape is (10,).
        """
        inner_shape = self._child.check_shape()

        # Get batch size from first batch (all batches have same size along axis)
        first_shape = Vmap._get_batch_shape(
            self._batches[0],
            self._is_parameter[0],
            self._is_state[0],
            self._is_control[0],
        )
        batch_size = first_shape[self._axis]

        return (batch_size,) + inner_shape

    def _hash_into(self, hasher):
        """Hash Vmap including data sources, axis, and child expression.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Vmap")
        hasher.update(str(self._axis).encode())
        hasher.update(str(len(self._batches)).encode())

        for b, is_param, is_state, is_control in zip(
            self._batches, self._is_parameter, self._is_state, self._is_control
        ):
            hasher.update(str(is_param).encode())
            hasher.update(str(is_state).encode())
            hasher.update(str(is_control).encode())
            if is_param or is_state or is_control:
                # Hash Parameter/State/Control by name and shape (not value - value can change)
                b._hash_into(hasher)
            else:
                # Hash Constant by value (baked in, won't change)
                hasher.update(b.value.tobytes())

        self._child._hash_into(hasher)

    def __repr__(self) -> str:
        """String representation of the Vmap expression.

        Returns:
            str: Description of the Vmap
        """
        batch_strs = []
        for b, is_param, is_state, is_control in zip(
            self._batches, self._is_parameter, self._is_state, self._is_control
        ):
            if is_param:
                batch_strs.append(f"Parameter({b.name!r})")
            elif is_state:
                batch_strs.append(f"State({b.name!r})")
            elif is_control:
                batch_strs.append(f"Control({b.name!r})")
            else:
                batch_strs.append(f"Constant(shape={b.value.shape})")

        if len(batch_strs) == 1:
            batch_repr = batch_strs[0]
        else:
            batch_repr = "[" + ", ".join(batch_strs) + "]"

        return f"Vmap(batch={batch_repr}, axis={self._axis})"
