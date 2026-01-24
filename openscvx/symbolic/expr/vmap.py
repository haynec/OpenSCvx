"""Vmap expression for data-parallel operations.

This module provides symbolic support for JAX's vmap (vectorized map) operation,
enabling efficient data-parallel computations over batched data within the
symbolic expression framework.

Vmap supports two modes based on the type of `batch`:

- **Constant/array**: Values baked into the compiled function at trace time,
  equivalent to closure-captured values in BYOF. Use for static data.
- **Parameter**: Values looked up from params dict at runtime, allowing
  updates between SCP iterations. Use for data that may change.

Vmap also supports batching over multiple arguments by passing a list of
batch sources. Each batch source is mapped to a corresponding lambda argument.

Example:
    Compute distances from a position to multiple reference points::

        import openscvx as ox
        import numpy as np

        position = ox.State("position", shape=(3,))
        init_poses = np.random.randn(10, 3)  # 10 reference points

        # Option 1: Baked-in data (closure-equivalent)
        distances = ox.Vmap(
            lambda pose: ox.linalg.Norm(position - pose),
            batch=init_poses  # or batch=ox.Constant(init_poses)
        )

        # Option 2: Runtime-updateable Parameter
        refs = ox.Parameter("refs", shape=(10, 3), value=init_poses)
        distances = ox.Vmap(
            lambda pose: ox.linalg.Norm(position - pose),
            batch=refs
        )

    Batch over multiple arguments (e.g., centers and radii)::

        obs_centers = ox.Parameter("obs_centers", shape=(100, 3), value=centers)
        obs_radii = ox.Parameter("obs_radii", shape=(100,), value=radii)

        constraints = ox.Vmap(
            lambda center, radius: radius <= ox.linalg.Norm(position - center),
            batch=[obs_centers, obs_radii]
        )
"""

import uuid
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, Union

import numpy as np

from .expr import Constant, Expr, Leaf

if TYPE_CHECKING:
    from .expr import Parameter

# Type alias for a single batch source
BatchSource = Union[np.ndarray, Constant, "Parameter"]


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

    Attributes:
        _batches (tuple): Tuple of data sources (Constant or Parameter)
        _axis (int): The axis to vmap over (default: 0)
        _placeholders (tuple): Tuple of placeholders used in the expression
        _child (Expr): The expression tree built from the user's lambda
        _is_parameter (tuple): Tuple of bools indicating which batches are Parameters

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

    Note:
        - For static data that won't change, pass a numpy array or Constant
          to get closure-equivalent behavior (numerically identical to BYOF).
        - For data that needs to be updated between iterations, use Parameter.
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
                  - A single batch source (numpy array, Constant, or Parameter)
                  - A list/tuple of batch sources for multi-argument vmapping
                  Each batch source can be:
                  - numpy array: baked into compiled function (closure-equivalent)
                  - Constant: baked into compiled function (closure-equivalent)
                  - Parameter: looked up from params dict at runtime
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
        """
        from .expr import Parameter

        # Normalize input: convert single batch to list, then process each
        if isinstance(batch, (list, tuple)) and not isinstance(batch, np.ndarray):
            batch_list = list(batch)
        else:
            batch_list = [batch]

        # Normalize each batch source: wrap raw arrays in Constant
        normalized_batches = []
        is_parameter_flags = []
        for b in batch_list:
            if isinstance(b, np.ndarray):
                b = Constant(b)
            elif not isinstance(b, (Constant, Parameter)):
                # Try to convert to array then Constant
                b = Constant(np.asarray(b))
            normalized_batches.append(b)
            is_parameter_flags.append(isinstance(b, Parameter))

        self._batches = tuple(normalized_batches)
        self._axis = axis
        self._is_parameter = tuple(is_parameter_flags)

        # Get batch size from first batch and validate all batches match
        def get_batch_shape(b, is_param):
            return b.shape if is_param else b.value.shape

        first_shape = get_batch_shape(self._batches[0], self._is_parameter[0])
        if axis < 0 or axis >= len(first_shape):
            raise ValueError(f"Vmap axis {axis} out of bounds for data with shape {first_shape}")
        batch_size = first_shape[axis]

        # Validate all batches have the same size along the vmap axis
        for i, (b, is_param) in enumerate(zip(self._batches, self._is_parameter)):
            shape = get_batch_shape(b, is_param)
            if axis >= len(shape):
                raise ValueError(
                    f"Vmap axis {axis} out of bounds for batch {i} with shape {shape}"
                )
            if shape[axis] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: batch 0 has size {batch_size} along axis {axis}, "
                    f"but batch {i} has size {shape[axis]}"
                )

        # Create placeholders for each batch
        placeholders = []
        for b, is_param in zip(self._batches, self._is_parameter):
            shape = get_batch_shape(b, is_param)
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
    def batches(self) -> Tuple[Union[Constant, "Parameter"], ...]:
        """Tuple of batched data sources being vmapped over."""
        return self._batches

    @property
    def batch(self) -> Union[Constant, "Parameter"]:
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
    def num_batches(self) -> int:
        """Number of batch arguments."""
        return len(self._batches)

    def children(self):
        """Return child expressions.

        Returns:
            list: The vmapped expression and any Parameter data sources.
                  Parameters are included so traverse() finds them for parameter
                  collection in preprocessing.
        """
        result = [self._child]
        # Include Parameter batches so they are discovered during traversal
        for b, is_param in zip(self._batches, self._is_parameter):
            if is_param:
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
        first_batch = self._batches[0]
        if self._is_parameter[0]:
            batch_size = first_batch.shape[self._axis]
        else:
            batch_size = first_batch.value.shape[self._axis]

        return (batch_size,) + inner_shape

    def _hash_into(self, hasher):
        """Hash Vmap including data sources, axis, and child expression.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Vmap")
        hasher.update(str(self._axis).encode())
        hasher.update(str(len(self._batches)).encode())

        for b, is_param in zip(self._batches, self._is_parameter):
            hasher.update(str(is_param).encode())
            if is_param:
                # Hash Parameter by name and shape (not value - value can change)
                b._hash_into(hasher)
            else:
                # Hash Constant by value (baked in, won't change)
                hasher.update(b.value.tobytes())

        self._child._hash_into(hasher)

    def __repr__(self):
        """String representation of the Vmap expression.

        Returns:
            str: Description of the Vmap
        """
        batch_strs = []
        for b, is_param in zip(self._batches, self._is_parameter):
            if is_param:
                batch_strs.append(f"Parameter({b.name!r})")
            else:
                batch_strs.append(f"Constant(shape={b.value.shape})")

        if len(batch_strs) == 1:
            batch_repr = batch_strs[0]
        else:
            batch_repr = "[" + ", ".join(batch_strs) + "]"

        return f"Vmap(batch={batch_repr}, axis={self._axis})"
