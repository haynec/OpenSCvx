"""Typed autotuner hyperparameter containers.

:class:`HyperParams` is the base every autotuner subclasses to declare its
tunable knobs. Subclassing applies the frozen-dataclass transform and the JAX
pytree / ``jax.export`` registration so instances ride
:attr:`~openscvx.algorithms.state.AlgorithmState.hyper` through ``jit`` /
``vmap`` / the exported batched ``solve_batched`` artifact. The base sits at the
head of the algorithms import order: :mod:`state` depends on it, nothing here
depends on the rest of the package.
"""

from dataclasses import dataclass
from dataclasses import fields as dc_fields

import jax
from jax import export


class HyperParams:
    """Base for autotuner hyperparameter containers.

    Subclassing is the entire integration. Declare each tunable knob as an
    annotated field with its default — bare annotations, no ``@dataclass``
    decorator — and the base class does the rest:

    * applies the frozen-dataclass transform, so instances are immutable
      value objects updated with :func:`dataclasses.replace`;
    * registers the subclass as a JAX pytree (and for ``jax.export`` treedef
      serialization), so instances ride :attr:`AlgorithmState.hyper` through
      ``jit`` / ``vmap`` / the exported batched ``solve_batched`` artifact;
    * wires dtype handling off the field annotations when
      :meth:`AlgorithmState.from_settings` snapshots the instance onto the
      state — ``int`` fields get the iteration counter's dtype, ``float``
      fields the problem float dtype. Any other annotation is rejected at
      class definition.

    Example::

        class MyHyper(HyperParams):
            ramp: float = 2.0
            drop: int = -1

    The empty base is the "no declared hyperparameters" container: it
    flattens to zero pytree leaves.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # ``__init_subclass__`` runs before any decorator on the subclass
        # could, so the transform must happen here — which is exactly what
        # lets authors skip the dataclass and pytree mechanics entirely.
        dataclass(frozen=True)(cls)
        for fld in dc_fields(cls):
            # fld.type is a string under postponed annotation evaluation
            # (``from __future__ import annotations``); accept both forms.
            if fld.type not in (int, float, "int", "float"):
                raise TypeError(
                    f"{cls.__name__}.{fld.name}: HyperParams fields must be "
                    f"annotated int or float (got {fld.type!r}) — the "
                    f"annotation decides the dtype the knob gets on "
                    f"AlgorithmState.hyper."
                )
        jax.tree_util.register_dataclass(cls)
        export.register_pytree_node_serialization(
            cls,
            serialized_name=f"{cls.__module__}.{cls.__qualname__}",
            serialize_auxdata=lambda aux: b"",
            deserialize_auxdata=lambda data: (),
        )


# The base itself is a zero-field frozen dataclass and a zero-leaf pytree, so
# the shared ``HyperParams()`` default on ``AutotuningBase`` is a valid value
# for ``AlgorithmState.hyper``. Auxdata of a ``register_dataclass`` node is
# the (empty) tuple of meta-field values, hence ``()`` — not ``None`` — on
# deserialize; same for the subclasses registered above.
dataclass(frozen=True)(HyperParams)
jax.tree_util.register_dataclass(HyperParams)
export.register_pytree_node_serialization(
    HyperParams,
    # The serialized name is a stable serialization ID, not a module path: it
    # must match the string baked into already-exported ``solve_batched``
    # artifacts. It deliberately keeps the historical ``...base.HyperParams``
    # even though the class now lives in this module — changing it would
    # strand cached artifacts (``export.deserialize`` hard-errors on an
    # unknown name; see ``openscvx/utils/caching.py``).
    serialized_name="openscvx.algorithms.base.HyperParams",
    serialize_auxdata=lambda aux: b"",
    deserialize_auxdata=lambda data: (),
)
