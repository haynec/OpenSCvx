"""Every autotuner knob default is declared identically in all three places.

Each tunable knob is declared three times — on the :class:`HyperParams`
dataclass (the runtime container and pytree leaf), in the autotuner
``__init__`` signature (the Python constructor surface), and on the pydantic
Spec (dict / YAML validation). Keeping the three explicit buys IDE
completion, generated docs, and YAML validation, but nothing guarantees they
agree: a default edited in one place and not the others is a silent drift that
only surfaces when a user reaches the knob through a different channel than the
one that was updated.

This pins the agreement: for every field declared on the Hyper dataclass, the
default must match the ``__init__`` default and the Spec ``model_fields``
default. Mirrors the synthetic-free, full-solve-free style of the sibling
autotuner tests — it reads class metadata only, so it stays instant.
"""

import inspect
from dataclasses import fields as dc_fields

import pytest

from openscvx.algorithms.autotuner.acceptance_ratio import AcceptanceRatioHyper
from openscvx.algorithms.autotuner.adaptive_proximal_weight import (
    AdaptiveProximalWeight,
    AdaptiveProximalWeightSpec,
)
from openscvx.algorithms.autotuner.augmented_lagrangian import (
    AugmentedLagrangian,
    AugmentedLagrangianHyper,
    AugmentedLagrangianSpec,
)
from openscvx.algorithms.autotuner.constant_proximal_weight import (
    ConstantProximalWeight,
    ConstantProximalWeightHyper,
    ConstantProximalWeightSpec,
)
from openscvx.algorithms.autotuner.ramp_proximal_weight import (
    RampProximalWeight,
    RampProximalWeightHyper,
    RampProximalWeightSpec,
)

# Each triple binds an autotuner to its Hyper dataclass and pydantic Spec; the
# Hyper field set is the canonical knob list the other two must agree with.
_AUTOTUNER_TRIPLES = [
    pytest.param(
        AugmentedLagrangian,
        AugmentedLagrangianHyper,
        AugmentedLagrangianSpec,
        id="AugmentedLagrangian",
    ),
    pytest.param(
        AdaptiveProximalWeight,
        AcceptanceRatioHyper,
        AdaptiveProximalWeightSpec,
        id="AdaptiveProximalWeight",
    ),
    pytest.param(
        ConstantProximalWeight,
        ConstantProximalWeightHyper,
        ConstantProximalWeightSpec,
        id="ConstantProximalWeight",
    ),
    pytest.param(
        RampProximalWeight,
        RampProximalWeightHyper,
        RampProximalWeightSpec,
        id="RampProximalWeight",
    ),
]


# === Default agreement across Hyper / __init__ / Spec =======================


@pytest.mark.parametrize("autotuner_cls, hyper_cls, spec_cls", _AUTOTUNER_TRIPLES)
def test_defaults_agree(autotuner_cls, hyper_cls, spec_cls):
    """Hyper default == __init__ default == Spec default for every knob."""
    init_defaults = {
        name: param.default
        for name, param in inspect.signature(autotuner_cls.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    spec_defaults = {name: fld.default for name, fld in spec_cls.model_fields.items()}

    for fld in dc_fields(hyper_cls):
        name = fld.name
        hyper_default = fld.default

        assert name in init_defaults, (
            f"{autotuner_cls.__name__}.__init__ is missing a default for "
            f"declared hyperparameter {name!r}"
        )
        assert name in spec_defaults, (
            f"{spec_cls.__name__} is missing a default for declared "
            f"hyperparameter {name!r}"
        )

        assert hyper_default == init_defaults[name], (
            f"{name}: {hyper_cls.__name__} default {hyper_default!r} != "
            f"{autotuner_cls.__name__}.__init__ default {init_defaults[name]!r}"
        )
        assert hyper_default == spec_defaults[name], (
            f"{name}: {hyper_cls.__name__} default {hyper_default!r} != "
            f"{spec_cls.__name__} default {spec_defaults[name]!r}"
        )


@pytest.mark.parametrize("autotuner_cls, hyper_cls, spec_cls", _AUTOTUNER_TRIPLES)
def test_every_knob_is_declared(autotuner_cls, hyper_cls, spec_cls):
    """Every __init__ / Spec knob is a declared Hyper field.

    The reverse direction of :func:`test_defaults_agree`: a knob added to the
    constructor or the Spec but not declared on the Hyper would be baked into
    the trace — unreachable by per-solve overrides and batched sweeps — which
    is exactly the contract violation the HyperParams machinery exists to
    prevent.
    """
    hyper_fields = {fld.name for fld in dc_fields(hyper_cls)}
    init_knobs = {
        name
        for name, param in inspect.signature(autotuner_cls.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    spec_knobs = set(spec_cls.model_fields) - {"type"}

    undeclared_init = init_knobs - hyper_fields
    assert not undeclared_init, (
        f"{autotuner_cls.__name__}.__init__ takes knob(s) {sorted(undeclared_init)} "
        f"not declared on {hyper_cls.__name__} — they would be baked into the "
        f"trace and invisible to the override/sweep channel"
    )
    undeclared_spec = spec_knobs - hyper_fields
    assert not undeclared_spec, (
        f"{spec_cls.__name__} validates knob(s) {sorted(undeclared_spec)} "
        f"not declared on {hyper_cls.__name__}"
    )
