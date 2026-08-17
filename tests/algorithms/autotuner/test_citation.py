"""Autotuner ``citation()`` returns the references behind each update rule.

Published schemes cite their papers; unpublished heuristics return the base
class's empty default. Like the sibling autotuner tests, this reads class
metadata only — no solves.
"""

from openscvx.algorithms.autotuner.acceptance_ratio import AcceptanceRatioAutotuner
from openscvx.algorithms.autotuner.adaptive_proximal_weight import AdaptiveProximalWeight
from openscvx.algorithms.autotuner.augmented_lagrangian import AugmentedLagrangian
from openscvx.algorithms.autotuner.constant_proximal_weight import ConstantProximalWeight
from openscvx.algorithms.autotuner.ramp_proximal_weight import RampProximalWeight

# =============================================================================
# Published update rules cite their papers
# =============================================================================


def test_acceptance_ratio_cites_scvx():
    entries = AcceptanceRatioAutotuner().citation()
    assert any("mao2016scvx" in e for e in entries)
    assert any("mao2019scvx" in e for e in entries)


def test_augmented_lagrangian_extends_scvx_with_scvx_star():
    entries = AugmentedLagrangian().citation()
    # Inherits the SCvx acceptance-ratio references and adds SCvx*.
    assert any("mao2016scvx" in e for e in entries)
    assert any("oguri2023scvxstar" in e for e in entries)


def test_adaptive_proximal_weight_inherits_scvx_citations():
    entries = AdaptiveProximalWeight().citation()
    assert any("mao2016scvx" in e for e in entries)


# =============================================================================
# Unpublished heuristics keep the empty default
# =============================================================================


def test_uncited_autotuners_return_empty():
    assert ConstantProximalWeight().citation() == []
    assert RampProximalWeight().citation() == []
