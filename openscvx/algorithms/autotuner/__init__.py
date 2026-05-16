"""SCP weight autotuning strategies."""

from .adaptive_proximal_weight import AdaptiveProximalWeight, AdaptiveProximalWeightSpec
from .augmented_lagrangian import AugmentedLagrangian, AugmentedLagrangianSpec
from .constant_proximal_weight import ConstantProximalWeight, ConstantProximalWeightSpec
from .ramp_proximal_weight import RampProximalWeight, RampProximalWeightSpec

__all__ = [
    "AdaptiveProximalWeight",
    "AdaptiveProximalWeightSpec",
    "AugmentedLagrangian",
    "AugmentedLagrangianSpec",
    "ConstantProximalWeight",
    "ConstantProximalWeightSpec",
    "RampProximalWeight",
    "RampProximalWeightSpec",
]
