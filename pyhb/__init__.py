"""Generic pyHB harmonic-balance tools for nonlinear dynamics."""

from importlib import import_module

from .continuation import ContinuationConfig, ContinuationResult, ContinuationSolver
from .continuation_free_frequency import (
    ContinuationFreeFrequencyConfig,
    ContinuationFreeFrequencyResult,
    ContinuationFreeFrequencySolver,
    ContinuationFreeFrequencyStepLog,
)
from .floquet import FloquetConfig, FloquetResult, compute_floquet, compute_free_frequency_floquet
from .hb_operators import FrequencyGrid
from .models import (
    AutodiffFreeFrequencySecondOrderTimeModel,
    AutodiffSecondOrderTimeModel,
    ForcingTerm,
    FreeFrequencySecondOrderTimeModel,
    HarmonicCoefficientConstraint,
    LinearOperatorTerm,
    LocalJacobianMatrices,
    ReferencePhaseCondition,
    SecondOrderTimeModel,
)

__version__ = "0.1.0"

_AUTODIFF_EXPORTS = {
    "ContinuationAutodiffConfig": ("continuation_autodiff", "ContinuationAutodiffConfig"),
    "ContinuationAutodiffSolver": ("continuation_autodiff", "ContinuationAutodiffSolver"),
    "ContinuationFreeFrequencyAutodiffConfig": (
        "continuation_free_frequency_autodiff",
        "ContinuationFreeFrequencyAutodiffConfig",
    ),
    "ContinuationFreeFrequencyAutodiffSolver": (
        "continuation_free_frequency_autodiff",
        "ContinuationFreeFrequencyAutodiffSolver",
    ),
    "compute_free_frequency_floquet_autodiff": (
        "floquet_autodiff",
        "compute_free_frequency_floquet_autodiff",
    ),
    "compute_floquet_autodiff": ("floquet_autodiff", "compute_floquet_autodiff"),
    "compute_mixed_order_floquet_autodiff": (
        "floquet_mixed_autodiff",
        "compute_mixed_order_floquet_autodiff",
    ),
}

__all__ = [
    "AutodiffSecondOrderTimeModel",
    "ContinuationConfig",
    "ContinuationResult",
    "ContinuationSolver",
    "FrequencyGrid",
    "FloquetConfig",
    "FloquetResult",
    "ForcingTerm",
    "ContinuationFreeFrequencyConfig",
    "ContinuationFreeFrequencyResult",
    "ContinuationFreeFrequencySolver",
    "FreeFrequencySecondOrderTimeModel",
    "ContinuationFreeFrequencyStepLog",
    "LinearOperatorTerm",
    "LocalJacobianMatrices",
    "HarmonicCoefficientConstraint",
    "ReferencePhaseCondition",
    "SecondOrderTimeModel",
    "__version__",
    "AutodiffFreeFrequencySecondOrderTimeModel",
    "compute_free_frequency_floquet",
    "compute_floquet",
]


def __getattr__(name: str):
    if name in _AUTODIFF_EXPORTS:
        module_name, attribute_name = _AUTODIFF_EXPORTS[name]
        value = getattr(import_module(f".{module_name}", __name__), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pyhb' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_AUTODIFF_EXPORTS))
