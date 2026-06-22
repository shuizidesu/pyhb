"""Generic pyHB harmonic-balance tools for nonlinear dynamics."""

from .continuation import ContinuationConfig, ContinuationResult, ContinuationSolver
from .floquet import FloquetConfig, FloquetResult, compute_floquet, compute_free_frequency_floquet
from .free_frequency import (
    FreeFrequencyContinuationConfig,
    FreeFrequencyContinuationResult,
    FreeFrequencyContinuationSolver,
    FreeFrequencyStepLog,
)
from .hb_operators import FrequencyGrid, build_full_fft_nonlinear_harmonics
from .models import (
    AutodiffSecondOrderTimeModel,
    AutodiffFreeFrequencySecondOrderTimeModel,
    ForcingTerm,
    FreeFrequencySecondOrderTimeModel,
    HarmonicCoefficientConstraint,
    LinearOperatorTerm,
    LocalNonlinearJacobianTerm,
    LocalResidualJacobianTerm,
    NonlinearJacobianTerm,
    SecondOrderTimeModel,
)

_AUTODIFF_EXPORTS = {
    "ContinuationAutodiffConfig",
    "ContinuationAutodiffSolver",
    "FreeFrequencyContinuationAutodiffConfig",
    "FreeFrequencyContinuationAutodiffSolver",
    "compute_free_frequency_floquet_autodiff",
    "compute_floquet_autodiff",
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
    "FreeFrequencyContinuationConfig",
    "FreeFrequencyContinuationResult",
    "FreeFrequencyContinuationSolver",
    "FreeFrequencySecondOrderTimeModel",
    "FreeFrequencyStepLog",
    "LinearOperatorTerm",
    "LocalNonlinearJacobianTerm",
    "LocalResidualJacobianTerm",
    "NonlinearJacobianTerm",
    "HarmonicCoefficientConstraint",
    "SecondOrderTimeModel",
    "AutodiffFreeFrequencySecondOrderTimeModel",
    "build_full_fft_nonlinear_harmonics",
    "compute_free_frequency_floquet",
    "compute_floquet",
]


def __getattr__(name: str):
    if name in _AUTODIFF_EXPORTS:
        from .continuation_autodiff import ContinuationAutodiffConfig, ContinuationAutodiffSolver
        from .floquet_autodiff import compute_floquet_autodiff, compute_free_frequency_floquet_autodiff
        from .free_frequency_autodiff import (
            FreeFrequencyContinuationAutodiffConfig,
            FreeFrequencyContinuationAutodiffSolver,
        )

        values = {
            "ContinuationAutodiffConfig": ContinuationAutodiffConfig,
            "ContinuationAutodiffSolver": ContinuationAutodiffSolver,
            "FreeFrequencyContinuationAutodiffConfig": FreeFrequencyContinuationAutodiffConfig,
            "FreeFrequencyContinuationAutodiffSolver": FreeFrequencyContinuationAutodiffSolver,
            "compute_free_frequency_floquet_autodiff": compute_free_frequency_floquet_autodiff,
            "compute_floquet_autodiff": compute_floquet_autodiff,
        }
        return values[name]
    raise AttributeError(f"module 'pyhb' has no attribute {name!r}")
