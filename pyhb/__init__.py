"""Generic pyHB harmonic-balance tools for nonlinear dynamics."""

from .continuation import ContinuationConfig, ContinuationResult, ContinuationSolver
from .condensed_continuation import (
    CondensedContinuationConfig,
    CondensedContinuationLog,
    CondensedContinuationResult,
    CondensedContinuationSolver,
)
from .hb_operators import FrequencyGrid, build_full_fft_nonlinear_harmonics, build_quadratic_nonlinear_harmonics
from .models import (
    AutodiffSecondOrderTimeModel,
    ForcingTerm,
    LinearOperatorTerm,
    LocalNonlinearJacobianTerm,
    NonlinearJacobianTerm,
    SecondOrderTimeModel,
)

_AUTODIFF_EXPORTS = {
    "ContinuationAutodiffConfig",
    "ContinuationAutodiffSolver",
}

__all__ = [
    "CondensedContinuationConfig",
    "CondensedContinuationLog",
    "CondensedContinuationResult",
    "CondensedContinuationSolver",
    "AutodiffSecondOrderTimeModel",
    "ContinuationAutodiffConfig",
    "ContinuationAutodiffSolver",
    "ContinuationConfig",
    "ContinuationResult",
    "ContinuationSolver",
    "FrequencyGrid",
    "ForcingTerm",
    "LinearOperatorTerm",
    "LocalNonlinearJacobianTerm",
    "NonlinearJacobianTerm",
    "SecondOrderTimeModel",
    "build_full_fft_nonlinear_harmonics",
    "build_quadratic_nonlinear_harmonics",
]


def __getattr__(name: str):
    if name in _AUTODIFF_EXPORTS:
        from .continuation_autodiff import ContinuationAutodiffConfig, ContinuationAutodiffSolver

        values = {
            "ContinuationAutodiffConfig": ContinuationAutodiffConfig,
            "ContinuationAutodiffSolver": ContinuationAutodiffSolver,
        }
        return values[name]
    raise AttributeError(f"module 'pyhb' has no attribute {name!r}")
