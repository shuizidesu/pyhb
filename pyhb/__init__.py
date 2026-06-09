"""Generic pyHB harmonic-balance tools for nonlinear dynamics."""

from .continuation import ContinuationConfig, ContinuationResult, ContinuationSolver
from .condensed_continuation import (
    CondensedContinuationConfig,
    CondensedContinuationLog,
    CondensedContinuationResult,
    CondensedContinuationSolver,
)
from .floquet import FloquetConfig, FloquetResult, compute_floquet
from .hb_operators import FrequencyGrid, build_full_fft_nonlinear_harmonics
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
    "compute_floquet_autodiff",
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
    "FloquetConfig",
    "FloquetResult",
    "ForcingTerm",
    "LinearOperatorTerm",
    "LocalNonlinearJacobianTerm",
    "NonlinearJacobianTerm",
    "SecondOrderTimeModel",
    "build_full_fft_nonlinear_harmonics",
    "compute_floquet",
    "compute_floquet_autodiff",
]


def __getattr__(name: str):
    if name in _AUTODIFF_EXPORTS:
        from .continuation_autodiff import ContinuationAutodiffConfig, ContinuationAutodiffSolver
        from .floquet_autodiff import compute_floquet_autodiff

        values = {
            "ContinuationAutodiffConfig": ContinuationAutodiffConfig,
            "ContinuationAutodiffSolver": ContinuationAutodiffSolver,
            "compute_floquet_autodiff": compute_floquet_autodiff,
        }
        return values[name]
    raise AttributeError(f"module 'pyhb' has no attribute {name!r}")
