"""Generic EGA-IHB harmonic-balance tools for nonlinear dynamics."""

from .continuation import ContinuationConfig, ContinuationResult, ContinuationSolver
from .condensed_continuation import (
    CondensedContinuationConfig,
    CondensedContinuationLog,
    CondensedContinuationLoopEvent,
    CondensedContinuationResult,
    CondensedContinuationSolver,
    ForcingTerm,
    LinearOperatorTerm,
)
from .hb_operators import FrequencyGrid, build_full_fft_nonlinear_harmonics, build_quadratic_nonlinear_harmonics
from .models import NonlinearJacobianTerm, SecondOrderTimeModel

__all__ = [
    "CondensedContinuationConfig",
    "CondensedContinuationLog",
    "CondensedContinuationLoopEvent",
    "CondensedContinuationResult",
    "CondensedContinuationSolver",
    "ContinuationConfig",
    "ContinuationResult",
    "ContinuationSolver",
    "FrequencyGrid",
    "ForcingTerm",
    "LinearOperatorTerm",
    "NonlinearJacobianTerm",
    "SecondOrderTimeModel",
    "build_full_fft_nonlinear_harmonics",
    "build_quadratic_nonlinear_harmonics",
]
