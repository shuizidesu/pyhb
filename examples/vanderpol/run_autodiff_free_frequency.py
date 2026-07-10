from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.vanderpol.autodiff_model import VanderpolAutodiffModel
from pyhb import (
    ContinuationFreeFrequencyAutodiffConfig,
    ContinuationFreeFrequencyAutodiffSolver,
    ContinuationFreeFrequencyResult,
    HarmonicCoefficientConstraint,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "autodiff_free_frequency.npz"
DEFAULT_MAX_STEPS = 80
DEFAULT_SAMPLE_FFT = 2**11
HARMONICS = tuple(float(value) for value in range(1, 51))
FREQUENCY_RESOLUTION = 1.0
INIT_OMEGA = 1.0
INIT_PARAMETER = 0.5
MAX_EPOCH = 30
RES_TOLERANCE = 1e-10
DELTA_TOLERANCE = 1e-11
CONSTRAINT_TOLERANCE = 1e-10
Q_SCALE = 1.0
OMEGA_SCALE = 1.0
PARAMETER_SCALE = 1.0
MAX_PARAMETER_STEP = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the van der Pol autodiff free-frequency HB example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationFreeFrequencyAutodiffConfig:
    return ContinuationFreeFrequencyAutodiffConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=INIT_OMEGA,
        init_parameter=INIT_PARAMETER,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        constraint_tolerance=CONSTRAINT_TOLERANCE,
        s_initial=0.05,
        s_max=0.1,
        s_min=1e-8,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        parameter_scale=PARAMETER_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        initial_constraint=HarmonicCoefficientConstraint(dof=0, coefficient_index=1, value=0.0),
        torch_device=args.torch_device,
        progress_callback=print,
    )


def build_initial_coefficients(order: int) -> NDArray[np.float64]:
    coefficients = np.zeros((order, 1), dtype=np.float64)
    harmonic_count = (order - 1) // 2
    coefficients[1, 0] = 1.0
    coefficients[1 + harmonic_count, 0] = 1.0
    return coefficients


def save_result(result: ContinuationFreeFrequencyResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        omega_history=result.omega_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = VanderpolAutodiffModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    result = ContinuationFreeFrequencyAutodiffSolver(model, config).run(build_initial_coefficients(order))
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
