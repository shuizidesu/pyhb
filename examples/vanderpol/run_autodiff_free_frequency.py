from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb import FreeFrequencyContinuationAutodiffConfig, FreeFrequencyContinuationAutodiffSolver
from examples.vanderpol.autodiff_model import VanderpolAutodiffModel
from examples.vanderpol.run_free_frequency import (
    CONSTRAINT_TOLERANCE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SAMPLE_FFT,
    DELTA_TOLERANCE,
    FREQUENCY_RESOLUTION,
    HARMONICS,
    INIT_OMEGA,
    INIT_PARAMETER,
    MAX_EPOCH,
    OMEGA_SCALE,
    PARAMETER_SCALE,
    Q_SCALE,
    RES_TOLERANCE,
    build_initial_coefficients,
    save_result,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "autodiff_free_frequency.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the van der Pol autodiff free-frequency HB example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FreeFrequencyContinuationAutodiffConfig:
    return FreeFrequencyContinuationAutodiffConfig(
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
        max_parameter_step=0.1,
        max_steps=args.max_steps,
        torch_device=args.torch_device,
        progress_callback=print,
    )


def run_from_args(args: argparse.Namespace) -> None:
    model = VanderpolAutodiffModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    result = FreeFrequencyContinuationAutodiffSolver(model, config).run(
        np.asarray(build_initial_coefficients(order), dtype=np.float64)
    )
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()

