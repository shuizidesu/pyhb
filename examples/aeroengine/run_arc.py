from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb import (
    ContinuationConfig,
    ContinuationResult,
    ContinuationSolver,
)
from examples.aeroengine.model import AeroEngineRotorModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_MAX_STEPS = 800
DEFAULT_SAMPLE_FFT = 2 ** 11
FREQUENCY_RESOLUTION = 0.1
INIT_OMEGA = 145.0
MAX_EPOCH = 25
INITIAL_SCALE = 1e-5
RES_TOLERANCE = 1e-10
DELTA_TOLERANCE = 1e-12
Q_SCALE = 1e-4
OMEGA_SCALE = 100
MAX_PARAMETER_STEP = 0.7
DEFAULT_PLOT_DOFS = (34, 176, 68, 210)


def harmonic_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + step * index, 10) for index in range(count))


HARMONICS = harmonic_range(0.5, 3.1, 0.1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the aero-engine rotor pyHB arc-length example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=INIT_OMEGA,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.1,
        s_max=0.3,
        s_min=1e-9,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def build_initial_coefficients(model: AeroEngineRotorModel, order: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return rng.random((order, model.n_dof), dtype=np.float64) * INITIAL_SCALE


def save_result(result: ContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = AeroEngineRotorModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    initial_coefficients = build_initial_coefficients(model, order)
    result = ContinuationSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
