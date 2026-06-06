from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ega_ihb import ContinuationConfig, ContinuationResult, ContinuationSolver
from examples.four_dof_integer.model import FourDofIntegerModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_MAX_STEPS = 3000
DEFAULT_SAMPLE_FFT = 2**11
FREQUENCY_RESOLUTION = 1.0
HARMONICS = tuple(float(value) for value in range(1, 11))
DEFAULT_PLOT_DOFS = (0, 1, 2, 3)
INITIAL_SCALE = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4DOF integer-harmonic arc-length example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def build_initial_coefficients(model: FourDofIntegerModel, order: int) -> NDArray[np.float64]:
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
    model = FourDofIntegerModel()
    config = build_config(args)
    initial_coefficients = build_initial_coefficients(model, 2 * len(HARMONICS) + 1)
    result = ContinuationSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
