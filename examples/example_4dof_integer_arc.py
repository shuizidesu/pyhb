from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ega_ihb import ContinuationConfig, ContinuationResult, ContinuationSolver
from examples.four_dof_integer_model import FourDofIntegerModel


DEFAULT_OUTPUT = Path("results/example_4dof_integer_arc.npz")
DEFAULT_MAX_STEPS = 3000
DEFAULT_SAMPLE_FFT = 2**11
FREQUENCY_RESOLUTION = 1.0
HARMONICS = tuple(float(value) for value in range(1, 11))
DEFAULT_PLOT_DOFS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4DOF integer-harmonic arc-length example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        seed=args.seed,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def save_result(result: ContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    result = ContinuationSolver(FourDofIntegerModel(), build_config(args)).run()
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
