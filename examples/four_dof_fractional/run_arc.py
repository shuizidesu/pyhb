from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb import ContinuationConfig, ContinuationResult, ContinuationSolver
from examples.four_dof_fractional.model import FourDofFractionalModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_MAX_STEPS = 290
DEFAULT_SAMPLE_FFT = 2 ** 11
FREQUENCY_RESOLUTION = 0.2
FRACTIONAL_HARMONICS = (
    0.2,
    1.0,
    1.2,
    2.0,
    2.2,
    2.4,
    3.0,
    3.2,
    3.4,
    3.6,
    4.0,
    4.2,
    4.4,
    4.6,
    4.8,
    5.0,
    5.2,
    5.4,
    5.6,
    5.8,
    6.0,
)
BOOTSTRAP_HARMONICS = FRACTIONAL_HARMONICS[:15]
DEFAULT_PLOT_DOFS = (0, 1, 2, 3)
Q_SCALE = 2.0
OMEGA_SCALE = 300.0
INITIAL_SCALE = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4DOF fractional-harmonic arc-length example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-bootstrap", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        max_steps=args.max_steps,
        frequency_resolution=FREQUENCY_RESOLUTION,
        harmonics=FRACTIONAL_HARMONICS,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        progress_callback=print,
    )


def build_default_initial_coefficients(model: FourDofFractionalModel, order: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return rng.random((order, model.n_dof), dtype=np.float64) * INITIAL_SCALE


def build_bootstrap_config(sample_fft: int) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=sample_fft,
        harmonics=BOOTSTRAP_HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        max_steps=0,
        progress_callback=print,
    )


def build_bootstrap_initial_coefficients(model: FourDofFractionalModel, sample_fft: int) -> NDArray[np.float64]:
    bootstrap_config = build_bootstrap_config(sample_fft)
    bootstrap_initial = build_default_initial_coefficients(model, 2 * len(BOOTSTRAP_HARMONICS) + 1)
    bootstrap_result = ContinuationSolver(model, bootstrap_config).run(initial_coefficients=bootstrap_initial)
    final_order = 2 * len(FRACTIONAL_HARMONICS) + 1
    initial = np.zeros((final_order, model.n_dof), dtype=np.float64)
    initial[0, :] = bootstrap_result.coefficients[0, :]
    for source_index, harmonic in enumerate(BOOTSTRAP_HARMONICS):
        target_index = FRACTIONAL_HARMONICS.index(harmonic)
        initial[1 + target_index, :] = bootstrap_result.coefficients[1 + source_index, :]
        initial[1 + len(FRACTIONAL_HARMONICS) + target_index, :] = bootstrap_result.coefficients[
            1 + len(BOOTSTRAP_HARMONICS) + source_index,
            :,
        ]
    return initial


def save_result(result: ContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = FourDofFractionalModel()
    config = build_config(args)
    solver = ContinuationSolver(model, config)
    initial_coefficients = (
        build_default_initial_coefficients(model, 2 * len(FRACTIONAL_HARMONICS) + 1)
        if args.no_bootstrap
        else build_bootstrap_initial_coefficients(model, args.sample_fft)
    )
    result = solver.run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
