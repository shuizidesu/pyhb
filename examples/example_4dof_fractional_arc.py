from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ega_ihb import ContinuationConfig, ContinuationResult, ContinuationSolver, build_quadratic_nonlinear_harmonics
from examples.four_dof_fractional_model import FourDofFractionalModel


DEFAULT_OUTPUT = Path("results/example_4dof_fractional_arc.npz")
DEFAULT_MAX_STEPS = 4000
DEFAULT_SAMPLE_FFT = 2**11
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
FRACTIONAL_NONLINEAR_HARMONICS = build_quadratic_nonlinear_harmonics(
    FRACTIONAL_HARMONICS,
    FREQUENCY_RESOLUTION,
    max_harmonic=12.0,
)
BOOTSTRAP_HARMONICS = FRACTIONAL_HARMONICS[:15]
BOOTSTRAP_NONLINEAR_HARMONICS = build_quadratic_nonlinear_harmonics(
    BOOTSTRAP_HARMONICS,
    FREQUENCY_RESOLUTION,
    max_harmonic=9.6,
)
DEFAULT_PLOT_DOFS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4DOF fractional-harmonic arc-length example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-bootstrap", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        seed=args.seed,
        max_steps=args.max_steps,
        frequency_resolution=FREQUENCY_RESOLUTION,
        harmonics=FRACTIONAL_HARMONICS,
        nonlinear_harmonics=FRACTIONAL_NONLINEAR_HARMONICS,
        progress_callback=print,
    )


def build_bootstrap_initial_coefficients(model: FourDofFractionalModel, final_config: ContinuationConfig) -> NDArray[np.float64]:
    bootstrap_config = ContinuationConfig(
        sample_fft=final_config.sample_fft,
        harmonics=BOOTSTRAP_HARMONICS,
        nonlinear_harmonics=BOOTSTRAP_NONLINEAR_HARMONICS,
        frequency_resolution=final_config.frequency_resolution,
        frequency_tolerance=final_config.frequency_tolerance,
        strict_fft_grid=final_config.strict_fft_grid,
        seed=final_config.seed,
        init_omega=final_config.init_omega,
        max_epoch=final_config.max_epoch,
        res_tolerance=final_config.res_tolerance,
        delta_tolerance=final_config.delta_tolerance,
        s_initial=final_config.s_initial,
        s_max=final_config.s_max,
        s_min=final_config.s_min,
        q_scale=final_config.q_scale,
        omega_scale=final_config.omega_scale,
        max_steps=0,
        shrink_limit=final_config.shrink_limit,
        plot_dt=final_config.plot_dt,
        s3_method=final_config.s3_method,
        s3_quadrature_samples=final_config.s3_quadrature_samples,
        progress_callback=final_config.progress_callback,
    )
    bootstrap_result = ContinuationSolver(model, bootstrap_config).run()
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
    initial_coefficients = None if args.no_bootstrap else build_bootstrap_initial_coefficients(model, config)
    result = solver.run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
