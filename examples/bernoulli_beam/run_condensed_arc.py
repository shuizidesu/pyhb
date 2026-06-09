from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb import CondensedContinuationConfig, CondensedContinuationResult, CondensedContinuationSolver
from examples.bernoulli_beam.model import BernoulliBeamModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "condensed_arc.npz"
DEFAULT_MAX_STEPS = 80
DEFAULT_SAMPLE_FFT = 2 ** 11
FREQUENCY_RESOLUTION = 1.0
INIT_OMEGA = 4.0
MAX_EPOCH = 25
INITIAL_SCALE = 1e-2
RES_TOLERANCE = 5e-4
DELTA_TOLERANCE = 1e-8
Q_SCALE = 1.0
OMEGA_SCALE = 5.0
HARMONICS = tuple(float(value) for value in range(1, 6))
DEFAULT_PLOT_DOFS = (1998,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the condensed Bernoulli beam arc-length example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CondensedContinuationConfig:
    return CondensedContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=INIT_OMEGA,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.05,
        s_max=0.1,
        s_min=1e-9,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def build_full_initial_coefficients(model: BernoulliBeamModel, order: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return rng.standard_normal((order, model.n_dof), dtype=np.float64) * INITIAL_SCALE


def build_initial_coefficients(model: BernoulliBeamModel, order: int, nonlinear_dofs: tuple[int, ...]) -> NDArray[np.float64]:
    full_initial = build_full_initial_coefficients(model, order)
    return full_initial[:, list(nonlinear_dofs)].copy()


def save_result(result: CondensedContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = BernoulliBeamModel()
    solver = CondensedContinuationSolver(model, build_config(args))
    initial_coefficients = build_initial_coefficients(
        model,
        solver.prepared.context.order,
        solver.prepared.nonlinear_dofs,
    )
    result = solver.run(initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
