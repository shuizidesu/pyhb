from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ega_ihb import CondensedContinuationConfig, CondensedContinuationResult, CondensedContinuationSolver
from examples.aeroengine.model import AeroEngineRotorModel


def harmonic_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + step * index, 10) for index in range(count))


HARMONICS = harmonic_range(0.5, 3.1, 0.1)
FREQUENCY_RESOLUTION = 0.1
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "condensed_arc.npz"
DEFAULT_SAMPLE_FFT = 2**11
DEFAULT_MAX_STEPS = 400
DEFAULT_PLOT_DOFS = (34, 176, 68, 210)
INIT_OMEGA = 145.0
MAX_EPOCH = 25
RES_TOLERANCE = 1e-9
DELTA_TOLERANCE = 1e-12
Q_SCALE = 1e-4
OMEGA_SCALE = 200
LOOP_SWITCH_ENABLED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the condensed aero-engine arc-length continuation example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_config(args: argparse.Namespace, init_omega: float = INIT_OMEGA) -> CondensedContinuationConfig:
    return CondensedContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=init_omega,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.1,
        s_max=0.1,
        s_min=1e-9,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        loop_switch_enabled=LOOP_SWITCH_ENABLED,
        loop_revisit_tolerance=0.01,
        loop_restart_omega_delta=0.5,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def build_default_initial_coefficients(order: int, nonlinear_count: int) -> NDArray[np.float64]:
    index = np.arange(order * nonlinear_count, dtype=np.float64).reshape(order, nonlinear_count)
    return (0.5 + 0.5 * np.sin(index + 1.0)) * 1e-6


def extract_initial_from_saved_result(
    path: str | Path,
    index: int,
    nonlinear_dofs: tuple[int, ...],
) -> tuple[float, NDArray[np.float64]]:
    with np.load(path) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    return float(parameter_history[index]), coefficient_history[index][:, list(nonlinear_dofs)].copy()


def save_result(result: CondensedContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = AeroEngineRotorModel()
    solver = CondensedContinuationSolver(model, build_config(args))
    initial_coefficients = build_default_initial_coefficients(
        solver.prepared.context.order,
        len(solver.prepared.nonlinear_dofs),
    )
    result = solver.run(initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
