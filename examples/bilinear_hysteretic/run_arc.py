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
from examples.bilinear_hysteretic.model import BilinearHystereticModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_MAX_STEPS = 300
DEFAULT_SAMPLE_FFT = 2 ** 11
FREQUENCY_RESOLUTION = 1.0
INIT_OMEGA = 2.0
MAX_EPOCH = 30
RES_TOLERANCE = 1e-8
DELTA_TOLERANCE = 1e-10
Q_SCALE = 1.0
OMEGA_SCALE = 1.0
MAX_PARAMETER_STEP = 0.02
EQUIVALENT_LINEAR_STIFFNESS = 10.0
HARMONICS = tuple(float(value) for value in range(1, 21))
DEFAULT_PLOT_DOFS = (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bilinear hysteretic full arc-length example.")
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
        s_initial=0.02,
        s_max=0.05,
        s_min=1e-8,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def build_initial_coefficients(model: BilinearHystereticModel, order: int) -> NDArray[np.float64]:
    coefficients = np.zeros((order, model.n_dof), dtype=np.float64)
    mass = model.mass
    damping = model.damping
    stiffness = np.array([[EQUIVALENT_LINEAR_STIFFNESS]], dtype=np.float64)
    force_cos = np.array([float(model.parameters.force_amplitude)], dtype=np.float64)
    zero_force = np.zeros(model.n_dof, dtype=np.float64)
    harmonic_count = len(HARMONICS)
    for index, harmonic in enumerate(HARMONICS):
        omega_h = INIT_OMEGA * float(harmonic)
        dynamic_stiffness = stiffness - omega_h**2 * mass
        gyroscopic_block = omega_h * damping
        block = np.block(
            [
                [dynamic_stiffness, gyroscopic_block],
                [-gyroscopic_block, dynamic_stiffness],
            ]
        )
        rhs = np.concatenate((force_cos if np.isclose(harmonic, 1.0) else zero_force, zero_force))
        solution = np.linalg.solve(block, rhs)
        coefficients[1 + index, :] = solution[: model.n_dof]
        coefficients[1 + harmonic_count + index, :] = solution[model.n_dof :]
    return coefficients


def save_result(result: ContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = BilinearHystereticModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    initial_coefficients = build_initial_coefficients(model, order)
    result = ContinuationSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
