from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.piezoelectric_magnetic_harvester.autodiff_model import (
    PiezoelectricMagneticHarvesterAutodiffModel,
)
from pyhb import ContinuationAutodiffConfig, ContinuationAutodiffSolver, ContinuationResult

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc.npz"
DEFAULT_MAX_STEPS = 400
DEFAULT_SAMPLE_FFT = 2**11
FREQUENCY_RESOLUTION = 1.0
INIT_OMEGA = 2.0 * np.pi * 18.0
MAX_EPOCH = 40
RES_TOLERANCE = 1e-8
DELTA_TOLERANCE = 1e-10
Q_SCALE = 1.0
OMEGA_SCALE = 100.0
MAX_PARAMETER_STEP = 1.0
HARMONICS = tuple(float(value) for value in range(1, 21))
DEFAULT_PLOT_DOFS = (1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the piezoelectric magnetic harvester autodiff arc-length example."
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--torch-device", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationAutodiffConfig:
    return ContinuationAutodiffConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=INIT_OMEGA,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.2,
        s_max=0.5,
        s_min=1e-8,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        progress_callback=print,
        torch_device=args.torch_device,
    )


def build_initial_coefficients(
    model: PiezoelectricMagneticHarvesterAutodiffModel,
    order: int,
) -> NDArray[np.float64]:
    coefficients = np.zeros((order, model.n_dof), dtype=np.float64)
    harmonic_count = len(HARMONICS)
    stiffness = _linearized_stiffness(model)
    constant_nonlinear = _magnetic_force_at_zero(model)
    force_cos = model.forcing_terms(np.array([0.0], dtype=np.float64))[0].samples[0]

    coefficients[0, :] = np.linalg.solve(stiffness, -constant_nonlinear)
    zero_force = np.zeros(model.n_dof, dtype=np.float64)
    for index, harmonic in enumerate(HARMONICS):
        omega_h = INIT_OMEGA * float(harmonic)
        dynamic_stiffness = stiffness - omega_h**2 * model.mass
        damping_block = omega_h * model.damping
        block = np.block(
            [
                [dynamic_stiffness, damping_block],
                [-damping_block, dynamic_stiffness],
            ]
        )
        rhs = np.concatenate((force_cos if np.isclose(harmonic, 1.0) else zero_force, zero_force))
        solution = np.linalg.solve(block, rhs)
        coefficients[1 + index, :] = solution[: model.n_dof]
        coefficients[1 + harmonic_count + index, :] = solution[model.n_dof :]
    return coefficients


def _magnetic_force_at_zero(model: PiezoelectricMagneticHarvesterAutodiffModel) -> NDArray[np.float64]:
    value = -float(model.magnetic_coefficient) / float(model.parameters.initial_gap) ** 4
    return np.asarray([value, value, 0.0], dtype=np.float64)


def _linearized_stiffness(model: PiezoelectricMagneticHarvesterAutodiffModel) -> NDArray[np.float64]:
    stiffness = np.asarray(model.stiffness, dtype=np.float64).copy()
    gap = float(model.parameters.initial_gap)
    slope = 4.0 * float(model.magnetic_coefficient) / gap**5
    stiffness[0, 0] += slope
    stiffness[0, 1] -= slope
    stiffness[1, 0] += slope
    stiffness[1, 1] -= slope
    return stiffness


def save_result(result: ContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = PiezoelectricMagneticHarvesterAutodiffModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    initial_coefficients = build_initial_coefficients(model, order)
    result = ContinuationAutodiffSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
