from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.bistable_beam_structure.autodiff_model import BistableBeamStructureAutodiffModel
from pyhb import ContinuationAutodiffConfig, ContinuationAutodiffSolver, ContinuationResult

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc.npz"
DEFAULT_MAX_STEPS = 350
DEFAULT_SAMPLE_FFT = 2**11
FREQUENCY_RESOLUTION = 0.25
INIT_OMEGA = 30.0
INITIAL_DIRECTION = "up"
MAX_EPOCH = 40
RES_TOLERANCE = 1e-8
DELTA_TOLERANCE = 1e-10
Q_SCALE = 1e-3
OMEGA_SCALE = 100.0
MAX_PARAMETER_STEP = 1.0
HARMONICS = tuple(float(value * 0.25) for value in range(1, 51))
DEFAULT_PLOT_DOFS = (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-DOF bistable beam structure example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--initial-direction", choices=("up", "down"), default=INITIAL_DIRECTION)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationAutodiffConfig:
    return ContinuationAutodiffConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=INIT_OMEGA,
        initial_direction=args.initial_direction,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.05,
        s_max=0.1,
        s_min=1e-8,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        progress_callback=print,
        torch_device=args.torch_device,
    )


def build_initial_coefficients(
    model: BistableBeamStructureAutodiffModel,
    order: int,
) -> NDArray[np.float64]:
    coefficients = np.zeros((order, model.n_dof), dtype=np.float64)
    coefficients[0, 0] = -model.positive_well_displacement
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
    model = BistableBeamStructureAutodiffModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    initial_coefficients = build_initial_coefficients(model, order)
    result = ContinuationAutodiffSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
