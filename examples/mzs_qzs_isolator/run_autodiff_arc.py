from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.mzs_qzs_isolator.autodiff_model import MzsQzsAutodiffModel
from pyhb import ContinuationAutodiffConfig, ContinuationAutodiffSolver, ContinuationResult

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc.npz"
DEFAULT_INITIAL_CONDITION = Path(__file__).resolve().parent / "data" / "initial_condition_1period.npz"
DEFAULT_MAX_STEPS = 480
DEFAULT_SAMPLE_FFT = 2**11
FREQUENCY_RESOLUTION = 1.0
INIT_OMEGA = 0.01
MAX_EPOCH = 30
RES_TOLERANCE = 1e-8
DELTA_TOLERANCE = 1e-10
Q_SCALE = 1.0
OMEGA_SCALE = 1.0
MAX_PARAMETER_STEP = 0.01
HARMONICS = tuple(float(value) for value in range(1, 50 + 1))
DEFAULT_PLOT_DOFS = (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MZS-QZS isolator autodiff arc-length example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-condition", type=Path, default=DEFAULT_INITIAL_CONDITION)
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
        s_initial=0.005,
        s_max=0.02,
        s_min=1e-8,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        progress_callback=print,
        torch_device=args.torch_device,
    )


def build_initial_coefficients(model: MzsQzsAutodiffModel, order: int) -> NDArray[np.float64]:
    return load_initial_coefficients(DEFAULT_INITIAL_CONDITION, model, order)[0]


def load_initial_coefficients(
    input_path: Path,
    model: MzsQzsAutodiffModel,
    order: int,
) -> tuple[NDArray[np.float64], float]:
    expected_order = 2 * len(HARMONICS) + 1
    if order != expected_order:
        raise ValueError(f"order must be {expected_order} for the configured harmonics, got {order}")
    with np.load(input_path) as data:
        coefficients = np.asarray(data["hb_coefficient"], dtype=np.float64)
        omega = float(np.asarray(data["omega"], dtype=np.float64).item())
        harmonics = np.asarray(data["harmonics"], dtype=np.float64).reshape(-1)
        frequency_resolution = float(np.asarray(data["frequency_resolution"], dtype=np.float64).item())

    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(order, model.n_dof)
    if coefficients.shape != (order, model.n_dof):
        raise ValueError(f"initial hb_coefficient must have shape {(order, model.n_dof)}, got {coefficients.shape}")
    if not np.allclose(harmonics, np.asarray(HARMONICS, dtype=np.float64), atol=1e-12, rtol=0.0):
        raise ValueError("initial condition harmonics do not match this HB run")
    if not np.isclose(frequency_resolution, FREQUENCY_RESOLUTION, atol=1e-12, rtol=0.0):
        raise ValueError(
            "initial condition frequency_resolution does not match this HB run; "
            f"got {frequency_resolution}, expected {FREQUENCY_RESOLUTION}"
        )
    if not np.isclose(omega, INIT_OMEGA, atol=1e-12, rtol=0.0):
        raise ValueError(f"initial condition omega={omega} does not match INIT_OMEGA={INIT_OMEGA}")
    return coefficients, omega


def save_result(result: ContinuationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        parameter_history=result.parameter_history,
        coefficient_history=result.coefficient_history,
    )
    print(f"Saved result to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    model = MzsQzsAutodiffModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    initial_condition = Path(getattr(args, "initial_condition", DEFAULT_INITIAL_CONDITION))
    initial_coefficients, _ = load_initial_coefficients(initial_condition, model, order)
    result = ContinuationAutodiffSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
