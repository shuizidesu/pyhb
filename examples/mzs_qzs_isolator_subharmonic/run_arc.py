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
from examples.mzs_qzs_isolator_subharmonic.model import MzsQzsModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_INITIAL_CONDITION = Path(__file__).resolve().parent / "data" / "initial_condition_subharmonic.npz"
DEFAULT_MAX_STEPS = 180
DEFAULT_SAMPLE_FFT = 2 ** 11
FREQUENCY_RESOLUTION = 1.0 / 3.0
MAX_EPOCH = 30
RES_TOLERANCE = 1e-8
DELTA_TOLERANCE = 1e-10
Q_SCALE = 1.0
OMEGA_SCALE = 1.0
MAX_PARAMETER_STEP = 0.003
HARMONICS = tuple(float(index * FREQUENCY_RESOLUTION) for index in range(1, 16))
DEFAULT_PLOT_DOFS = (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MZS-QZS isolator subharmonic analytical arc-length example.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-condition", type=Path, default=DEFAULT_INITIAL_CONDITION)
    return parser.parse_args()


def build_config(args: argparse.Namespace, init_omega: float) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=float(init_omega),
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.002,
        s_max=0.005,
        s_min=1e-8,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_parameter_step=MAX_PARAMETER_STEP,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def harmonic_key(harmonic: float) -> int:
    raw_key = float(harmonic) / FREQUENCY_RESOLUTION
    key = int(round(raw_key))
    if not np.isclose(raw_key, key, atol=1e-9, rtol=0.0):
        raise ValueError(f"harmonic {harmonic} is not aligned with frequency_resolution={FREQUENCY_RESOLUTION}")
    return key


def load_initial_coefficients(
    input_path: Path,
    model: MzsQzsModel,
    order: int,
) -> tuple[NDArray[np.float64], float]:
    with np.load(input_path) as data:
        source_coefficients = np.asarray(data["hb_coefficient"], dtype=np.float64)
        omega = float(np.asarray(data["omega"], dtype=np.float64).item())
        source_harmonics = tuple(float(value) for value in np.asarray(data["harmonics"], dtype=np.float64).reshape(-1))
        source_frequency_resolution = float(np.asarray(data["frequency_resolution"], dtype=np.float64).item())

    if not np.isclose(source_frequency_resolution, FREQUENCY_RESOLUTION, atol=1e-12, rtol=0.0):
        raise ValueError(
            "initial condition frequency_resolution does not match this HB run; "
            f"got {source_frequency_resolution}, expected {FREQUENCY_RESOLUTION}"
        )
    if omega <= 0.0:
        raise ValueError(f"initial omega must be positive, got {omega}")

    expected_source_order = 2 * len(source_harmonics) + 1
    if source_coefficients.ndim == 1:
        source_coefficients = source_coefficients.reshape(-1, model.n_dof)
    if source_coefficients.shape != (expected_source_order, model.n_dof):
        raise ValueError(
            "initial hb_coefficient shape is incompatible with the source harmonics; "
            f"got {source_coefficients.shape}, expected {(expected_source_order, model.n_dof)}"
        )

    coefficients = np.zeros((order, model.n_dof), dtype=np.float64)
    coefficients[0, :] = source_coefficients[0, :]
    target_index_by_key = {harmonic_key(harmonic): index for index, harmonic in enumerate(HARMONICS)}
    mapped_count = 0
    for source_index, source_harmonic in enumerate(source_harmonics):
        target_index = target_index_by_key.get(harmonic_key(source_harmonic))
        if target_index is None:
            continue
        coefficients[1 + target_index, :] = source_coefficients[1 + source_index, :]
        coefficients[1 + len(HARMONICS) + target_index, :] = source_coefficients[
            1 + len(source_harmonics) + source_index,
            :,
        ]
        mapped_count += 1

    if mapped_count == 0:
        raise ValueError("initial condition does not contain any harmonics used by this HB run")
    print(f"Loaded subharmonic initial condition from {input_path}, omega={omega:.10g}, mapped_harmonics={mapped_count}")
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
    model = MzsQzsModel()
    order = 2 * len(HARMONICS) + 1
    initial_condition = Path(getattr(args, "initial_condition", DEFAULT_INITIAL_CONDITION))
    initial_coefficients, init_omega = load_initial_coefficients(initial_condition, model, order)
    config = build_config(args, init_omega)
    result = ContinuationSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args.output)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
