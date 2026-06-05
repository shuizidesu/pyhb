from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ega_ihb import (
    ContinuationConfig,
    ContinuationResult,
    ContinuationSolver,
)
from examples.aeroengine_condensed_model import AeroEngineRotorModel


DEFAULT_MATRIX_PATH = Path(__file__).resolve().parent / "data" / "aero_engine_system_parameter_matrix.mat"
DEFAULT_OUTPUT = Path("results/example_aeroengine_arc.npz")
DEFAULT_MAX_STEPS = 400
DEFAULT_SAMPLE_FFT = 2 ** 11
FREQUENCY_RESOLUTION = 0.1
INIT_OMEGA = 145.0
MAX_EPOCH = 25
INITIAL_SCALE = 1e-5
RES_TOLERANCE = 1e-9
DELTA_TOLERANCE = 1e-12
Q_SCALE = 1e-4
OMEGA_SCALE = 100


def harmonic_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + step * index, 10) for index in range(count))


HARMONICS = harmonic_range(0.5, 3.1, 0.1)
# HARMONICS = (1.0, 1.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the aero-engine rotor EGA-IHB arc-length example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npy-output", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--ode-check", action="store_true")
    parser.add_argument("--save-coeff-history", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        frequency_resolution=FREQUENCY_RESOLUTION,
        strict_fft_grid=True,
        seed=args.seed,
        init_omega=INIT_OMEGA,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        s_initial=0.1,
        s_max=0.3,
        s_min=1e-9,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        max_steps=args.max_steps,
        enable_ode_check=args.ode_check,
        progress_callback=print,
    )


def build_initial_coefficients(model: AeroEngineRotorModel, order: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.random((order, model.n_dof), dtype=np.float64) * INITIAL_SCALE


def default_plot_dofs(model: AeroEngineRotorModel) -> tuple[int, int, int, int]:
    return (model.bearing_ix, model.bearing_iy, model.bearing_ox, model.bearing_oy)


def save_result(result: ContinuationResult, args: argparse.Namespace, plot_dofs: tuple[int, int, int, int]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curve_table = (
        np.column_stack((result.omega_list, result.amplitudes))
        if result.omega_list.size
        else np.empty((0, result.coefficients.shape[1] + 1), dtype=np.float64)
    )
    payload = {
        "omega_list": result.omega_list,
        "amplitudes": result.amplitudes,
        "curve_table": curve_table,
        "coefficients": result.coefficients,
        "omega": np.array(result.omega),
        "harmonics": result.harmonics,
        "nonlinear_harmonics": result.nonlinear_harmonics,
        "frequency_resolution": np.array(result.frequency_resolution),
        "period": np.array(result.period),
        "plot_dofs": np.asarray(plot_dofs, dtype=np.int64),
    }
    if args.save_coeff_history:
        payload["coefficient_history"] = result.coefficient_history
        payload["parameter_history"] = result.parameter_history
    np.savez(args.output, **payload)
    print(f"Saved result to {args.output}")

    table_output = args.npy_output or args.output.with_suffix(".npy")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(table_output, curve_table)
    print(f"Saved curve table to {table_output}")

    if args.plot:
        save_plot(result, args.output.with_suffix(".png"), plot_dofs)
    if result.ode_check is not None:
        print(
            "ODE check finished, "
            f"samples = {result.ode_check.t.size}, "
            f"last t = {result.ode_check.t[-1]:.6g}"
        )


def save_plot(result: ContinuationResult, plot_path: Path, plot_dofs: tuple[int, int, int, int]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for dof in plot_dofs:
        ax.plot(result.omega_list, result.amplitudes[:, dof], label=f"DOF {dof + 1}")
    ax.set_xlabel("omega")
    ax.set_ylabel("RMS amplitude")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def run_from_args(args: argparse.Namespace) -> None:
    model = AeroEngineRotorModel()
    config = build_config(args)
    order = 2 * len(HARMONICS) + 1
    initial_coefficients = build_initial_coefficients(model, order, args.seed)
    result = ContinuationSolver(model, config).run(initial_coefficients=initial_coefficients)
    save_result(result, args, default_plot_dofs(model))


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
