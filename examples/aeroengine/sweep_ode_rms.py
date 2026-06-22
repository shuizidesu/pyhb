from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb.harmonics import generate_hb_items
from examples.aeroengine.compare_time_history import make_aeroengine_rhs, reconstruct_hb_time_history
from examples.aeroengine.model import AeroEngineRotorModel
from examples.aeroengine.run_arc import DEFAULT_PLOT_DOFS, FREQUENCY_RESOLUTION, HARMONICS


DEFAULT_OUTPUT_FIG = Path(__file__).resolve().parent / "results" / "ode_sweep_rms.png"
DEFAULT_DATA_OUTPUT = Path(__file__).resolve().parent / "results" / "ode_sweep_rms.npz"
INTEGRATION_METHODS = ("DOP853", "RK45", "Radau", "BDF")
DEFAULT_HB_INPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc_0.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an aero-engine ODE speed sweep and compute steady RMS response.")
    parser.add_argument("--omega-start", type=float, default=200.0)
    parser.add_argument("--omega-end", type=float, default=145.0)
    parser.add_argument("--omega-step", type=float, default=-1.0)
    parser.add_argument("--settle-periods", type=int, default=8)
    parser.add_argument("--rms-periods", type=int, default=1)
    parser.add_argument("--points-per-period", type=int, default=512)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--method", choices=INTEGRATION_METHODS, default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--initial-hb-input", type=Path, default=None)
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--hb-input", type=Path, default=DEFAULT_HB_INPUT)
    return parser.parse_args()


def build_omega_sequence(omega_start: float, omega_end: float, omega_step: float) -> NDArray[np.float64]:
    if omega_step == 0.0:
        raise ValueError("omega_step must be nonzero")
    if omega_step > 0.0 and omega_start > omega_end:
        raise ValueError("positive omega_step requires omega_start <= omega_end")
    if omega_step < 0.0 and omega_start < omega_end:
        raise ValueError("negative omega_step requires omega_start >= omega_end")
    direction = 1.0 if omega_step > 0.0 else -1.0
    values = [float(omega_start)]
    current = float(omega_start)
    tolerance = max(1.0, abs(omega_end), abs(omega_step)) * 1e-12
    while direction * (current + omega_step - omega_end) < -tolerance:
        current += omega_step
        values.append(float(current))
    if abs(values[-1] - omega_end) > tolerance:
        values.append(float(omega_end))
    else:
        values[-1] = float(omega_end)
    return np.asarray(values, dtype=np.float64)


def validate_sweep_settings(settle_periods: int, rms_periods: int, points_per_period: int) -> None:
    if settle_periods <= 0:
        raise ValueError("settle_periods must be positive")
    if rms_periods <= 0:
        raise ValueError("rms_periods must be positive")
    if points_per_period <= 0:
        raise ValueError("points_per_period must be positive")


def rms_from_tail(
    displacement: NDArray[np.float64],
    dofs: tuple[int, ...],
    rms_periods: int,
    points_per_period: int,
) -> NDArray[np.float64]:
    tail_count = int(rms_periods * points_per_period) + 1
    tail = displacement[-tail_count:, list(dofs)]
    return np.sqrt(np.mean(tail * tail, axis=0))


def load_hb_initial_state(path: Path, index: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with np.load(path) as data:
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    point_count = coefficient_history.shape[0]
    selected_index = index + point_count if index < 0 else index
    if selected_index < 0 or selected_index >= point_count:
        raise IndexError(f"initial_index {index} is out of range for {point_count} continuation points")
    _, x_hb, v_hb = reconstruct_hb_time_history(coefficient_history[selected_index], periods=1, points_per_period=1)
    return x_hb[0], v_hb[0]


def integrate_one_speed(
    model: AeroEngineRotorModel,
    omega: float,
    initial_state: NDArray[np.float64],
    total_periods: int,
    points_per_period: int,
    method: Literal["DOP853", "RK45", "Radau", "BDF"],
    rtol: float,
    atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    interval_count = int(total_periods * points_per_period)
    tau = np.arange(interval_count + 1, dtype=np.float64) * (period / points_per_period)
    solution = solve_ivp(
        make_aeroengine_rhs(model, omega),
        (float(tau[0]), float(tau[-1])),
        initial_state,
        method=method,
        t_eval=tau,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed at omega={omega:.10g}: {solution.message}")
    state = np.asarray(solution.y, dtype=np.float64).T
    return tau, state[:, : model.n_dof], state[:, model.n_dof :]


def run_sweep(
    model: AeroEngineRotorModel,
    omega_values: NDArray[np.float64],
    dofs: tuple[int, ...],
    settle_periods: int,
    rms_periods: int,
    points_per_period: int,
    method: Literal["DOP853", "RK45", "Radau", "BDF"],
    rtol: float,
    atol: float,
    initial_state: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    validate_sweep_settings(settle_periods, rms_periods, points_per_period)
    if initial_state is None:
        current_state = np.zeros(2 * model.n_dof, dtype=np.float64)
    else:
        current_state = np.asarray(initial_state, dtype=np.float64).reshape(-1).copy()
        if current_state.shape != (2 * model.n_dof,):
            raise ValueError(f"initial_state must have shape {(2 * model.n_dof,)}, got {current_state.shape}")

    rms_history = np.empty((omega_values.size, len(dofs)), dtype=np.float64)
    final_state_history = np.empty((omega_values.size, 2 * model.n_dof), dtype=np.float64)
    success_flag = np.zeros(omega_values.size, dtype=np.bool_)
    total_periods = int(settle_periods + rms_periods)
    for index, omega in enumerate(omega_values, start=1):
        print(f"ODE sweep {index}/{omega_values.size}, omega={float(omega):.10g}")
        _, displacement, velocity = integrate_one_speed(
            model,
            float(omega),
            current_state,
            total_periods,
            points_per_period,
            method,
            rtol,
            atol,
        )
        rms_history[index - 1] = rms_from_tail(displacement, dofs, rms_periods, points_per_period)
        current_state = np.concatenate((displacement[-1], velocity[-1]))
        final_state_history[index - 1] = current_state
        success_flag[index - 1] = True
        print(f"  RMS max={float(np.max(rms_history[index - 1])):.6e}")
    return rms_history, final_state_history, success_flag


def compute_hb_rms(
    input_path: Path,
    dofs: tuple[int, ...],
    points_per_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with np.load(input_path) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    tau = np.arange(points_per_period, dtype=np.float64) * (period / points_per_period)
    hb_item, _, _ = generate_hb_items(tau, HARMONICS)
    selected_coefficients = coefficient_history[:, :, list(dofs)]
    response = np.einsum("to,sod->std", hb_item, selected_coefficients)
    return parameter_history, np.sqrt(np.mean(response * response, axis=1))


def save_plot(
    omega_values: NDArray[np.float64],
    rms_history: NDArray[np.float64],
    dofs: tuple[int, ...],
    output: Path,
    hb_omega: NDArray[np.float64] | None = None,
    hb_rms: NDArray[np.float64] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for column, dof in enumerate(dofs):
        ax.plot(omega_values, rms_history[:, column], "o-", label=f"ODE DOF {dof + 1}")
        if hb_omega is not None and hb_rms is not None:
            ax.plot(hb_omega, hb_rms[:, column], "--", label=f"HB DOF {dof + 1}")
    ax.set_xlabel("omega")
    ax.set_ylabel("RMS amplitude")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved ODE sweep RMS plot to {output}")


def save_data(
    output: Path,
    omega_values: NDArray[np.float64],
    rms_history: NDArray[np.float64],
    dofs: tuple[int, ...],
    final_state_history: NDArray[np.float64],
    success_flag: NDArray[np.bool_],
    settle_periods: int,
    rms_periods: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        omega=omega_values,
        rms_history=rms_history,
        dofs=np.asarray(dofs, dtype=np.int64),
        final_state_history=final_state_history,
        success_flag=success_flag,
        settle_periods=np.asarray(settle_periods, dtype=np.int64),
        rms_periods=np.asarray(rms_periods, dtype=np.int64),
    )
    print(f"Saved ODE sweep RMS data to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    omega_values = build_omega_sequence(args.omega_start, args.omega_end, args.omega_step)
    dofs = tuple(int(dof) for dof in args.dofs)
    model = AeroEngineRotorModel()
    initial_state = None
    if args.initial_hb_input is not None:
        initial_x, initial_v = load_hb_initial_state(args.initial_hb_input, args.initial_index)
        initial_state = np.concatenate((initial_x, initial_v))
    rms_history, final_state_history, success_flag = run_sweep(
        model,
        omega_values,
        dofs,
        args.settle_periods,
        args.rms_periods,
        args.points_per_period,
        args.method,
        args.rtol,
        args.atol,
        initial_state,
    )
    hb_omega = None
    hb_rms = None
    if args.hb_input is not None:
        hb_omega, hb_rms = compute_hb_rms(args.hb_input, dofs, args.points_per_period)
    save_plot(omega_values, rms_history, dofs, args.output, hb_omega, hb_rms)
    if args.data_output is not None:
        save_data(
            args.data_output,
            omega_values,
            rms_history,
            dofs,
            final_state_history,
            success_flag,
            args.settle_periods,
            args.rms_periods,
        )


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
