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
from examples.vanderpol.model import VanderpolModel


DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "autodiff_free_frequency.npz"
DEFAULT_OUTPUT_FIG = Path(__file__).resolve().parent / "results" / "time_history_compare.png"
DEFAULT_POINTS_PER_PERIOD = 2048
DEFAULT_PERIODS = 5
INTEGRATION_METHODS = ("DOP853", "RK45", "Radau", "BDF")
HARMONICS = tuple(float(value) for value in range(1, 51))
FREQUENCY_RESOLUTION = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare van der Pol HB and ODE time histories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--index", type=int, default=40)
    parser.add_argument("--periods", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--points-per-period", type=int, default=DEFAULT_POINTS_PER_PERIOD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--data-output", type=Path, default=None)
    parser.add_argument("--method", choices=INTEGRATION_METHODS, default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    return parser.parse_args()


def normalize_index(index: int, point_count: int) -> int:
    normalized = index + point_count if index < 0 else index
    if normalized < 0 or normalized >= point_count:
        raise IndexError(f"index {index} is out of range for {point_count} continuation points")
    return normalized


def load_selected_solution(input_path: Path, index: int) -> tuple[int, float, float, NDArray[np.float64]]:
    with np.load(input_path) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        omega_history = np.asarray(data["omega_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    selected_index = normalize_index(index, int(parameter_history.size))
    return (
        selected_index,
        float(parameter_history[selected_index]),
        float(omega_history[selected_index]),
        coefficient_history[selected_index],
    )


def reconstruct_hb_time_history(
    coefficients: NDArray[np.float64],
    omega: float,
    periods: int,
    points_per_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if periods <= 0:
        raise ValueError("periods must be positive")
    if points_per_period <= 0:
        raise ValueError("points_per_period must be positive")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    interval_count = int(periods * points_per_period)
    tau = np.arange(interval_count + 1, dtype=np.float64) * (period / points_per_period)
    hb_item, hb_item_dt, _ = generate_hb_items(tau, HARMONICS)
    displacement = hb_item @ coefficients
    velocity_tau = hb_item_dt @ coefficients
    return tau / float(omega), tau, displacement, float(omega) * velocity_tau


def make_vanderpol_rhs(parameter: float, lambda_value: float):
    def rhs(time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        displacement = state[0]
        velocity = state[1]
        acceleration = float(parameter) * (float(lambda_value) - displacement**2) * velocity - displacement
        return np.asarray([velocity, acceleration], dtype=np.float64)

    return rhs


def integrate_ode_time_history(
    parameter: float,
    lambda_value: float,
    time: NDArray[np.float64],
    initial_displacement: float,
    initial_velocity: float,
    method: Literal["DOP853", "RK45", "Radau", "BDF"],
    rtol: float,
    atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    solution = solve_ivp(
        make_vanderpol_rhs(parameter, lambda_value),
        (float(time[0]), float(time[-1])),
        np.asarray([initial_displacement, initial_velocity], dtype=np.float64),
        method=method,
        t_eval=time,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    state = np.asarray(solution.y, dtype=np.float64).T
    return state[:, 0:1], state[:, 1:2]


def save_plot(
    time: NDArray[np.float64],
    x_hb: NDArray[np.float64],
    v_hb: NDArray[np.float64],
    x_ode: NDArray[np.float64],
    v_ode: NDArray[np.float64],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_displacement, ax_velocity, ax_phase) = plt.subplots(3, 1, figsize=(7.5, 9.0))
    ax_displacement.plot(time, x_hb[:, 0], label="HB")
    ax_displacement.plot(time, x_ode[:, 0], "--", label="ODE")
    ax_displacement.set_ylabel("displacement")
    ax_displacement.legend(loc="best")
    ax_displacement.grid(True, alpha=0.3)

    ax_velocity.plot(time, v_hb[:, 0], label="HB")
    ax_velocity.plot(time, v_ode[:, 0], "--", label="ODE")
    ax_velocity.set_xlabel("time")
    ax_velocity.set_ylabel("velocity")
    ax_velocity.legend(loc="best")
    ax_velocity.grid(True, alpha=0.3)

    ax_phase.plot(x_hb[:, 0], v_hb[:, 0], label="HB")
    ax_phase.plot(x_ode[:, 0], v_ode[:, 0], "--", label="ODE")
    ax_phase.set_xlabel("displacement")
    ax_phase.set_ylabel("velocity")
    ax_phase.legend(loc="best")
    ax_phase.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=600)
    plt.close(fig)
    print(f"Saved time-history comparison plot to {output}")


def save_data(
    output: Path,
    time: NDArray[np.float64],
    tau: NDArray[np.float64],
    parameter: float,
    omega: float,
    selected_index: int,
    x_hb: NDArray[np.float64],
    v_hb: NDArray[np.float64],
    x_ode: NDArray[np.float64],
    v_ode: NDArray[np.float64],
    max_abs_error_x: float,
    max_abs_error_v: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        time=time,
        tau=tau,
        parameter=np.asarray(parameter, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        selected_index=np.asarray(selected_index, dtype=np.int64),
        x_hb=x_hb,
        v_hb=v_hb,
        x_ode=x_ode,
        v_ode=v_ode,
        max_abs_error_x=np.asarray(max_abs_error_x, dtype=np.float64),
        max_abs_error_v=np.asarray(max_abs_error_v, dtype=np.float64),
    )
    print(f"Saved time-history comparison data to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    selected_index, parameter, omega, coefficients = load_selected_solution(args.input, args.index)
    time, tau, x_hb, v_hb = reconstruct_hb_time_history(
        coefficients,
        omega,
        args.periods,
        args.points_per_period,
    )
    model = VanderpolModel()
    x_ode, v_ode = integrate_ode_time_history(
        parameter,
        model.parameters.lambda_value,
        time,
        float(x_hb[0, 0]),
        float(v_hb[0, 0]),
        args.method,
        args.rtol,
        args.atol,
    )
    max_abs_error_x = float(np.max(np.abs(x_ode - x_hb)))
    max_abs_error_v = float(np.max(np.abs(v_ode - v_hb)))
    print(
        "Time-history comparison, "
        f"index={selected_index}, epsilon={parameter:.10g}, omega={omega:.10g}, method={args.method}, "
        f"max|dx|={max_abs_error_x:.6e}, max|dv|={max_abs_error_v:.6e}"
    )
    save_plot(time, x_hb, v_hb, x_ode, v_ode, args.output)
    if args.data_output is not None:
        save_data(
            args.data_output,
            time,
            tau,
            parameter,
            omega,
            selected_index,
            x_hb,
            v_hb,
            x_ode,
            v_ode,
            max_abs_error_x,
            max_abs_error_v,
        )


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
