from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb.harmonics import generate_hb_items
from examples.aeroengine.model import AeroEngineRotorModel
from examples.aeroengine.run_arc import DEFAULT_PLOT_DOFS, FREQUENCY_RESOLUTION, HARMONICS


DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc_0.npz"
DEFAULT_OUTPUT_FIG = Path(__file__).resolve().parent / "results" / "time_history_compare.png"
DEFAULT_POINTS_PER_PERIOD = 2048
DEFAULT_PERIODS = 5
INTEGRATION_METHODS = ("DOP853", "RK45", "Radau", "BDF")
DEFAULT_PLOT_DOFS = (34, 176)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare aero-engine HB and ODE time histories at one continuation point.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--index", type=int, default=54)
    parser.add_argument("--periods", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--points-per-period", type=int, default=DEFAULT_POINTS_PER_PERIOD)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--data-output", type=Path, default=None)
    parser.add_argument("--method", choices=INTEGRATION_METHODS, default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def normalize_index(index: int, point_count: int) -> int:
    normalized = index + point_count if index < 0 else index
    if normalized < 0 or normalized >= point_count:
        raise IndexError(f"index {index} is out of range for {point_count} continuation points")
    return normalized


def load_selected_solution(input_path: Path, index: int) -> tuple[int, float, NDArray[np.float64]]:
    with np.load(input_path) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    selected_index = normalize_index(index, int(parameter_history.size))
    return selected_index, float(parameter_history[selected_index]), coefficient_history[selected_index]


def reconstruct_hb_time_history(
    coefficients: NDArray[np.float64],
    periods: int,
    points_per_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if periods <= 0:
        raise ValueError("periods must be positive")
    if points_per_period <= 0:
        raise ValueError("points_per_period must be positive")
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    interval_count = int(periods * points_per_period)
    tau = np.arange(interval_count + 1, dtype=np.float64) * (period / points_per_period)
    hb_item, hb_item_dt, _ = generate_hb_items(tau, HARMONICS)
    return tau, hb_item @ coefficients, hb_item_dt @ coefficients


def aeroengine_nonlinear_force(
    model: AeroEngineRotorModel,
    tau: float,
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    omega: float,
) -> NDArray[np.float64]:
    nonlinear = np.zeros(model.n_dof, dtype=np.float64)
    coordinate_dofs = tuple(model.nonlinear_coordinate_dofs)
    force_dofs = tuple(model.nonlinear_force_dofs)
    if not coordinate_dofs or not force_dofs:
        return nonlinear
    local_x = displacement[list(coordinate_dofs)].reshape(1, -1)
    local_dx = velocity[list(coordinate_dofs)].reshape(1, -1)
    local_ddx = np.zeros_like(local_x)
    local_force = model.local_nonlinear_force(
        np.asarray([tau], dtype=np.float64),
        local_x,
        local_dx,
        local_ddx,
        omega,
    )[0]
    nonlinear[list(force_dofs)] = local_force
    return nonlinear


def make_aeroengine_rhs(model: AeroEngineRotorModel, omega: float):
    mass_lu = lu_factor((omega**2) * model.mass)

    def rhs(tau: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        displacement = state[: model.n_dof]
        velocity = state[model.n_dof :]
        base_force = model._base_unbalance_force(np.asarray([tau], dtype=np.float64))[0]
        nonlinear = aeroengine_nonlinear_force(model, tau, displacement, velocity, omega)
        rhs_force = (
            (omega**2) * base_force
            - omega * (model.damping @ velocity)
            - (omega**2) * (model.gyroscopic @ velocity)
            - model.stiffness @ displacement
            - nonlinear
        )
        acceleration = lu_solve(mass_lu, rhs_force)
        return np.concatenate((velocity, acceleration))

    return rhs


def integrate_ode_time_history(
    model: AeroEngineRotorModel,
    omega: float,
    tau: NDArray[np.float64],
    initial_displacement: NDArray[np.float64],
    initial_velocity: NDArray[np.float64],
    method: Literal["DOP853", "RK45", "Radau", "BDF"],
    rtol: float,
    atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    state0 = np.concatenate((initial_displacement, initial_velocity))
    solution = solve_ivp(
        make_aeroengine_rhs(model, omega),
        (float(tau[0]), float(tau[-1])),
        state0,
        method=method,
        t_eval=tau,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    state = np.asarray(solution.y, dtype=np.float64).T
    return state[:, : model.n_dof], state[:, model.n_dof :]


def save_plot(
    tau: NDArray[np.float64],
    x_hb: NDArray[np.float64],
    x_ode: NDArray[np.float64],
    dofs: tuple[int, ...],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for dof in dofs:
        ax.plot(tau, x_hb[:, dof], label=f"HB DOF {dof + 1}")
        ax.plot(tau, x_ode[:, dof], "--", label=f"ODE DOF {dof + 1}")
    ax.set_xlabel("tau")
    ax.set_ylabel("displacement")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=600)
    plt.close(fig)
    print(f"Saved time-history comparison plot to {output}")


def save_data(
    output: Path,
    tau: NDArray[np.float64],
    omega: float,
    selected_index: int,
    dofs: tuple[int, ...],
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
        tau=tau,
        omega=np.asarray(omega, dtype=np.float64),
        selected_index=np.asarray(selected_index, dtype=np.int64),
        dofs=np.asarray(dofs, dtype=np.int64),
        x_hb=x_hb,
        v_hb=v_hb,
        x_ode=x_ode,
        v_ode=v_ode,
        max_abs_error_x=np.asarray(max_abs_error_x, dtype=np.float64),
        max_abs_error_v=np.asarray(max_abs_error_v, dtype=np.float64),
    )
    print(f"Saved time-history comparison data to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    selected_index, omega, coefficients = load_selected_solution(args.input, args.index)
    tau, x_hb, v_hb = reconstruct_hb_time_history(coefficients, args.periods, args.points_per_period)
    model = AeroEngineRotorModel()
    x_ode, v_ode = integrate_ode_time_history(
        model,
        omega,
        tau,
        x_hb[0],
        v_hb[0],
        args.method,
        args.rtol,
        args.atol,
    )
    dofs = tuple(int(dof) for dof in args.dofs)
    max_abs_error_x = float(np.max(np.abs(x_ode - x_hb)))
    max_abs_error_v = float(np.max(np.abs(v_ode - v_hb)))
    print(
        "Time-history comparison, "
        f"index={selected_index}, omega={omega:.10g}, method={args.method}, "
        f"max|dx|={max_abs_error_x:.6e}, max|dv|={max_abs_error_v:.6e}"
    )
    save_plot(tau, x_hb, x_ode, dofs, args.output)
    if args.data_output is not None:
        save_data(
            args.data_output,
            tau,
            omega,
            selected_index,
            dofs,
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
