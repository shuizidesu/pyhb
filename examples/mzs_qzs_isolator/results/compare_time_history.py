from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb.harmonics import generate_hb_items
from examples.mzs_qzs_isolator.autodiff_model import MzsQzsAutodiffModel
from examples.mzs_qzs_isolator.run_autodiff_arc import DEFAULT_PLOT_DOFS, FREQUENCY_RESOLUTION, HARMONICS


DEFAULT_INPUT = Path(__file__).resolve().parent / "autodiff_arc.npz"
DEFAULT_OUTPUT_FIG = Path(__file__).resolve().parent / "time_history_compare.png"
DEFAULT_OUTPUT_DATA = Path(__file__).resolve().parent / "time_history_compare.npz"
DEFAULT_POINTS_PER_PERIOD = 2048
DEFAULT_PERIODS = 30
INTEGRATION_METHODS = ("DOP853", "RK45", "Radau", "BDF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MZS-QZS HB and ODE time histories at one continuation point.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--index", type=int, default=130) # 130
    parser.add_argument("--periods", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--points-per-period", type=int, default=DEFAULT_POINTS_PER_PERIOD)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_OUTPUT_DATA)
    parser.add_argument("--method", choices=INTEGRATION_METHODS, default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def normalize_index(index: int, point_count: int) -> int:
    normalized = index + point_count if index < 0 else index
    if normalized < 0 or normalized >= point_count:
        raise IndexError(f"index {index} is out of range for {point_count} continuation points")
    return normalized


def validate_dofs(dofs: tuple[int, ...], n_dof: int) -> tuple[int, ...]:
    normalized = tuple(int(dof) for dof in dofs)
    if len(set(normalized)) != len(normalized):
        raise ValueError("dofs must contain unique DOFs")
    if any(dof < 0 or dof >= n_dof for dof in normalized):
        raise ValueError("dofs contains an out-of-range DOF")
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
    theta = np.arange(interval_count + 1, dtype=np.float64) * (period / points_per_period)
    hb_item, hb_item_dtheta, _ = generate_hb_items(theta, HARMONICS)
    return theta, hb_item @ coefficients, hb_item_dtheta @ coefficients


def mzs_qzs_nonlinear_force(
    model: MzsQzsAutodiffModel,
    displacement: NDArray[np.float64],
    phase_velocity: NDArray[np.float64],
    omega: float,
) -> NDArray[np.float64]:
    parameters = model.parameters
    x = float(displacement[0])
    dx_dtheta = float(phase_velocity[0])

    mu1 = float(parameters.mu1)
    mu2 = float(parameters.mu2)
    mu3 = float(parameters.mu3)
    mu4 = float(parameters.mu4)
    mu5 = float(parameters.mu5)
    lambda1 = float(parameters.lambda1)
    zeta1 = float(parameters.zeta1)
    zeta2 = float(parameters.zeta2)
    zeta3 = float(parameters.zeta3)

    mu2_sq = mu2**2
    mu5_sq = mu5**2
    upper_denominator = (mu3 + x) ** 2 + mu2_sq
    middle_denominator = mu5_sq + x**2
    lower_denominator = (mu3 - x) ** 2 + mu2_sq

    nonlinear_damping = (
        8.0 * zeta1 * mu2_sq / upper_denominator**2
        + 8.0 * zeta2 * mu5_sq / middle_denominator**2
        + 8.0 * zeta3 * mu2_sq / lower_denominator**2
    )
    damping_force = float(omega) * nonlinear_damping * dx_dtheta
    restoring_force = (
        -(1.0 + mu1) * (mu3 + x) / np.sqrt(upper_denominator)
        - (1.0 + mu1) * (x - mu3) / np.sqrt(lower_denominator)
        - lambda1 * mu4 * x / np.sqrt(middle_denominator)
    )
    nonlinear = np.zeros(model.n_dof, dtype=np.float64)
    nonlinear[0] = damping_force + restoring_force
    return nonlinear


def forcing_at_theta(model: MzsQzsAutodiffModel, theta: float, omega: float) -> NDArray[np.float64]:
    force = np.zeros(model.n_dof, dtype=np.float64)
    for term in model.forcing_terms(np.asarray([theta], dtype=np.float64)):
        force += (float(omega) ** float(term.omega_power)) * term.samples[0]
    return force


def make_mzs_qzs_rhs(model: MzsQzsAutodiffModel, omega: float):
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    mass_lu = lu_factor((omega**2) * model.mass)

    def rhs(theta: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        displacement = state[: model.n_dof]
        phase_velocity = state[model.n_dof :]
        nonlinear = mzs_qzs_nonlinear_force(model, displacement, phase_velocity, omega)
        rhs_force = (
            forcing_at_theta(model, theta, omega)
            - omega * (model.damping @ phase_velocity)
            - model.stiffness @ displacement
            - nonlinear
        )
        phase_acceleration = lu_solve(mass_lu, rhs_force)
        return np.concatenate((phase_velocity, phase_acceleration))

    return rhs


def integrate_ode_time_history(
    model: MzsQzsAutodiffModel,
    omega: float,
    theta: NDArray[np.float64],
    initial_displacement: NDArray[np.float64],
    initial_phase_velocity: NDArray[np.float64],
    method: Literal["DOP853", "RK45", "Radau", "BDF"],
    rtol: float,
    atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    state0 = np.concatenate((initial_displacement, initial_phase_velocity))
    solution = solve_ivp(
        make_mzs_qzs_rhs(model, omega),
        (float(theta[0]), float(theta[-1])),
        state0,
        method=method,
        t_eval=theta,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    state = np.asarray(solution.y, dtype=np.float64).T
    return state[:, : model.n_dof], state[:, model.n_dof :]


def save_plot(
    theta: NDArray[np.float64],
    x_hb: NDArray[np.float64],
    v_hb: NDArray[np.float64],
    x_ode: NDArray[np.float64],
    v_ode: NDArray[np.float64],
    dofs: tuple[int, ...],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0))
    for dof in dofs:
        axes[0].plot(theta, x_hb[:, dof], label=f"HB DOF {dof + 1}")
        axes[0].plot(theta, x_ode[:, dof], "--", label=f"ODE DOF {dof + 1}")
        axes[1].plot(theta, v_hb[:, dof], label=f"HB DOF {dof + 1}")
        axes[1].plot(theta, v_ode[:, dof], "--", label=f"ODE DOF {dof + 1}")
        axes[2].plot(x_hb[:, dof], v_hb[:, dof], label=f"HB DOF {dof + 1}")
        axes[2].plot(x_ode[:, dof], v_ode[:, dof], "--", label=f"ODE DOF {dof + 1}")
    axes[0].set_ylabel("displacement")
    axes[1].set_ylabel("phase velocity")
    axes[2].set_xlabel("displacement")
    axes[2].set_ylabel("phase velocity")
    axes[1].set_xlabel("theta")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved time-history comparison plot to {output}")


def save_data(
    output: Path,
    theta: NDArray[np.float64],
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
        theta=theta,
        tau=theta / float(omega),
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
    theta, x_hb, v_hb = reconstruct_hb_time_history(coefficients, args.periods, args.points_per_period)
    model = MzsQzsAutodiffModel()
    dofs = validate_dofs(tuple(int(dof) for dof in getattr(args, "dofs", DEFAULT_PLOT_DOFS)), model.n_dof)
    x_ode, v_ode = integrate_ode_time_history(
        model,
        omega,
        theta,
        x_hb[0],
        v_hb[0],
        args.method,
        args.rtol,
        args.atol,
    )
    max_abs_error_x = float(np.max(np.abs(x_ode - x_hb)))
    max_abs_error_v = float(np.max(np.abs(v_ode - v_hb)))
    print(
        "Time-history comparison, "
        f"index={selected_index}, omega={omega:.10g}, method={args.method}, "
        f"max|dx|={max_abs_error_x:.6e}, max|dv|={max_abs_error_v:.6e}"
    )
    save_plot(theta, x_hb, v_hb, x_ode, v_ode, dofs, args.output)
    if args.data_output is not None:
        save_data(
            args.data_output,
            theta,
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
