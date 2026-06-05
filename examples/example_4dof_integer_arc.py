from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    ForcingTerm,
    LinearOperatorTerm,
    LocalNonlinearJacobianTerm,
    SecondOrderTimeModel,
)


DEFAULT_OUTPUT = Path("results/example_4dof_integer_arc.npz")
DEFAULT_MAX_STEPS = 3000


@dataclass(frozen=True)
class FourDofSystem:
    n_dof: int
    mass: NDArray[np.float64]
    damping: NDArray[np.float64]
    rotational_damping: NDArray[np.float64]
    stiffness: NDArray[np.float64]
    excitation: float
    nonlinear_scale: float


def default_four_dof_system() -> FourDofSystem:
    n_dof = 4
    mass_value = 20.0
    l1 = 0.25
    l2 = 0.5
    eccentricity_length = 0.03e-3
    damping_ratio = 0.015
    polar_inertia = 0.144
    diametral_inertia = 0.072
    stiffness_value = 1.5711e6
    cubic_stiffness = 2.8e12
    static_deflection = mass_value * 9.8 / stiffness_value
    damping_value = 2.0 * mass_value * damping_ratio

    w1 = 2.0 * stiffness_value / mass_value
    w2 = (l1**2 + l2**2) * stiffness_value / diametral_inertia
    a1 = stiffness_value * (l2 - l1) / mass_value / (l1 + l2)
    b1 = stiffness_value * (l2**2 - l1**2) / diametral_inertia
    a2 = 2.0 * damping_value / mass_value
    a3 = damping_value * (l2 - l1) / mass_value / (l1 + l2)
    b2 = damping_value * (l1**2 + l2**2) / diametral_inertia
    b3 = polar_inertia / diametral_inertia
    b4 = damping_value * (l1**2 - l2**2) / diametral_inertia
    nonlinear_scale = 2.0 * static_deflection**2 * cubic_stiffness / mass_value
    excitation = eccentricity_length / static_deflection

    mass = np.eye(4, dtype=np.float64)
    damping = np.array(
        [
            [a2, 0.0, 0.0, a3],
            [0.0, a2, a3, 0.0],
            [0.0, b4, b2, 0.0],
            [b4, 0.0, 0.0, b2],
        ],
        dtype=np.float64,
    )
    rotational_damping = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, b3],
            [0.0, 0.0, -b3, 0.0],
        ],
        dtype=np.float64,
    )
    stiffness = np.array(
        [
            [w1, 0.0, 0.0, a1],
            [0.0, w1, a1, 0.0],
            [0.0, b1, w2, 0.0],
            [b1, 0.0, 0.0, w2],
        ],
        dtype=np.float64,
    )
    return FourDofSystem(n_dof, mass, damping, rotational_damping, stiffness, excitation, nonlinear_scale)


class FourDofIntegerModel(SecondOrderTimeModel):
    def __init__(self, system: FourDofSystem | None = None) -> None:
        self.system = system or default_four_dof_system()

    @property
    def n_dof(self) -> int:
        return self.system.n_dof

    @property
    def nonlinear_force_dofs(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, int]:
        return (0, 1)

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.system.mass, "ddx", 0.0),
            LinearOperatorTerm(self.system.damping, "dx", 0.0),
            LinearOperatorTerm(self.system.rotational_damping, "dx", -1.0),
            LinearOperatorTerm(self.system.stiffness, "x", -2.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, 0] = self.system.excitation * np.cos(t)
        force[:, 1] = self.system.excitation * np.sin(t)
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return (self.system.nonlinear_scale / parameter**2) * local_x**3

    def local_nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[LocalNonlinearJacobianTerm, ...]:
        factor = 3.0 * self.system.nonlinear_scale / parameter**2
        return (
            LocalNonlinearJacobianTerm(0, "x", 0, factor * local_x[:, 0] ** 2),
            LocalNonlinearJacobianTerm(1, "x", 1, factor * local_x[:, 1] ** 2),
        )

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return (-2.0 * self.system.nonlinear_scale / parameter**3) * local_x**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4DOF integer-harmonic arc-length example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=2**11)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npy-output", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--ode-check", action="store_true")
    parser.add_argument("--save-coeff-history", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        sample_fft=args.sample_fft,
        seed=args.seed,
        max_steps=args.max_steps,
        enable_ode_check=args.ode_check,
        progress_callback=print,
    )


def save_result(result: ContinuationResult, args: argparse.Namespace) -> None:
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
        save_plot(result, args.output.with_suffix(".png"))
    if result.ode_check is not None:
        print(
            "ODE check finished, "
            f"samples = {result.ode_check.t.size}, "
            f"last t = {result.ode_check.t[-1]:.6g}"
        )


def save_plot(result: ContinuationResult, plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(result.omega_list, result.amplitudes)
    ax.set_xlabel("omega")
    ax.set_ylabel("RMS amplitude")
    ax.legend([f"DOF {idx}" for idx in range(1, result.amplitudes.shape[1] + 1)], loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def run_from_args(args: argparse.Namespace) -> None:
    result = ContinuationSolver(FourDofIntegerModel(), build_config(args)).run()
    save_result(result, args)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
