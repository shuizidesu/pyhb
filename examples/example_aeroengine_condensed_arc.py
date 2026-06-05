from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ega_ihb import (
    CondensedContinuationConfig,
    CondensedContinuationResult,
    CondensedContinuationSolver,
    ForcingTerm,
    LinearOperatorTerm,
    NonlinearJacobianTerm,
    build_full_fft_nonlinear_harmonics,
)


def harmonic_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + step * index, 10) for index in range(count))


HARMONICS = harmonic_range(0.5, 3.1, 0.1)

DEFAULT_MATRIX_PATH = Path(__file__).resolve().parent / "data" / "aero_engine_system_parameter_matrix.mat"
DEFAULT_OUTPUT = Path("results/example_aeroengine_condensed_arc.npz")
DEFAULT_SAMPLE_FFT = 2 ** 11
DEFAULT_MAX_STEPS = 600
# HARMONICS = (0.2, 0.5, 0.6, 1.0, 1.2, 2.0, 2.2, 2.4)
FREQUENCY_RESOLUTION = 0.1
INIT_OMEGA = 145.0
INITIAL_SCALE = 1e-6
MAX_EPOCH = 25
RES_TOLERANCE = 1e-9
DELTA_TOLERANCE = 1e-12
Q_SCALE = 1e-4
OMEGA_SCALE = 200
LOOP_SWITCH_ENABLED = True


@dataclass(frozen=True)
class AeroEngineParameters:
    speed_ratio: float = 1.2
    lp_disk_e: tuple[float, ...] = (20e-6, 20e-6, 30e-6)
    hp_disk_e: tuple[float, ...] = (0e-6, 100e-6, 0e-6, 0e-6, 0e-6, 0e-6, 134e-6, 10e-6)
    lp_disk_m: tuple[float, ...] = (50.292, 51.1704, 73.115)
    hp_disk_m: tuple[float, ...] = (38.2962, 12.1848, 9.1364, 8.6804, 8.2476, 8.2278, 9.4462, 78.436)
    lp_disk_loc: tuple[int, ...] = (2, 3, 19)
    hp_disk_loc: tuple[int, ...] = (23, 24, 25, 26, 27, 28, 29, 33)
    bearing_node_i: int = 18
    bearing_node_o: int = 35
    bearing_di: float = 118.94e-3
    bearing_do: float = 164.064e-3
    bearing_nb: int = 28
    bearing_clearance: float = 2e-6
    bearing_kb: float = 2.5e8


class AeroEngineRotorModel:
    def __init__(
        self,
        matrix_path: str | Path = DEFAULT_MATRIX_PATH,
        parameters: AeroEngineParameters | None = None,
    ) -> None:
        self.parameters = parameters or AeroEngineParameters()
        matrix_data = loadmat(matrix_path)
        self.mass = np.asarray(matrix_data["M"], dtype=np.float64)
        self.stiffness = np.asarray(matrix_data["K"], dtype=np.float64)
        self.damping = np.asarray(matrix_data["C"], dtype=np.float64)
        self.gyroscopic = np.asarray(matrix_data["J"], dtype=np.float64)
        self._n_dof = int(self.mass.shape[0])
        if (
            self.mass.shape != self.stiffness.shape
            or self.mass.shape != self.damping.shape
            or self.mass.shape != self.gyroscopic.shape
        ):
            raise ValueError("M, C, J, and K must have the same shape")
        if self.mass.shape[0] != self.mass.shape[1]:
            raise ValueError("system matrices must be square")

        parameters = self.parameters
        self.speed_ratio = float(parameters.speed_ratio)
        self.omega_c = (parameters.bearing_di + self.speed_ratio * parameters.bearing_do) / (
            parameters.bearing_di + parameters.bearing_do
        )
        self.lp_disk_e = np.asarray(parameters.lp_disk_e, dtype=np.float64)
        self.hp_disk_e = np.asarray(parameters.hp_disk_e, dtype=np.float64)
        self.lp_disk_m = np.asarray(parameters.lp_disk_m, dtype=np.float64)
        self.hp_disk_m = np.asarray(parameters.hp_disk_m, dtype=np.float64)
        self.lp_disk_x = self._node_x_indices(parameters.lp_disk_loc)
        self.hp_disk_x = self._node_x_indices(parameters.hp_disk_loc)
        self.lp_disk_y = self.lp_disk_x + self._n_dof // 2
        self.hp_disk_y = self.hp_disk_x + self._n_dof // 2
        self.bearing_ix = self._node_x_indices((parameters.bearing_node_i,))[0]
        self.bearing_ox = self._node_x_indices((parameters.bearing_node_o,))[0]
        self.bearing_iy = self.bearing_ix + self._n_dof // 2
        self.bearing_oy = self.bearing_ox + self._n_dof // 2
        self.roller_phase = 2.0 * np.pi / parameters.bearing_nb * np.arange(
            parameters.bearing_nb,
            dtype=np.float64,
        )

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def bearing_nonlinear_dofs(self) -> tuple[int, int, int, int]:
        return (self.bearing_ix, self.bearing_ox, self.bearing_iy, self.bearing_oy)

    def _node_x_indices(self, node_locations: tuple[int, ...]) -> NDArray[np.int64]:
        return np.asarray([2 * location - 2 for location in node_locations], dtype=np.int64)

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.gyroscopic, "dx", 2.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        return (ForcingTerm(self._base_unbalance_force(t), 2.0),)

    def _base_unbalance_force(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        lp_me = self.lp_disk_e * self.lp_disk_m
        hp_me = self.hp_disk_e * self.hp_disk_m

        force[:, self.lp_disk_x] = np.cos(t)[:, None] * lp_me[None, :]
        force[:, self.lp_disk_y] = np.sin(t)[:, None] * lp_me[None, :]

        hp_scale = self.speed_ratio ** 2
        hp_angle = self.speed_ratio * t
        force[:, self.hp_disk_x] = hp_scale * np.cos(hp_angle)[:, None] * hp_me[None, :]
        force[:, self.hp_disk_y] = hp_scale * np.sin(hp_angle)[:, None] * hp_me[None, :]
        return force

    def nonlinear_force(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        fx, fy, _ = self._bearing_force_and_partials(t, x)
        force[:, self.bearing_ix] = fx
        force[:, self.bearing_iy] = fy
        force[:, self.bearing_ox] = -fx
        force[:, self.bearing_oy] = -fy
        return force

    def local_nonlinear_force_and_partials(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        fx, fy, partials = self._bearing_force_and_partials_from_local(t, local_x)
        dfx_dx, dfx_dy, dfy_dx, dfy_dy = partials
        force = np.column_stack((fx, -fx, fy, -fy))
        jacobian = np.empty((t.size, 4, 4), dtype=np.float64)
        jacobian[:, 0, :] = np.column_stack((dfx_dx, -dfx_dx, dfx_dy, -dfx_dy))
        jacobian[:, 1, :] = -jacobian[:, 0, :]
        jacobian[:, 2, :] = np.column_stack((dfy_dx, -dfy_dx, dfy_dy, -dfy_dy))
        jacobian[:, 3, :] = -jacobian[:, 2, :]
        return force, jacobian

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return np.zeros((t.size, local_x.shape[1]), dtype=np.float64)

    def nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NonlinearJacobianTerm, ...]:
        _, _, partials = self._bearing_force_and_partials(t, x)
        dfx_dx, dfx_dy, dfy_dx, dfy_dy = partials
        ix, iy, ox, oy = self.bearing_ix, self.bearing_iy, self.bearing_ox, self.bearing_oy
        return (
            NonlinearJacobianTerm(ix, "x", ix, dfx_dx),
            NonlinearJacobianTerm(ix, "x", ox, -dfx_dx),
            NonlinearJacobianTerm(ix, "x", iy, dfx_dy),
            NonlinearJacobianTerm(ix, "x", oy, -dfx_dy),
            NonlinearJacobianTerm(iy, "x", ix, dfy_dx),
            NonlinearJacobianTerm(iy, "x", ox, -dfy_dx),
            NonlinearJacobianTerm(iy, "x", iy, dfy_dy),
            NonlinearJacobianTerm(iy, "x", oy, -dfy_dy),
            NonlinearJacobianTerm(ox, "x", ix, -dfx_dx),
            NonlinearJacobianTerm(ox, "x", ox, dfx_dx),
            NonlinearJacobianTerm(ox, "x", iy, -dfx_dy),
            NonlinearJacobianTerm(ox, "x", oy, dfx_dy),
            NonlinearJacobianTerm(oy, "x", ix, -dfy_dx),
            NonlinearJacobianTerm(oy, "x", ox, dfy_dx),
            NonlinearJacobianTerm(oy, "x", iy, -dfy_dy),
            NonlinearJacobianTerm(oy, "x", oy, dfy_dy),
        )

    def _bearing_force_and_partials(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        local_x = x[:, list(self.bearing_nonlinear_dofs)]
        return self._bearing_force_and_partials_from_local(t, local_x)

    def _bearing_force_and_partials_from_local(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        theta = self.roller_phase[:, None] + self.omega_c * t[None, :]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        dx_io = local_x[:, 0] - local_x[:, 1]
        dy_io = local_x[:, 2] - local_x[:, 3]
        delta = dx_io[None, :] * cos_theta + dy_io[None, :] * sin_theta - self.parameters.bearing_clearance
        active_delta = np.maximum(delta, 0.0)

        kb = self.parameters.bearing_kb
        fx = (kb * active_delta**1.5 * cos_theta).sum(axis=0)
        fy = (kb * active_delta**1.5 * sin_theta).sum(axis=0)

        common = 1.5 * kb * np.sqrt(active_delta)
        dfx_dx = (common * cos_theta * cos_theta).sum(axis=0)
        dfx_dy = (common * cos_theta * sin_theta).sum(axis=0)
        dfy_dx = (common * sin_theta * cos_theta).sum(axis=0)
        dfy_dy = (common * sin_theta * sin_theta).sum(axis=0)
        return fx, fy, (dfx_dx, dfx_dy, dfy_dx, dfy_dy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the condensed aero-engine arc-length continuation example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-fft", type=int, default=DEFAULT_SAMPLE_FFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npy-output", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-coeff-history", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CondensedContinuationConfig:
    return CondensedContinuationConfig(
        sample_fft=args.sample_fft,
        harmonics=HARMONICS,
        nonlinear_harmonics=build_full_fft_nonlinear_harmonics(args.sample_fft, FREQUENCY_RESOLUTION),
        frequency_resolution=FREQUENCY_RESOLUTION,
        init_omega=INIT_OMEGA,
        max_epoch=MAX_EPOCH,
        res_tolerance=RES_TOLERANCE,
        delta_tolerance=DELTA_TOLERANCE,
        seed=args.seed,
        initial_scale=INITIAL_SCALE,
        s_initial=0.1,
        s_max=0.1,
        s_min=1e-9,
        q_scale=Q_SCALE,
        omega_scale=OMEGA_SCALE,
        loop_switch_enabled=LOOP_SWITCH_ENABLED,
        loop_revisit_tolerance=0.01,
        loop_restart_omega_delta=0.5,
        max_steps=args.max_steps,
        progress_callback=print,
    )


def default_plot_dofs(model: AeroEngineRotorModel) -> tuple[int, int, int, int]:
    return (model.bearing_ix, model.bearing_iy, model.bearing_ox, model.bearing_oy)


def save_result(
    result: CondensedContinuationResult,
    args: argparse.Namespace,
    plot_dofs: tuple[int, int, int, int],
) -> None:
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
        "nonlinear_coefficients": result.nonlinear_coefficients,
        "omega": np.array(result.omega),
        "harmonics": result.harmonics,
        "nonlinear_harmonics": result.nonlinear_harmonics,
        "frequency_resolution": np.array(result.frequency_resolution),
        "period": np.array(result.period),
        "q_scale": np.array(Q_SCALE),
        "omega_scale": np.array(OMEGA_SCALE),
        "loop_switch_enabled": np.array(LOOP_SWITCH_ENABLED),
        "condensed_dimension": np.array(result.condensed_dimension),
        "full_dimension": np.array(result.full_dimension),
        "plot_dofs": np.asarray(plot_dofs, dtype=np.int64),
        "log_step": np.asarray([log.step for log in result.logs], dtype=np.int64),
        "log_epoch": np.asarray([log.epoch for log in result.logs], dtype=np.int64),
        "log_max_residual": np.asarray([log.max_residual for log in result.logs], dtype=np.float64),
        "log_max_delta": np.asarray([log.max_delta for log in result.logs], dtype=np.float64),
        "log_omega": np.asarray([log.omega for log in result.logs], dtype=np.float64),
        "log_arc_length": np.asarray([log.arc_length for log in result.logs], dtype=np.float64),
        "log_converged": np.asarray([log.converged for log in result.logs], dtype=np.bool_),
        "loop_event_step": np.asarray([event.step for event in result.loop_events], dtype=np.int64),
        "loop_event_current_index": np.asarray([event.current_index for event in result.loop_events], dtype=np.int64),
        "loop_event_matched_index": np.asarray([event.matched_index for event in result.loop_events], dtype=np.int64),
        "loop_event_distance": np.asarray([event.distance for event in result.loop_events], dtype=np.float64),
        "loop_event_anchor_index": np.asarray([event.anchor_index for event in result.loop_events], dtype=np.int64),
        "loop_event_restart_omega": np.asarray([event.restart_omega for event in result.loop_events], dtype=np.float64),
        "loop_event_restarted": np.asarray([event.restarted for event in result.loop_events], dtype=np.bool_),
    }
    if result.initial_log is not None:
        payload["initial_epoch"] = np.array(result.initial_log.epoch)
        payload["initial_max_residual"] = np.array(result.initial_log.max_residual)
        payload["initial_max_delta"] = np.array(result.initial_log.max_delta)
        payload["initial_omega"] = np.array(result.initial_log.omega)
    if args.save_coeff_history:
        payload["coefficient_history"] = result.coefficient_history
        payload["nonlinear_coefficient_history"] = result.nonlinear_coefficient_history
        payload["parameter_history"] = result.parameter_history
    np.savez(args.output, **payload)
    print(f"Saved result to {args.output}")

    table_output = args.npy_output or args.output.with_suffix(".npy")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(table_output, curve_table)
    print(f"Saved curve table to {table_output}")

    if args.plot:
        save_plot(result, args.output.with_suffix(".png"), plot_dofs)


def save_plot(result: CondensedContinuationResult, plot_path: Path, plot_dofs: tuple[int, int, int, int]) -> None:
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
    solver = CondensedContinuationSolver(model, build_config(args), model.bearing_nonlinear_dofs)
    result = solver.run()
    save_result(result, args, default_plot_dofs(model))


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
