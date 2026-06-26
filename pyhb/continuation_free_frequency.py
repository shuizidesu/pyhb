"""Free-frequency HB continuation for autonomous second-order systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from .continuation import ContinuationConfig, ContinuationSolver
from .continuation_core import (
    _parameter_step_too_large,
    _residual_stats,
    _shrink_arc_length_for_parameter_step,
    _solve_sparse,
    _validate_optional_positive_scale,
    _validate_positive_scale,
    assemble_hb_jacobian_from_terms,
)
from .harmonics import flatten_coefficients, stack_fft_coefficients
from .models import FreeFrequencySecondOrderTimeModel, HarmonicCoefficientConstraint


@dataclass(frozen=True)
class ContinuationFreeFrequencyConfig(ContinuationConfig):
    """Continuation config for autonomous free-frequency HB systems."""

    init_parameter: float = 0.5
    parameter_scale: float = 1.0
    constraint: HarmonicCoefficientConstraint = field(
        default_factory=lambda: HarmonicCoefficientConstraint(dof=0, coefficient_index=1, value=0.0)
    )
    constraint_tolerance: float = 1e-10
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ContinuationFreeFrequencyStepLog:
    step: int
    epoch: int
    max_residual: float
    relative_residual: float
    constraint_residual: float
    max_delta: float
    omega: float
    parameter: float
    arc_length: float
    converged: bool


@dataclass(frozen=True)
class ContinuationFreeFrequencyResult:
    coefficients: NDArray[np.float64]
    omega: float
    parameter: float
    coefficient_history: NDArray[np.float64]
    omega_history: NDArray[np.float64]
    parameter_history: NDArray[np.float64]
    harmonics: NDArray[np.float64]
    nonlinear_harmonics: NDArray[np.float64]
    frequency_resolution: float
    period: float
    logs: list[ContinuationFreeFrequencyStepLog] = field(default_factory=list)
    initial_log: ContinuationFreeFrequencyStepLog | None = None


class ContinuationFreeFrequencySolver(ContinuationSolver):
    """Arc-length continuation with unknown response frequency and one true parameter."""

    def __init__(
        self,
        model: FreeFrequencySecondOrderTimeModel,
        config: ContinuationFreeFrequencyConfig | None = None,
    ) -> None:
        if not isinstance(model, FreeFrequencySecondOrderTimeModel):
            raise TypeError("ContinuationFreeFrequencySolver requires a FreeFrequencySecondOrderTimeModel")
        super().__init__(model, config or ContinuationFreeFrequencyConfig())
        self.model: FreeFrequencySecondOrderTimeModel
        self.config: ContinuationFreeFrequencyConfig
        _validate_positive_scale("parameter_scale", self.config.parameter_scale)
        _validate_positive_scale("constraint_tolerance", self.config.constraint_tolerance)
        _validate_optional_positive_scale("max_parameter_step", self.config.max_parameter_step)
        self._constraint_index = _constraint_flat_index(
            self.config.constraint,
            self.prepared.context.order,
            self.model.n_dof,
        )

    def _residual_terms(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        generalized_coefficients = stack_fft_coefficients(
            self._generalized_force_samples(x, dx, ddx, omega, parameter),
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )
        forcing_coefficients = self._forcing_coefficients(omega)
        linear_coefficients = self._linear_jacobian(omega) @ coeff_line
        residual = forcing_coefficients - generalized_coefficients - linear_coefficients
        return residual, forcing_coefficients, generalized_coefficients, linear_coefficients

    def _jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> sparse.csc_matrix:
        return self._linear_jacobian(omega) + self._generalized_jacobian(x, dx, ddx, omega, parameter)

    def _generalized_jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> sparse.csc_matrix:
        terms = tuple(self._generalized_jacobian_terms(x, dx, ddx, omega, parameter))
        return assemble_hb_jacobian_from_terms(
            terms,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
        )

    def _omega_column(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        generalized_omega = stack_fft_coefficients(
            self._generalized_omega_derivative_samples(x, dx, ddx, omega, parameter),
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )
        return (
            self._forcing_derivative_coefficients(omega)
            - generalized_omega
            - self._linear_jacobian_derivative(omega) @ coeff_line
        )

    def _parameter_column(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        generalized_parameter = stack_fft_coefficients(
            self._generalized_parameter_derivative_samples(x, dx, ddx, omega, parameter),
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )
        return -generalized_parameter

    def _generalized_force_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        return self.model.residual_force(self.prepared.t, x, dx, ddx, omega, parameter)

    def _generalized_jacobian_terms(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ):
        return self.model.residual_jacobian_terms(self.prepared.t, x, dx, ddx, omega, parameter)

    def _generalized_omega_derivative_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        return self.model.residual_omega_derivative(self.prepared.t, x, dx, ddx, omega, parameter)

    def _generalized_parameter_derivative_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        return self.model.residual_parameter_derivative(self.prepared.t, x, dx, ddx, omega, parameter)

    def _constraint_residual(self, coeff_line: NDArray[np.float64]) -> float:
        return float(coeff_line[self._constraint_index] - self.config.constraint.value)

    def _constraint_row(self, width: int) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (
                np.array([1.0], dtype=np.float64),
                (
                    np.array([0], dtype=np.int64),
                    np.array([self._constraint_index], dtype=np.int64),
                ),
            ),
            shape=(1, width),
        )

    def _initial_matrix(
        self,
        jacobian: sparse.spmatrix,
        omega_column: NDArray[np.float64],
    ) -> sparse.csc_matrix:
        size = jacobian.shape[0]
        top = sparse.hstack(
            (
                jacobian,
                -sparse.csc_matrix(np.asarray(omega_column, dtype=np.float64).reshape(size, 1)),
            ),
            format="csc",
        )
        constraint = sparse.hstack((self._constraint_row(size), sparse.csr_matrix((1, 1))), format="csr")
        return sparse.vstack((top, constraint), format="csc")

    def _arc_matrix(
        self,
        jacobian: sparse.spmatrix,
        omega_column: NDArray[np.float64],
        parameter_column: NDArray[np.float64],
        arc_row: NDArray[np.float64],
    ) -> sparse.csc_matrix:
        size = jacobian.shape[0]
        top = sparse.hstack(
            (
                -jacobian,
                sparse.csc_matrix(np.asarray(omega_column, dtype=np.float64).reshape(size, 1)),
                sparse.csc_matrix(np.asarray(parameter_column, dtype=np.float64).reshape(size, 1)),
            ),
            format="csc",
        )
        constraint = sparse.hstack((self._constraint_row(size), sparse.csr_matrix((1, 2))), format="csr")
        arc_sparse = sparse.csr_matrix(np.asarray(arc_row, dtype=np.float64).reshape(1, size + 2))
        return sparse.vstack((top, constraint, arc_sparse), format="csc")

    def _solve_initial_free(
        self,
        coeff_line: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], sparse.csc_matrix, ContinuationFreeFrequencyStepLog]:
        config = self.config
        epoch = 1
        residual_vector = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        residual_stats = _residual_stats(residual_vector, (), config.residual_floor)
        constraint_residual = np.inf
        delta = np.full(coeff_line.size + 1, np.inf, dtype=np.float64)
        coeff = x = dx = ddx = None
        jacobian = None

        while (
            epoch < config.max_epoch
            and not _free_converged(
                residual_stats.relative_residual,
                constraint_residual,
                float(np.max(np.abs(delta))),
                config.res_tolerance,
                config.constraint_tolerance,
                config.delta_tolerance,
            )
        ):
            coeff, x, dx, ddx = self._evaluate_state(coeff_line)
            jacobian = self._jacobian(x, dx, ddx, omega, parameter)
            residual_terms = self._residual_terms(coeff_line, x, dx, ddx, omega, parameter)
            residual_vector = residual_terms[0]
            residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
            omega_column = self._omega_column(coeff_line, x, dx, ddx, omega, parameter)
            constraint_residual = self._constraint_residual(coeff_line)
            initial_matrix = self._initial_matrix(jacobian, omega_column)
            rhs = np.concatenate((residual_vector, np.array([-constraint_residual], dtype=np.float64)))
            delta = _solve_sparse(initial_matrix, rhs)
            coeff_line = coeff_line + delta[:-1]
            omega = float(omega + delta[-1])
            epoch += 1

        coeff, x, dx, ddx = self._evaluate_state(coeff_line)
        jacobian = self._jacobian(x, dx, ddx, omega, parameter)
        residual_terms = self._residual_terms(coeff_line, x, dx, ddx, omega, parameter)
        residual_vector = residual_terms[0]
        residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
        constraint_residual = self._constraint_residual(coeff_line)
        max_delta = float(np.max(np.abs(delta)))
        log = ContinuationFreeFrequencyStepLog(
            step=0,
            epoch=epoch,
            relative_residual=residual_stats.relative_residual,
            max_residual=residual_stats.max_residual,
            constraint_residual=abs(constraint_residual),
            max_delta=max_delta,
            omega=float(omega),
            parameter=float(parameter),
            arc_length=0.0,
            converged=_free_converged(
                residual_stats.relative_residual,
                constraint_residual,
                max_delta,
                config.res_tolerance,
                config.constraint_tolerance,
                config.delta_tolerance,
            ),
        )
        return coeff_line, float(omega), coeff, x, jacobian, log

    def _arc_weights(self, size: int) -> NDArray[np.float64]:
        return np.concatenate(
            (
                np.full(size - 2, 1.0 / float(self.config.q_scale) ** 2, dtype=np.float64),
                np.array(
                    [
                        1.0 / float(self.config.omega_scale) ** 2,
                        1.0 / float(self.config.parameter_scale) ** 2,
                    ],
                    dtype=np.float64,
                ),
            )
        )

    def _orient_initial_tangent(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        oriented = np.asarray(tangent, dtype=np.float64)
        if oriented[-1] < 0.0:
            oriented = -oriented
        return self._weighted_normalize(oriented)

    def _initial_tangent_free(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
        jacobian: sparse.csc_matrix,
    ) -> NDArray[np.float64]:
        omega_column = self._omega_column(coeff_line, x, dx, ddx, omega, parameter)
        parameter_column = self._parameter_column(x, dx, ddx, omega, parameter)
        tangent_matrix = self._initial_matrix(jacobian, omega_column)
        rhs = np.concatenate((parameter_column, np.array([0.0], dtype=np.float64)))
        tangent_q_omega = _solve_sparse(tangent_matrix, rhs)
        return self._orient_initial_tangent(
            np.concatenate((tangent_q_omega, np.array([1.0], dtype=np.float64)))
        )

    def run(
        self,
        initial_coefficients: NDArray[np.float64],
        initial_omega: float | None = None,
        initial_parameter: float | None = None,
    ) -> ContinuationFreeFrequencyResult:
        """Run initial free-frequency Newton solve followed by arc-length continuation."""

        config = self.config
        order = self.prepared.context.order
        coeff_line = _as_coefficient_vector(initial_coefficients, order, self.model.n_dof)
        omega = float(config.init_omega if initial_omega is None else initial_omega)
        parameter = float(config.init_parameter if initial_parameter is None else initial_parameter)

        coeff_line, omega, coeff, x, jacobian, initial_log = self._solve_initial_free(
            coeff_line,
            omega,
            parameter,
        )
        self._emit_progress(
            "Initial computation, "
            f"Epoch = {initial_log.epoch}, RelRes = {initial_log.relative_residual:.6e}, "
            f"MaxRes = {initial_log.max_residual:.6e}, Constraint = {initial_log.constraint_residual:.6e}, "
            f"Delta = {initial_log.max_delta:.6e}, Omega = {initial_log.omega:.10g}, "
            f"Parameter = {initial_log.parameter:.10g}"
        )

        _, _, dx, ddx = self._evaluate_state(coeff_line)
        tangent = self._initial_tangent_free(coeff_line, x, dx, ddx, omega, parameter, jacobian)
        y0 = np.concatenate((coeff_line, np.array([omega, parameter], dtype=np.float64)))

        omega_history: list[float] = [float(omega)] if initial_log.converged else []
        parameter_history: list[float] = [float(parameter)] if initial_log.converged else []
        coefficient_history: list[NDArray[np.float64]] = [coeff.copy()] if initial_log.converged else []
        logs: list[ContinuationFreeFrequencyStepLog] = []
        arc_length_step = float(config.s_initial)
        shrink_count = 0

        for step in range(1, config.max_steps + 1):
            while True:
                epoch = 1
                y = y0 + arc_length_step * tangent
                coeff_line = y[:-2].copy()
                omega = float(y[-2])
                parameter = float(y[-1])
                residual_vector = np.full(self.model.n_dof * order, np.inf, dtype=np.float64)
                residual_stats = _residual_stats(residual_vector, (), config.residual_floor)
                constraint_residual = np.inf
                delta = np.full(self.model.n_dof * order + 2, np.inf, dtype=np.float64)
                j_arc_lu = None

                while (
                    epoch < config.max_epoch
                    and not _free_converged(
                        residual_stats.relative_residual,
                        constraint_residual,
                        float(np.max(np.abs(delta))),
                        config.res_tolerance,
                        config.constraint_tolerance,
                        config.delta_tolerance,
                    )
                ):
                    coeff, x, dx, ddx = self._evaluate_state(coeff_line)
                    jacobian = self._jacobian(x, dx, ddx, omega, parameter)
                    residual_terms = self._residual_terms(coeff_line, x, dx, ddx, omega, parameter)
                    residual_vector = residual_terms[0]
                    residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
                    omega_column = self._omega_column(coeff_line, x, dx, ddx, omega, parameter)
                    parameter_column = self._parameter_column(x, dx, ddx, omega, parameter)
                    constraint_residual = self._constraint_residual(coeff_line)
                    r_arc = np.concatenate(
                        (
                            residual_vector,
                            np.array(
                                [
                                    constraint_residual,
                                    self._weighted_inner(y - y0, tangent) - arc_length_step,
                                ],
                                dtype=np.float64,
                            ),
                        )
                    )
                    j_arc_v = self._arc_matrix(
                        jacobian,
                        omega_column,
                        parameter_column,
                        self._weighted_constraint_row(tangent),
                    )
                    j_arc_lu = splu(j_arc_v)
                    delta = j_arc_lu.solve(r_arc)
                    y = y - delta
                    coeff_line = y[:-2].copy()
                    omega = float(y[-2])
                    parameter = float(y[-1])
                    epoch += 1

                max_delta = float(np.max(np.abs(delta)))
                converged = _free_converged(
                    residual_stats.relative_residual,
                    constraint_residual,
                    max_delta,
                    config.res_tolerance,
                    config.constraint_tolerance,
                    config.delta_tolerance,
                )
                parameter_step = abs(float(parameter) - float(y0[-1]))
                raw_parameter_step_too_large = _parameter_step_too_large(config.max_parameter_step, parameter_step)
                parameter_step_too_large = bool(converged and raw_parameter_step_too_large)
                accepted = bool(converged and not parameter_step_too_large)
                step_log = ContinuationFreeFrequencyStepLog(
                    step=step,
                    epoch=epoch,
                    max_residual=residual_stats.max_residual,
                    relative_residual=residual_stats.relative_residual,
                    constraint_residual=abs(float(constraint_residual)),
                    max_delta=max_delta,
                    omega=float(omega),
                    parameter=float(parameter),
                    arc_length=float(arc_length_step),
                    converged=accepted,
                )
                logs.append(step_log)
                if accepted:
                    status = "ok"
                elif not converged:
                    status = "no convergence"
                else:
                    status = f"parameter step too large, dParameter = {parameter_step:.6g}"
                self._emit_progress(
                    f"Times = {step_log.step}, Epoch = {step_log.epoch}, "
                    f"RelRes = {step_log.relative_residual:.6e}, MaxRes = {step_log.max_residual:.6e}, "
                    f"Constraint = {step_log.constraint_residual:.6e}, Delta = {step_log.max_delta:.6e}, "
                    f"Omega = {step_log.omega:.10g}, Parameter = {step_log.parameter:.10g}, "
                    f"s = {step_log.arc_length:.6g}, {status}"
                )

                if accepted:
                    y0 = y.copy()
                    if j_arc_lu is None:
                        raise RuntimeError("converged before assembling free-frequency arc Jacobian")
                    tangent_candidate = j_arc_lu.solve(
                        np.concatenate((np.zeros(jacobian.shape[0] + 1), np.array([1.0], dtype=np.float64)))
                    )
                    tangent = self._weighted_normalize(tangent_candidate)
                    coeff, _, _, _ = self._evaluate_state(coeff_line)
                    omega_history.append(float(omega))
                    parameter_history.append(float(parameter))
                    coefficient_history.append(coeff.copy())
                    shrink_count = 0
                    arc_length_step = min(2.0 * arc_length_step, config.s_max)
                    break

                shrink_count += 1
                if parameter_step_too_large:
                    arc_length_step = _shrink_arc_length_for_parameter_step(
                        arc_length_step,
                        parameter_step,
                        float(config.max_parameter_step),
                        config.parameter_step_safety,
                        config.s_min,
                    )
                    if shrink_count >= config.shrink_limit:
                        break
                    continue

                if epoch >= config.max_epoch:
                    arc_length_step = max(0.5 * arc_length_step, config.s_min)
                else:
                    arc_length_step = min(2.0 * arc_length_step, config.s_max)
                break

            if shrink_count >= config.shrink_limit:
                break

        final_coeff, _, _, _ = self._evaluate_state(y0[:-2])
        coefficient_history_array = (
            np.asarray(coefficient_history, dtype=np.float64)
            if coefficient_history
            else np.empty((0, self.prepared.context.order, self.model.n_dof), dtype=np.float64)
        )
        return ContinuationFreeFrequencyResult(
            coefficients=final_coeff,
            omega=float(y0[-2]),
            parameter=float(y0[-1]),
            coefficient_history=coefficient_history_array,
            omega_history=np.asarray(omega_history, dtype=np.float64),
            parameter_history=np.asarray(parameter_history, dtype=np.float64),
            harmonics=np.asarray(self.prepared.context.harmonics, dtype=np.float64),
            nonlinear_harmonics=np.asarray(self.prepared.context.nonlinear_harmonics, dtype=np.float64),
            frequency_resolution=float(self.prepared.context.frequency_resolution),
            period=float(self.prepared.context.period),
            logs=logs,
            initial_log=initial_log,
        )


def _constraint_flat_index(constraint: HarmonicCoefficientConstraint, order: int, n_dof: int) -> int:
    dof = int(constraint.dof)
    coefficient_index = int(constraint.coefficient_index)
    if not (0 <= dof < n_dof):
        raise ValueError(f"constraint dof out of range: {dof}")
    if not (0 <= coefficient_index < order):
        raise ValueError(f"constraint coefficient_index out of range: {coefficient_index}")
    return dof * order + coefficient_index


def _as_coefficient_vector(values: NDArray[np.float64], order: int, n_dof: int) -> NDArray[np.float64]:
    initial = np.asarray(values, dtype=np.float64)
    expected_shape = (order, n_dof)
    expected_size = order * n_dof
    if initial.ndim == 2:
        if initial.shape != expected_shape:
            raise ValueError(f"initial_coefficients must have shape {expected_shape}, got {initial.shape}")
        return flatten_coefficients(initial)
    if initial.ndim == 1:
        if initial.size != expected_size:
            raise ValueError(f"initial_coefficients must have length {expected_size}, got {initial.size}")
        return initial.reshape(-1)
    raise ValueError(
        "initial_coefficients must be a 1D coefficient vector or "
        f"a 2D coefficient matrix shaped {expected_shape}, got {initial.shape}"
    )


def _free_converged(
    relative_residual: float,
    constraint_residual: float,
    max_delta: float,
    res_tolerance: float,
    constraint_tolerance: float,
    delta_tolerance: float,
) -> bool:
    return bool(
        (relative_residual <= res_tolerance and abs(float(constraint_residual)) <= constraint_tolerance)
        or max_delta <= delta_tolerance
    )
