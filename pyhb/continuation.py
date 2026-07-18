"""Newton and arc-length continuation for second-order HB models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .continuation_core import (
    _ArcLengthMetric,
    _array_is_finite,
    _coefficient_matrix,
    _combine_powered_dense_blocks,
    _combine_powered_sparse_blocks,
    _evaluate_local_state,
    _local_samples_to_global_coefficients,
    _parameter_step_too_large,
    _PoweredEvaluation,
    _prepare_continuation_problem,
    _residual_stats,
    _ResidualStats,
    _shrink_arc_length_for_parameter_step,
    _solve_one_parameter_bordered_arc,
    _solve_sparse,
    _sparse_is_finite,
    _validate_optional_positive_scale,
    _validate_positive_scale,
    _validated_dofs,
    assemble_hb_jacobian_from_local_matrices,
)
from .harmonics import flatten_coefficients
from .models import SecondOrderTimeModel


@dataclass(frozen=True)
class ContinuationConfig:
    sample_fft: int = 2**11
    harmonics: tuple[float, ...] = tuple(float(v) for v in range(1, 11))
    frequency_resolution: float = 1.0
    frequency_tolerance: float = 1e-10
    s3_method: str = "fast"
    s3_quadrature_samples: int | None = None
    init_omega: float = 300.0
    initial_direction: str = "up"
    max_epoch: int = 10
    res_tolerance: float = 1e-10
    delta_tolerance: float = 1e-10
    s_initial: float = 0.05
    s_max: float = 0.1
    s_min: float = 1e-6
    q_scale: float = 1.0
    omega_scale: float = 1.0
    max_parameter_step: float | None = None
    parameter_step_safety: float = 0.8
    max_steps: int = 500
    shrink_limit: int = 20
    residual_floor: float = 1e-16
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StepLog:
    step: int
    epoch: int
    max_residual: float
    max_delta: float
    omega: float
    arc_length: float
    converged: bool
    relative_residual: float = np.inf


@dataclass(frozen=True)
class ContinuationResult:
    coefficients: NDArray[np.float64]
    omega: float
    coefficient_history: NDArray[np.float64]
    parameter_history: NDArray[np.float64]
    harmonics: NDArray[np.float64]
    nonlinear_harmonics: NDArray[np.float64]
    frequency_resolution: float
    period: float
    logs: list[StepLog] = field(default_factory=list)
    initial_log: StepLog | None = None


@dataclass(frozen=True)
class _NonlinearEvaluation:
    coefficients: NDArray[np.float64]
    jacobian: sparse.csc_matrix
    parameter_coefficients: NDArray[np.float64] | None = None


def _nonlinear_evaluation_is_finite(nonlinear: _NonlinearEvaluation) -> bool:
    return (
        _array_is_finite(nonlinear.coefficients)
        and _sparse_is_finite(nonlinear.jacobian)
        and (nonlinear.parameter_coefficients is None or _array_is_finite(nonlinear.parameter_coefficients))
    )


class ContinuationSolver:
    """Generic single-parameter arc-length continuation solver."""

    def __init__(self, model: SecondOrderTimeModel, config: ContinuationConfig | None = None) -> None:
        self.model = model
        self.config = config or ContinuationConfig()
        _validate_continuation_config(self.config)
        self.prepared = _prepare_continuation_problem(
            self.model,
            sample_fft=self.config.sample_fft,
            harmonics=self.config.harmonics,
            frequency_resolution=self.config.frequency_resolution,
            frequency_tolerance=self.config.frequency_tolerance,
            s3_method=self.config.s3_method,
            s3_quadrature_samples=self.config.s3_quadrature_samples,
            progress_callback=self.config.progress_callback,
        )
        self._arc_metric = _ArcLengthMetric.build(
            self.model.n_dof * self.prepared.context.order,
            self.config.q_scale,
            (self.config.omega_scale,),
        )

    def _emit_progress(self, message: str) -> None:
        if self.config.progress_callback is not None:
            self.config.progress_callback(message)

    def _coefficient_matrix(self, coeff_line: NDArray[np.float64]) -> NDArray[np.float64]:
        return _coefficient_matrix(coeff_line, self.prepared.context.order, self.model.n_dof)

    def _evaluate_local_state(
        self,
        coefficients: NDArray[np.float64],
        coordinate_dofs: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return _evaluate_local_state(
            coefficients,
            coordinate_dofs,
            self.prepared.hb_item,
            self.prepared.hb_item_dt,
            self.prepared.hb_item_ddt,
        )

    def _linear_jacobian(self, parameter: float) -> sparse.csc_matrix:
        return _combine_powered_sparse_blocks(self.prepared.operator_blocks, parameter)

    def _linear_jacobian_derivative(self, parameter: float) -> sparse.csc_matrix:
        return _combine_powered_sparse_blocks(self.prepared.operator_blocks, parameter, derivative=True)

    def _forcing_coefficients(self, parameter: float) -> NDArray[np.float64]:
        return _combine_powered_dense_blocks(self.prepared.forcing_blocks, parameter)

    def _forcing_derivative_coefficients(self, parameter: float) -> NDArray[np.float64]:
        return _combine_powered_dense_blocks(self.prepared.forcing_blocks, parameter, derivative=True)

    def _residual_terms(
        self,
        coeff_line: NDArray[np.float64],
        nonlinear: _NonlinearEvaluation,
        parameter: float,
        powered: _PoweredEvaluation | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        active_powered = powered or self.prepared.evaluate_powered(parameter, derivative=False)
        forcing_coefficients = active_powered.forcing
        linear_coefficients = active_powered.operator @ coeff_line
        residual = forcing_coefficients - nonlinear.coefficients - linear_coefficients
        return residual, forcing_coefficients, nonlinear.coefficients, linear_coefficients

    def _jacobian(
        self,
        nonlinear: _NonlinearEvaluation,
        parameter: float,
        powered: _PoweredEvaluation | None = None,
    ) -> sparse.csc_matrix:
        active_powered = powered or self.prepared.evaluate_powered(parameter, derivative=False)
        return active_powered.operator + nonlinear.jacobian

    def _evaluate_nonlinear(
        self,
        coefficients: NDArray[np.float64],
        parameter: float,
        *,
        include_parameter: bool,
    ) -> _NonlinearEvaluation:
        force_dofs = _validated_dofs("nonlinear_force_dofs", self.model.nonlinear_force_dofs, self.model.n_dof)
        coordinate_dofs = _validated_dofs(
            "nonlinear_coordinate_dofs",
            self.model.nonlinear_coordinate_dofs,
            self.model.n_dof,
        )
        local_x, local_dx, local_ddx = self._evaluate_local_state(coefficients, coordinate_dofs)

        local_force = self.model.local_nonlinear_force(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            parameter,
        )
        nonlinear_coefficients = _local_samples_to_global_coefficients(
            local_force,
            force_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local nonlinear force",
        )

        local_jacobian = self.model.local_nonlinear_jacobian(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            parameter,
        )
        nonlinear_jacobian = assemble_hb_jacobian_from_local_matrices(
            local_jacobian,
            force_dofs,
            coordinate_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local nonlinear",
        )
        parameter_coefficients = None
        if include_parameter:
            local_parameter = self.model.local_nonlinear_parameter_derivative(
                self.prepared.t,
                local_x,
                local_dx,
                local_ddx,
                parameter,
            )
            parameter_coefficients = _local_samples_to_global_coefficients(
                local_parameter,
                force_dofs,
                self.prepared.context,
                self.prepared.t.size,
                self.model.n_dof,
                "local nonlinear parameter derivative",
            )
        return _NonlinearEvaluation(nonlinear_coefficients, nonlinear_jacobian, parameter_coefficients)

    def _parameter_jacobian(
        self,
        coeff_line: NDArray[np.float64],
        nonlinear: _NonlinearEvaluation,
        parameter: float,
        powered: _PoweredEvaluation | None = None,
    ) -> NDArray[np.float64]:
        if nonlinear.parameter_coefficients is None:
            raise ValueError("nonlinear evaluation does not include parameter coefficients")
        active_powered = powered or self.prepared.evaluate_powered(parameter, derivative=True)
        if active_powered.forcing_derivative is None or active_powered.operator_derivative is None:
            raise ValueError("powered evaluation does not include parameter derivatives")
        parameter_column = (
            active_powered.forcing_derivative
            - nonlinear.parameter_coefficients
            - active_powered.operator_derivative @ coeff_line
        )
        return parameter_column.reshape(-1, 1)

    def _solve_initial(
        self,
        coeff_line: NDArray[np.float64],
        parameter: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        _NonlinearEvaluation,
        sparse.csc_matrix,
        _PoweredEvaluation,
        StepLog,
    ]:
        config = self.config
        epoch = 1
        residual_vector = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        residual_stats = _ResidualStats(np.inf, np.inf)
        delta = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        coeff = None
        nonlinear = None
        jacobian = None

        while (
            epoch < config.max_epoch
            and residual_stats.relative_residual >= config.res_tolerance
            and np.max(np.abs(delta)) >= config.delta_tolerance
        ):
            coeff = self._coefficient_matrix(coeff_line)
            nonlinear = self._evaluate_nonlinear(coeff, parameter, include_parameter=False)
            powered = self.prepared.evaluate_powered(parameter, derivative=False)
            jacobian = self._jacobian(nonlinear, parameter, powered)
            residual_terms = self._residual_terms(coeff_line, nonlinear, parameter, powered)
            residual_vector = residual_terms[0]
            residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
            delta = _solve_sparse(jacobian, residual_vector)
            coeff_line = coeff_line + delta
            epoch += 1

        coeff = self._coefficient_matrix(coeff_line)
        nonlinear = self._evaluate_nonlinear(coeff, parameter, include_parameter=True)
        powered = self.prepared.evaluate_powered(parameter, derivative=True)
        jacobian = self._jacobian(nonlinear, parameter, powered)
        log = StepLog(
            step=0,
            epoch=epoch,
            relative_residual=residual_stats.relative_residual,
            max_residual=residual_stats.max_residual,
            max_delta=float(np.max(np.abs(delta))),
            omega=float(parameter),
            arc_length=0.0,
            converged=bool(
                residual_stats.relative_residual < config.res_tolerance
                or np.max(np.abs(delta)) < config.delta_tolerance
            ),
        )
        return coeff_line, coeff, nonlinear, jacobian, powered, log

    def _orient_initial_tangent(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        oriented = np.asarray(tangent, dtype=np.float64)
        direction = 1.0 if self.config.initial_direction == "up" else -1.0
        if oriented[-1] * direction < 0.0:
            oriented = -oriented
        return self._arc_metric.normalize(oriented)

    def _initial_tangent(
        self, jacobian: sparse.csc_matrix, parameter_column: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        tangent_q = _solve_sparse(jacobian, parameter_column.reshape(-1))
        return self._orient_initial_tangent(np.concatenate((tangent_q, np.array([1.0], dtype=np.float64))))

    def _run_full(
        self,
        initial_coefficients: NDArray[np.float64] | None = None,
        initial_parameter: float | None = None,
    ) -> ContinuationResult:
        config = self.config
        order = self.prepared.context.order
        if initial_coefficients is None:
            raise ValueError("initial_coefficients must be provided")
        initial = np.asarray(initial_coefficients, dtype=np.float64)
        expected_shape = (order, self.model.n_dof)
        expected_size = order * self.model.n_dof
        if initial.ndim == 2:
            if initial.shape != expected_shape:
                raise ValueError(f"initial_coefficients must have shape {expected_shape}, got {initial.shape}")
            coeff_line = flatten_coefficients(initial)
        elif initial.ndim == 1:
            if initial.size != expected_size:
                raise ValueError(f"initial_coefficients must have length {expected_size}, got {initial.size}")
            coeff_line = initial.reshape(-1)
        else:
            raise ValueError(
                "initial_coefficients must be a 1D coefficient vector or "
                f"a 2D coefficient matrix shaped {expected_shape}, got {initial.shape}"
            )

        parameter = float(config.init_omega if initial_parameter is None else initial_parameter)
        coeff_line, coeff, nonlinear, jacobian, powered, initial_log = self._solve_initial(coeff_line, parameter)
        self._emit_progress(
            "Initial computation, "
            f"Epoch = {initial_log.epoch}, RelRes = {initial_log.relative_residual:.6e}, "
            f"MaxRes = {initial_log.max_residual:.6e}, "
            f"Delta = {initial_log.max_delta:.6e}, Omega = {initial_log.omega:.10g}"
        )
        y0 = np.concatenate((coeff_line, np.array([parameter], dtype=np.float64)))

        parameter_history: list[float] = [float(parameter)] if initial_log.converged else []
        coefficient_history: list[NDArray[np.float64]] = [coeff.copy()] if initial_log.converged else []
        logs: list[StepLog] = []
        if not initial_log.converged or config.max_steps == 0:
            return self._build_result(y0, coefficient_history, parameter_history, logs, initial_log)

        j_parameter = self._parameter_jacobian(coeff_line, nonlinear, parameter, powered)
        tangent = self._initial_tangent(jacobian, j_parameter)
        arc_length_step = float(config.s_initial)
        shrink_count = 0

        for step in range(1, config.max_steps + 1):
            while True:
                epoch = 1
                y = y0 + arc_length_step * tangent
                coeff_line = y[:-1].copy()
                parameter = float(y[-1])
                residual_vector = np.full(self.model.n_dof * order, np.inf, dtype=np.float64)
                residual_stats = _ResidualStats(np.inf, np.inf)
                delta = np.full(self.model.n_dof * order + 1, np.inf, dtype=np.float64)
                arc_solve = None
                nonfinite_trial = not _array_is_finite(y)

                while (
                    not nonfinite_trial
                    and epoch < config.max_epoch
                    and residual_stats.relative_residual >= config.res_tolerance
                    and np.max(np.abs(delta)) >= config.delta_tolerance
                ):
                    if not _array_is_finite(coeff_line) or not np.isfinite(parameter):
                        nonfinite_trial = True
                        break
                    coeff = self._coefficient_matrix(coeff_line)
                    nonlinear = self._evaluate_nonlinear(coeff, parameter, include_parameter=True)
                    if not _nonlinear_evaluation_is_finite(nonlinear):
                        nonfinite_trial = True
                        break
                    powered = self.prepared.evaluate_powered(parameter, derivative=True)
                    jacobian = self._jacobian(nonlinear, parameter, powered)
                    if not _sparse_is_finite(jacobian):
                        nonfinite_trial = True
                        break
                    residual_terms = self._residual_terms(coeff_line, nonlinear, parameter, powered)
                    residual_vector = residual_terms[0]
                    if not all(_array_is_finite(term) for term in residual_terms):
                        residual_stats = _ResidualStats(np.inf, np.inf)
                        nonfinite_trial = True
                        break
                    residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
                    j_parameter = self._parameter_jacobian(coeff_line, nonlinear, parameter, powered)
                    arc_residual = self._arc_metric.inner(y - y0, tangent) - arc_length_step
                    if not _array_is_finite(j_parameter) or not np.isfinite(arc_residual):
                        nonfinite_trial = True
                        break
                    arc_solve = _solve_one_parameter_bordered_arc(
                        jacobian,
                        j_parameter,
                        self._arc_metric.constraint_row(tangent),
                        residual_vector,
                        arc_residual,
                    )
                    delta = arc_solve.delta
                    if not _array_is_finite(delta) or not _array_is_finite(arc_solve.tangent_candidate):
                        nonfinite_trial = True
                        break
                    y = y - delta
                    coeff_line = y[:-1].copy()
                    parameter = float(y[-1])
                    epoch += 1

                max_delta = float(np.max(np.abs(delta))) if _array_is_finite(delta) else np.inf
                converged = bool(
                    not nonfinite_trial
                    and (
                        residual_stats.relative_residual <= config.res_tolerance or max_delta <= config.delta_tolerance
                    )
                )
                parameter_step = abs(float(parameter) - float(y0[-1])) if np.isfinite(parameter) else np.inf
                raw_parameter_step_too_large = _parameter_step_too_large(config.max_parameter_step, parameter_step)
                parameter_step_too_large = bool(converged and raw_parameter_step_too_large)
                accepted = bool(converged and not parameter_step_too_large)
                step_log = StepLog(
                    step=step,
                    epoch=epoch,
                    max_residual=residual_stats.max_residual,
                    max_delta=max_delta,
                    omega=float(parameter),
                    arc_length=float(arc_length_step),
                    converged=accepted,
                    relative_residual=residual_stats.relative_residual,
                )
                logs.append(step_log)
                if accepted:
                    status = "ok"
                elif not converged:
                    status = "no convergence"
                else:
                    status = f"parameter step too large, dOmega = {parameter_step:.6g}"
                self._emit_progress(
                    f"Times = {step_log.step}, Epoch = {step_log.epoch}, RelRes = {step_log.relative_residual:.6e}, "
                    f"MaxRes = {step_log.max_residual:.6e}, "
                    f"Delta = {step_log.max_delta:.6e}, Omega = {step_log.omega:.10g}, "
                    f"s = {step_log.arc_length:.6g}, {status}"
                )

                if accepted:
                    y0 = y.copy()
                    if arc_solve is None:
                        raise RuntimeError("converged before assembling arc Jacobian")
                    tangent = self._arc_metric.normalize(arc_solve.tangent_candidate)
                    coeff = self._coefficient_matrix(coeff_line)
                    parameter_history.append(float(parameter))
                    coefficient_history.append(coeff.copy())
                    shrink_count = 0
                    arc_length_step = min(2.0 * arc_length_step, config.s_max)
                    break

                shrink_count += 1
                if parameter_step_too_large:
                    if config.max_parameter_step is None:
                        raise RuntimeError("max_parameter_step is required for parameter-step shrinking")
                    arc_length_step = _shrink_arc_length_for_parameter_step(
                        arc_length_step,
                        parameter_step,
                        config.max_parameter_step,
                        config.parameter_step_safety,
                        config.s_min,
                    )
                    if shrink_count >= config.shrink_limit:
                        break
                    continue

                if nonfinite_trial or epoch >= config.max_epoch:
                    arc_length_step = max(0.5 * arc_length_step, config.s_min)
                else:
                    arc_length_step = min(2.0 * arc_length_step, config.s_max)
                break

            if shrink_count >= config.shrink_limit:
                break

        return self._build_result(y0, coefficient_history, parameter_history, logs, initial_log)

    def _build_result(
        self,
        y0: NDArray[np.float64],
        coefficient_history: list[NDArray[np.float64]],
        parameter_history: list[float],
        logs: list[StepLog],
        initial_log: StepLog,
    ) -> ContinuationResult:
        final_coeff = self._coefficient_matrix(y0[:-1])
        final_parameter = float(y0[-1])
        coefficient_history_array = (
            np.asarray(coefficient_history, dtype=np.float64)
            if coefficient_history
            else np.empty((0, self.prepared.context.order, self.model.n_dof), dtype=np.float64)
        )
        parameter_history_array = np.asarray(parameter_history, dtype=np.float64)
        return ContinuationResult(
            coefficients=final_coeff,
            omega=final_parameter,
            coefficient_history=coefficient_history_array,
            parameter_history=parameter_history_array,
            harmonics=np.asarray(self.prepared.context.harmonics, dtype=np.float64),
            nonlinear_harmonics=np.asarray(self.prepared.context.nonlinear_harmonics, dtype=np.float64),
            frequency_resolution=float(self.prepared.context.frequency_resolution),
            period=float(self.prepared.context.period),
            logs=logs,
            initial_log=initial_log,
        )

    def run(
        self,
        initial_coefficients: NDArray[np.float64] | None = None,
        initial_parameter: float | None = None,
    ) -> ContinuationResult:
        """Run initial Newton solve followed by arc-length continuation."""

        return self._run_full(initial_coefficients, initial_parameter)


def _validate_initial_direction(value: str) -> None:
    if value not in {"up", "down"}:
        raise ValueError("initial_direction must be 'up' or 'down'")


def _validate_continuation_config(config: ContinuationConfig) -> None:
    _validate_initial_direction(config.initial_direction)
    _validate_positive_scale("q_scale", config.q_scale)
    _validate_positive_scale("omega_scale", config.omega_scale)
    _validate_optional_positive_scale("max_parameter_step", config.max_parameter_step)
    _validate_positive_scale("parameter_step_safety", config.parameter_step_safety)
    _validate_positive_scale("residual_floor", config.residual_floor)
