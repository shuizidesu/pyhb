"""Free-frequency HB continuation for autonomous second-order systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .continuation import ContinuationConfig, _validate_continuation_config
from .continuation_core import (
    _ArcLengthMetric,
    _array_is_finite,
    _coefficient_matrix,
    _combine_linear_and_nonlinear_jacobians,
    _evaluate_local_state,
    _factorize_linear_system,
    _HBJacobian,
    _local_samples_to_global_coefficients,
    _matrix_is_finite,
    _parameter_step_too_large,
    _PoweredEvaluation,
    _prepare_continuation_problem,
    _residual_stats,
    _ResidualStats,
    _shrink_arc_length_for_parameter_step,
    _solve_linear_system,
    _validate_linear_solver,
    _validate_optional_positive_scale,
    _validate_positive_scale,
    _validated_dofs,
    assemble_hb_jacobian_from_local_matrices,
)
from .harmonics import flatten_coefficients
from .hb_operators import coefficient_derivative_maps
from .models import FreeFrequencySecondOrderTimeModel, HarmonicCoefficientConstraint, ReferencePhaseCondition


@dataclass(frozen=True)
class ContinuationFreeFrequencyConfig(ContinuationConfig):
    """Continuation config for autonomous free-frequency HB systems."""

    init_parameter: float = 0.5
    parameter_scale: float = 1.0
    initial_constraint: HarmonicCoefficientConstraint = field(
        default_factory=lambda: HarmonicCoefficientConstraint(dof=0, coefficient_index=1, value=0.0)
    )
    phase_condition: HarmonicCoefficientConstraint | ReferencePhaseCondition = field(
        default_factory=ReferencePhaseCondition
    )
    constraint: HarmonicCoefficientConstraint | None = field(default=None, repr=False, compare=False)
    constraint_tolerance: float = 1e-10
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.constraint is not None:
            object.__setattr__(self, "initial_constraint", self.constraint)
            if isinstance(self.phase_condition, ReferencePhaseCondition):
                object.__setattr__(self, "phase_condition", self.constraint)


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


@dataclass(frozen=True)
class _GeneralizedEvaluation:
    coefficients: NDArray[np.float64]
    jacobian: _HBJacobian
    omega_coefficients: NDArray[np.float64]
    parameter_coefficients: NDArray[np.float64] | None = None


def _generalized_evaluation_is_finite(generalized: _GeneralizedEvaluation) -> bool:
    return (
        _array_is_finite(generalized.coefficients)
        and _matrix_is_finite(generalized.jacobian)
        and _array_is_finite(generalized.omega_coefficients)
        and (generalized.parameter_coefficients is None or _array_is_finite(generalized.parameter_coefficients))
    )


class ContinuationFreeFrequencySolver:
    """Arc-length continuation with unknown response frequency and one true parameter."""

    def __init__(
        self,
        model: FreeFrequencySecondOrderTimeModel,
        config: ContinuationFreeFrequencyConfig | None = None,
    ) -> None:
        if not isinstance(model, FreeFrequencySecondOrderTimeModel):
            raise TypeError("ContinuationFreeFrequencySolver requires a FreeFrequencySecondOrderTimeModel")
        self.model = model
        self.config = config or ContinuationFreeFrequencyConfig()
        _validate_continuation_config(self.config)
        self._linear_solver = _validate_linear_solver(self.config.linear_solver)
        _validate_positive_scale("parameter_scale", self.config.parameter_scale)
        _validate_positive_scale("constraint_tolerance", self.config.constraint_tolerance)
        _validate_optional_positive_scale("max_parameter_step", self.config.max_parameter_step)
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
            (self.config.omega_scale, self.config.parameter_scale),
        )
        self._initial_constraint_index = _constraint_flat_index(
            self.config.initial_constraint,
            self.prepared.context.order,
            self.model.n_dof,
        )
        if isinstance(self.config.phase_condition, HarmonicCoefficientConstraint):
            self._phase_constraint_index: int | None = _constraint_flat_index(
                self.config.phase_condition,
                self.prepared.context.order,
                self.model.n_dof,
            )
        elif isinstance(self.config.phase_condition, ReferencePhaseCondition):
            self._phase_constraint_index = None
        else:
            raise TypeError("phase_condition must be HarmonicCoefficientConstraint or ReferencePhaseCondition")
        self._coefficient_dt_map = coefficient_derivative_maps(self.prepared.context.harmonics)[0]

    def _emit_progress(self, message: str) -> None:
        if self.config.progress_callback is not None:
            self.config.progress_callback(message)

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

    def _residual_terms(
        self,
        coeff_line: NDArray[np.float64],
        generalized: _GeneralizedEvaluation,
        omega: float,
        parameter: float,
        powered: _PoweredEvaluation | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        active_powered = powered or self.prepared.evaluate_powered(omega, derivative=False)
        forcing_coefficients = active_powered.forcing
        linear_coefficients = active_powered.operator @ coeff_line
        residual = forcing_coefficients - generalized.coefficients - linear_coefficients
        return residual, forcing_coefficients, generalized.coefficients, linear_coefficients

    def _jacobian(
        self,
        generalized: _GeneralizedEvaluation,
        omega: float,
        parameter: float,
        powered: _PoweredEvaluation | None = None,
    ) -> _HBJacobian:
        active_powered = powered or self.prepared.evaluate_powered(omega, derivative=False)
        return _combine_linear_and_nonlinear_jacobians(
            active_powered.operator,
            generalized.jacobian,
            self._linear_solver,
        )

    def _evaluate_generalized(
        self,
        coefficients: NDArray[np.float64],
        omega: float,
        parameter: float,
        *,
        include_parameter: bool,
    ) -> _GeneralizedEvaluation:
        force_dofs = _validated_dofs("residual_force_dofs", self.model.residual_force_dofs, self.model.n_dof)
        coordinate_dofs = _validated_dofs(
            "residual_coordinate_dofs",
            self.model.residual_coordinate_dofs,
            self.model.n_dof,
        )
        local_x, local_dx, local_ddx = _evaluate_local_state(
            coefficients,
            coordinate_dofs,
            self.prepared.hb_item,
            self.prepared.hb_item_dt,
            self.prepared.hb_item_ddt,
        )
        local_force = self.model.local_residual_force(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            omega,
            parameter,
        )
        generalized_coefficients = _local_samples_to_global_coefficients(
            local_force,
            force_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local residual force",
        )
        local_jacobian = self.model.local_residual_jacobian(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            omega,
            parameter,
        )
        generalized_jacobian = assemble_hb_jacobian_from_local_matrices(
            local_jacobian,
            force_dofs,
            coordinate_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local residual",
            self._linear_solver,
        )
        local_omega = self.model.local_residual_omega_derivative(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            omega,
            parameter,
        )
        omega_coefficients = _local_samples_to_global_coefficients(
            local_omega,
            force_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local residual omega derivative",
        )
        parameter_coefficients = None
        if include_parameter:
            local_parameter = self.model.local_residual_parameter_derivative(
                self.prepared.t,
                local_x,
                local_dx,
                local_ddx,
                omega,
                parameter,
            )
            parameter_coefficients = _local_samples_to_global_coefficients(
                local_parameter,
                force_dofs,
                self.prepared.context,
                self.prepared.t.size,
                self.model.n_dof,
                "local residual parameter derivative",
            )
        return _GeneralizedEvaluation(
            generalized_coefficients,
            generalized_jacobian,
            omega_coefficients,
            parameter_coefficients,
        )

    def _omega_column(
        self,
        coeff_line: NDArray[np.float64],
        generalized: _GeneralizedEvaluation,
        omega: float,
        parameter: float,
        powered: _PoweredEvaluation | None = None,
    ) -> NDArray[np.float64]:
        active_powered = powered or self.prepared.evaluate_powered(omega, derivative=True)
        if active_powered.forcing_derivative is None or active_powered.operator_derivative is None:
            raise ValueError("powered evaluation does not include omega derivatives")
        return (
            active_powered.forcing_derivative
            - generalized.omega_coefficients
            - active_powered.operator_derivative @ coeff_line
        )

    def _parameter_column(
        self,
        generalized: _GeneralizedEvaluation,
    ) -> NDArray[np.float64]:
        if generalized.parameter_coefficients is None:
            raise ValueError("generalized evaluation does not include parameter coefficients")
        return -generalized.parameter_coefficients

    def _fixed_constraint_residual(
        self,
        coeff_line: NDArray[np.float64],
        constraint: HarmonicCoefficientConstraint,
        constraint_index: int,
    ) -> float:
        return float(coeff_line[constraint_index] - constraint.value)

    def _fixed_constraint_row(
        self,
        width: int,
        constraint_index: int,
    ) -> sparse.csr_matrix | NDArray[np.float64]:
        if self._linear_solver == "dense":
            row = np.zeros((1, width), dtype=np.float64)
            row[0, constraint_index] = 1.0
            return row
        return sparse.csr_matrix(
            (
                np.array([1.0], dtype=np.float64),
                (
                    np.array([0], dtype=np.int64),
                    np.array([constraint_index], dtype=np.int64),
                ),
            ),
            shape=(1, width),
        )

    def _phase_constraint_row_vector(self, phase_reference: NDArray[np.float64]) -> NDArray[np.float64]:
        if isinstance(self.config.phase_condition, HarmonicCoefficientConstraint):
            row = np.zeros(phase_reference.shape, dtype=np.float64)
            if self._phase_constraint_index is None:
                raise RuntimeError("fixed phase constraint index was not initialized")
            row[self._phase_constraint_index] = 1.0
            return row
        return _reference_phase_row(
            phase_reference,
            self._coefficient_dt_map,
            self.prepared.context.order,
            self.model.n_dof,
        )

    def _phase_constraint_residual(
        self,
        coeff_line: NDArray[np.float64],
        phase_reference: NDArray[np.float64],
    ) -> float:
        if isinstance(self.config.phase_condition, HarmonicCoefficientConstraint):
            if self._phase_constraint_index is None:
                raise RuntimeError("fixed phase constraint index was not initialized")
            return self._fixed_constraint_residual(
                coeff_line,
                self.config.phase_condition,
                self._phase_constraint_index,
            )
        return float(self._phase_constraint_row_vector(phase_reference) @ coeff_line)

    def _phase_constraint_row(
        self,
        width: int,
        phase_reference: NDArray[np.float64],
    ) -> sparse.csr_matrix | NDArray[np.float64]:
        row = self._phase_constraint_row_vector(phase_reference)
        if row.size != width:
            raise ValueError(f"phase reference width mismatch: expected {width}, got {row.size}")
        if self._linear_solver == "dense":
            return row.reshape(1, width)
        return sparse.csr_matrix(row.reshape(1, width))

    def _initial_matrix(
        self,
        jacobian: _HBJacobian,
        omega_column: NDArray[np.float64],
        constraint_row: sparse.spmatrix | NDArray[np.float64],
    ) -> _HBJacobian:
        size = jacobian.shape[0]
        if self._linear_solver == "dense":
            matrix = np.zeros((size + 1, size + 1), dtype=np.float64, order="F")
            matrix[:size, :size] = np.asarray(jacobian, dtype=np.float64)
            matrix[:size, size] = -np.asarray(omega_column, dtype=np.float64).reshape(size)
            matrix[size, :size] = np.asarray(constraint_row, dtype=np.float64).reshape(size)
            return matrix
        top = sparse.hstack(
            (
                jacobian,
                -sparse.csc_matrix(np.asarray(omega_column, dtype=np.float64).reshape(size, 1)),
            ),
            format="csc",
        )
        constraint = sparse.hstack((constraint_row, sparse.csr_matrix((1, 1))), format="csr")
        return sparse.vstack((top, constraint), format="csc")

    def _arc_matrix(
        self,
        jacobian: _HBJacobian,
        omega_column: NDArray[np.float64],
        parameter_column: NDArray[np.float64],
        constraint_row: sparse.spmatrix | NDArray[np.float64],
        arc_row: NDArray[np.float64],
    ) -> _HBJacobian:
        size = jacobian.shape[0]
        if self._linear_solver == "dense":
            matrix = np.zeros((size + 2, size + 2), dtype=np.float64, order="F")
            matrix[:size, :size] = -np.asarray(jacobian, dtype=np.float64)
            matrix[:size, size] = np.asarray(omega_column, dtype=np.float64).reshape(size)
            matrix[:size, size + 1] = np.asarray(parameter_column, dtype=np.float64).reshape(size)
            matrix[size, :size] = np.asarray(constraint_row, dtype=np.float64).reshape(size)
            matrix[size + 1, :] = np.asarray(arc_row, dtype=np.float64).reshape(size + 2)
            return matrix
        top = sparse.hstack(
            (
                -jacobian,
                sparse.csc_matrix(np.asarray(omega_column, dtype=np.float64).reshape(size, 1)),
                sparse.csc_matrix(np.asarray(parameter_column, dtype=np.float64).reshape(size, 1)),
            ),
            format="csc",
        )
        constraint = sparse.hstack((constraint_row, sparse.csr_matrix((1, 2))), format="csr")
        arc_sparse = sparse.csr_matrix(np.asarray(arc_row, dtype=np.float64).reshape(1, size + 2))
        return sparse.vstack((top, constraint, arc_sparse), format="csc")

    def _solve_initial_free(
        self,
        coeff_line: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[
        NDArray[np.float64],
        float,
        NDArray[np.float64],
        _GeneralizedEvaluation,
        _HBJacobian,
        _PoweredEvaluation,
        ContinuationFreeFrequencyStepLog,
    ]:
        config = self.config
        epoch = 1
        residual_vector = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        residual_stats = _residual_stats(residual_vector, (), config.residual_floor)
        constraint_residual = np.inf
        delta = np.full(coeff_line.size + 1, np.inf, dtype=np.float64)
        coeff = None
        generalized = None
        jacobian = None

        while epoch < config.max_epoch and not _free_converged(
            residual_stats.relative_residual,
            constraint_residual,
            float(np.max(np.abs(delta))),
            config.res_tolerance,
            config.constraint_tolerance,
            config.delta_tolerance,
        ):
            coeff = _coefficient_matrix(coeff_line, self.prepared.context.order, self.model.n_dof)
            generalized = self._evaluate_generalized(coeff, omega, parameter, include_parameter=False)
            powered = self.prepared.evaluate_powered(omega, derivative=True)
            jacobian = self._jacobian(generalized, omega, parameter, powered)
            residual_terms = self._residual_terms(coeff_line, generalized, omega, parameter, powered)
            residual_vector = residual_terms[0]
            residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
            omega_column = self._omega_column(coeff_line, generalized, omega, parameter, powered)
            constraint_residual = self._fixed_constraint_residual(
                coeff_line,
                config.initial_constraint,
                self._initial_constraint_index,
            )
            initial_constraint_row = self._fixed_constraint_row(jacobian.shape[0], self._initial_constraint_index)
            initial_matrix = self._initial_matrix(jacobian, omega_column, initial_constraint_row)
            rhs = np.concatenate((residual_vector, np.array([-constraint_residual], dtype=np.float64)))
            delta = _solve_linear_system(initial_matrix, rhs, self._linear_solver)
            coeff_line = coeff_line + delta[:-1]
            omega = float(omega + delta[-1])
            epoch += 1

        coeff = _coefficient_matrix(coeff_line, self.prepared.context.order, self.model.n_dof)
        generalized = self._evaluate_generalized(coeff, omega, parameter, include_parameter=True)
        powered = self.prepared.evaluate_powered(omega, derivative=True)
        jacobian = self._jacobian(generalized, omega, parameter, powered)
        residual_terms = self._residual_terms(coeff_line, generalized, omega, parameter, powered)
        residual_vector = residual_terms[0]
        residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
        constraint_residual = self._fixed_constraint_residual(
            coeff_line,
            config.initial_constraint,
            self._initial_constraint_index,
        )
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
        return coeff_line, float(omega), coeff, generalized, jacobian, powered, log

    def _orient_initial_tangent(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        oriented = np.asarray(tangent, dtype=np.float64)
        direction = 1.0 if self.config.initial_direction == "up" else -1.0
        if oriented[-1] * direction < 0.0:
            oriented = -oriented
        return self._arc_metric.normalize(oriented)

    def _initial_tangent_free(
        self,
        coeff_line: NDArray[np.float64],
        generalized: _GeneralizedEvaluation,
        omega: float,
        parameter: float,
        jacobian: _HBJacobian,
        powered: _PoweredEvaluation,
    ) -> NDArray[np.float64]:
        omega_column = self._omega_column(coeff_line, generalized, omega, parameter, powered)
        parameter_column = self._parameter_column(generalized)
        phase_row = self._phase_constraint_row(jacobian.shape[0], coeff_line)
        tangent_matrix = self._initial_matrix(jacobian, omega_column, phase_row)
        rhs = np.concatenate((parameter_column, np.array([0.0], dtype=np.float64)))
        tangent_q_omega = _solve_linear_system(tangent_matrix, rhs, self._linear_solver)
        return self._orient_initial_tangent(np.concatenate((tangent_q_omega, np.array([1.0], dtype=np.float64))))

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

        coeff_line, omega, coeff, generalized, jacobian, powered, initial_log = self._solve_initial_free(
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

        y0 = np.concatenate((coeff_line, np.array([omega, parameter], dtype=np.float64)))

        omega_history: list[float] = [float(omega)] if initial_log.converged else []
        parameter_history: list[float] = [float(parameter)] if initial_log.converged else []
        coefficient_history: list[NDArray[np.float64]] = [coeff.copy()] if initial_log.converged else []
        logs: list[ContinuationFreeFrequencyStepLog] = []
        if not initial_log.converged or config.max_steps == 0:
            return self._build_result(
                y0,
                coefficient_history,
                omega_history,
                parameter_history,
                logs,
                initial_log,
            )

        tangent = self._initial_tangent_free(coeff_line, generalized, omega, parameter, jacobian, powered)
        arc_length_step = float(config.s_initial)
        shrink_count = 0

        for step in range(1, config.max_steps + 1):
            while True:
                epoch = 1
                y = y0 + arc_length_step * tangent
                phase_reference = y0[:-2].copy()
                phase_row = self._phase_constraint_row(self.model.n_dof * order, phase_reference)
                coeff_line = y[:-2].copy()
                omega = float(y[-2])
                parameter = float(y[-1])
                residual_vector = np.full(self.model.n_dof * order, np.inf, dtype=np.float64)
                residual_stats = _residual_stats(residual_vector, (), config.residual_floor)
                constraint_residual = np.inf
                delta = np.full(self.model.n_dof * order + 2, np.inf, dtype=np.float64)
                j_arc_lu = None
                nonfinite_trial = not _array_is_finite(y)

                while (
                    not nonfinite_trial
                    and epoch < config.max_epoch
                    and not _free_converged(
                        residual_stats.relative_residual,
                        constraint_residual,
                        float(np.max(np.abs(delta))),
                        config.res_tolerance,
                        config.constraint_tolerance,
                        config.delta_tolerance,
                    )
                ):
                    if not _array_is_finite(coeff_line) or not np.isfinite(omega) or not np.isfinite(parameter):
                        nonfinite_trial = True
                        break
                    coeff = _coefficient_matrix(coeff_line, self.prepared.context.order, self.model.n_dof)
                    generalized = self._evaluate_generalized(coeff, omega, parameter, include_parameter=True)
                    if not _generalized_evaluation_is_finite(generalized):
                        nonfinite_trial = True
                        break
                    powered = self.prepared.evaluate_powered(omega, derivative=True)
                    jacobian = self._jacobian(generalized, omega, parameter, powered)
                    if not _matrix_is_finite(jacobian):
                        nonfinite_trial = True
                        break
                    residual_terms = self._residual_terms(coeff_line, generalized, omega, parameter, powered)
                    residual_vector = residual_terms[0]
                    if not all(_array_is_finite(term) for term in residual_terms):
                        residual_stats = _ResidualStats(np.inf, np.inf)
                        nonfinite_trial = True
                        break
                    residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
                    omega_column = self._omega_column(coeff_line, generalized, omega, parameter, powered)
                    parameter_column = self._parameter_column(generalized)
                    constraint_residual = self._phase_constraint_residual(coeff_line, phase_reference)
                    if (
                        not _array_is_finite(omega_column)
                        or not _array_is_finite(parameter_column)
                        or not np.isfinite(constraint_residual)
                    ):
                        nonfinite_trial = True
                        break
                    r_arc = np.concatenate(
                        (
                            residual_vector,
                            np.array(
                                [
                                    constraint_residual,
                                    self._arc_metric.inner(y - y0, tangent) - arc_length_step,
                                ],
                                dtype=np.float64,
                            ),
                        )
                    )
                    if not _array_is_finite(r_arc):
                        nonfinite_trial = True
                        break
                    j_arc_v = self._arc_matrix(
                        jacobian,
                        omega_column,
                        parameter_column,
                        phase_row,
                        self._arc_metric.constraint_row(tangent),
                    )
                    if not _matrix_is_finite(j_arc_v):
                        nonfinite_trial = True
                        break
                    j_arc_lu = _factorize_linear_system(j_arc_v, self._linear_solver)
                    delta = j_arc_lu.solve(r_arc)
                    if not _array_is_finite(delta):
                        nonfinite_trial = True
                        break
                    y = y - delta
                    coeff_line = y[:-2].copy()
                    omega = float(y[-2])
                    parameter = float(y[-1])
                    epoch += 1

                max_delta = float(np.max(np.abs(delta))) if _array_is_finite(delta) else np.inf
                converged = bool(
                    not nonfinite_trial
                    and _free_converged(
                        residual_stats.relative_residual,
                        constraint_residual,
                        max_delta,
                        config.res_tolerance,
                        config.constraint_tolerance,
                        config.delta_tolerance,
                    )
                )
                parameter_step = abs(float(parameter) - float(y0[-1])) if np.isfinite(parameter) else np.inf
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
                    next_phase_row = self._phase_constraint_row(jacobian.shape[0], coeff_line)
                    tangent_matrix = self._arc_matrix(
                        jacobian,
                        omega_column,
                        parameter_column,
                        next_phase_row,
                        self._arc_metric.constraint_row(tangent),
                    )
                    tangent_candidate = _solve_linear_system(
                        tangent_matrix,
                        np.concatenate((np.zeros(jacobian.shape[0] + 1), np.array([1.0], dtype=np.float64))),
                        self._linear_solver,
                    )
                    tangent = self._arc_metric.normalize(tangent_candidate)
                    coeff = _coefficient_matrix(coeff_line, self.prepared.context.order, self.model.n_dof)
                    omega_history.append(float(omega))
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

        return self._build_result(
            y0,
            coefficient_history,
            omega_history,
            parameter_history,
            logs,
            initial_log,
        )

    def _build_result(
        self,
        y0: NDArray[np.float64],
        coefficient_history: list[NDArray[np.float64]],
        omega_history: list[float],
        parameter_history: list[float],
        logs: list[ContinuationFreeFrequencyStepLog],
        initial_log: ContinuationFreeFrequencyStepLog,
    ) -> ContinuationFreeFrequencyResult:
        final_coeff = _coefficient_matrix(y0[:-2], self.prepared.context.order, self.model.n_dof)
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


def _reference_phase_row(
    phase_reference: NDArray[np.float64],
    coefficient_dt_map: NDArray[np.float64],
    order: int,
    n_dof: int,
) -> NDArray[np.float64]:
    reference = np.asarray(phase_reference, dtype=np.float64).reshape((order, n_dof), order="F")
    if coefficient_dt_map.shape != (order, order):
        raise ValueError(f"coefficient_dt_map must have shape {(order, order)}, got {coefficient_dt_map.shape}")
    row = (coefficient_dt_map @ reference).reshape(-1, order="F")
    row_norm = float(np.linalg.norm(row))
    if row_norm == 0.0:
        raise ValueError("reference phase condition requires a nonzero time-varying reference state")
    return row / row_norm


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
