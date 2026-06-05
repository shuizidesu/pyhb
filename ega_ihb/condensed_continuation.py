"""Condensed harmonic-balance solvers for localized nonlinearities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import null_space
from scipy.sparse.linalg import splu

from .harmonics import (
    coefficient_matrix_from_fft,
    flatten_coefficients,
    generate_hb_items,
    stack_fft_coefficients,
    unflatten_coefficients,
)
from .hb_operators import HBContext, build_full_fft_nonlinear_harmonics, harmonic_integral_matrices
from .models import ForcingTerm, LinearOperatorTerm, LocalNonlinearJacobianTerm, SecondOrderTimeModel


_LOOP_REVISIT_EXCLUSION_STEPS = 20
_LOOP_MIN_HISTORY_STEPS = 50
_LOOP_RESTART_RECENT_CANDIDATES = 3


@dataclass(frozen=True)
class CondensedContinuationConfig:
    sample_fft: int = 2**15
    harmonics: tuple[float, ...] = (1.0, 1.2)
    nonlinear_harmonics: tuple[float, ...] | None = None
    frequency_resolution: float = 0.1
    frequency_tolerance: float = 1e-10
    init_omega: float = 140.0
    max_epoch: int = 100
    res_tolerance: float = 5e-9
    delta_tolerance: float = 1e-19
    s_initial: float = 0.1
    s_max: float = 0.5
    s_min: float = 1e-9
    q_scale: float = 1.0
    omega_scale: float = 1.0
    loop_switch_enabled: bool = False
    loop_revisit_tolerance: float = 0.05
    loop_restart_omega_delta: float = 0.1
    loop_restart_min_delta: float = 1e-4
    loop_restart_shrink_limit: int = 8
    max_steps: int = 500
    shrink_limit: int = 20
    s3_method: str = "fast"
    s3_quadrature_samples: int | None = None
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CondensedContinuationLog:
    step: int
    epoch: int
    max_residual: float
    max_delta: float
    omega: float
    arc_length: float
    converged: bool


@dataclass(frozen=True)
class CondensedContinuationLoopEvent:
    step: int
    current_index: int
    matched_index: int
    distance: float
    anchor_index: int
    restart_omega: float
    restarted: bool


@dataclass(frozen=True)
class CondensedContinuationResult:
    parameter_history: NDArray[np.float64]
    coefficient_history: NDArray[np.float64]
    nonlinear_coefficient_history: NDArray[np.float64]
    logs: list[CondensedContinuationLog] = field(default_factory=list)
    loop_events: list[CondensedContinuationLoopEvent] = field(default_factory=list)
    initial_log: CondensedContinuationLog | None = None


@dataclass
class _PreparedCondensedProblem:
    context: HBContext
    t: NDArray[np.float64]
    hb_item: NDArray[np.float64]
    hb_item_dt: NDArray[np.float64]
    hb_item_ddt: NDArray[np.float64]
    nonlinear_dofs: tuple[int, ...]
    nonlinear_force_dofs: tuple[int, ...]
    nonlinear_coordinate_dofs: tuple[int, ...]
    linear_dofs: tuple[int, ...]
    nonlinear_indices: NDArray[np.int64]
    nonlinear_force_local_indices: NDArray[np.int64]
    nonlinear_coordinate_local_indices: NDArray[np.int64]
    linear_indices: NDArray[np.int64]
    operator_ll_blocks: dict[float, sparse.csc_matrix]
    operator_ln_blocks: dict[float, sparse.csc_matrix]
    operator_nl_blocks: dict[float, sparse.csc_matrix]
    operator_nn_blocks: dict[float, sparse.csc_matrix]
    forcing_l_blocks: dict[float, NDArray[np.float64]]
    forcing_n_blocks: dict[float, NDArray[np.float64]]


@dataclass
class _ParameterBlockSet:
    k_ll: sparse.csc_matrix
    k_ln: sparse.csc_matrix
    k_nl: sparse.csc_matrix
    k_nn: sparse.csc_matrix
    f_l: NDArray[np.float64]
    f_n: NDArray[np.float64]


@dataclass
class _ContinuationLinearState:
    condensed_linear: NDArray[np.float64]
    condensed_force: NDArray[np.float64]
    condensed_linear_derivative: NDArray[np.float64]
    condensed_force_derivative: NDArray[np.float64]
    linear_force_solution: NDArray[np.float64]
    linear_coupling_solution: NDArray[np.float64]


@dataclass
class _LoopRestartResult:
    anchor_index: int
    restart_omega: float
    restarted: bool
    nonlinear_vector: NDArray[np.float64] | None = None
    jacobian: NDArray[np.float64] | None = None
    linear_state: _ContinuationLinearState | None = None


class CondensedContinuationSolver:
    """Arc-length continuation solver condensed onto nonlinear DOFs."""

    def __init__(
        self,
        model: SecondOrderTimeModel,
        config: CondensedContinuationConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or CondensedContinuationConfig()
        _validate_positive_scale("q_scale", self.config.q_scale)
        _validate_positive_scale("omega_scale", self.config.omega_scale)
        _validate_positive_scale("loop_restart_omega_delta", self.config.loop_restart_omega_delta)
        _validate_positive_scale("loop_restart_min_delta", self.config.loop_restart_min_delta)
        if self.config.loop_revisit_tolerance <= 0.0 or not np.isfinite(self.config.loop_revisit_tolerance):
            raise ValueError(
                "loop_revisit_tolerance must be a positive finite value, "
                f"got {self.config.loop_revisit_tolerance!r}"
            )
        if self.config.loop_restart_shrink_limit < 1:
            raise ValueError(
                "loop_restart_shrink_limit must be at least 1, "
                f"got {self.config.loop_restart_shrink_limit!r}"
            )
        self.prepared = _prepare_condensed_problem(
            self.model,
            harmonics=self.config.harmonics,
            nonlinear_harmonics=self.config.nonlinear_harmonics,
            sample_fft=self.config.sample_fft,
            frequency_resolution=self.config.frequency_resolution,
            frequency_tolerance=self.config.frequency_tolerance,
            s3_method=self.config.s3_method,
            s3_quadrature_samples=self.config.s3_quadrature_samples,
            progress_callback=self.config.progress_callback,
        )

    def _emit_progress(self, message: str) -> None:
        if self.config.progress_callback is not None:
            self.config.progress_callback(message)

    def _initial_nonlinear_vector(self, initial_coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        order = self.prepared.context.order
        nonlinear_count = len(self.prepared.nonlinear_dofs)
        initial = np.asarray(initial_coefficients, dtype=np.float64)
        if initial.shape != (order, nonlinear_count):
            raise ValueError(
                "initial_coefficients must have shape "
                f"{(order, nonlinear_count)}, got {initial.shape}"
            )
        return flatten_coefficients(initial)

    def _build_linear_state(self, parameter: float) -> _ContinuationLinearState:
        prepared = self.prepared
        blocks = _evaluate_parameter_blocks(prepared, parameter)
        derivative_blocks = _evaluate_parameter_blocks(prepared, parameter, derivative=True)

        lu = splu(blocks.k_ll, permc_spec="COLAMD")
        linear_rhs = np.column_stack((blocks.f_l, blocks.k_ln.toarray()))
        linear_solution = lu.solve(linear_rhs)
        linear_force_solution = linear_solution[:, 0]
        linear_coupling_solution = linear_solution[:, 1:]

        derivative_rhs = np.column_stack(
            (
                derivative_blocks.f_l - derivative_blocks.k_ll @ linear_force_solution,
                derivative_blocks.k_ln.toarray() - derivative_blocks.k_ll @ linear_coupling_solution,
            )
        )
        derivative_solution = lu.solve(derivative_rhs)
        linear_force_derivative_solution = derivative_solution[:, 0]
        linear_coupling_derivative_solution = derivative_solution[:, 1:]

        condensed_linear_raw = blocks.k_nn.toarray() - blocks.k_nl @ linear_coupling_solution
        condensed_force_raw = blocks.f_n - blocks.k_nl @ linear_force_solution
        condensed_linear_derivative = (
            derivative_blocks.k_nn.toarray()
            - derivative_blocks.k_nl @ linear_coupling_solution
            - blocks.k_nl @ linear_coupling_derivative_solution
        )
        condensed_force_derivative_raw = (
            derivative_blocks.f_n
            - derivative_blocks.k_nl @ linear_force_solution
            - blocks.k_nl @ linear_force_derivative_solution
        )
        return _ContinuationLinearState(
            condensed_linear=condensed_linear_raw,
            condensed_force=condensed_force_raw,
            condensed_linear_derivative=condensed_linear_derivative,
            condensed_force_derivative=condensed_force_derivative_raw,
            linear_force_solution=linear_force_solution,
            linear_coupling_solution=linear_coupling_solution,
        )

    def _recover_full_vector(self, nonlinear_vector: NDArray[np.float64], linear_state: _ContinuationLinearState) -> NDArray[np.float64]:
        return _recover_full_vector(nonlinear_vector, self.prepared, linear_state)

    def _recover_full_coefficients(
        self,
        nonlinear_vector: NDArray[np.float64],
        linear_state: _ContinuationLinearState,
    ) -> NDArray[np.float64]:
        full_vector = self._recover_full_vector(nonlinear_vector, linear_state)
        return unflatten_coefficients(full_vector, self.prepared.context.order, self.model.n_dof)

    def _local_nonlinear_values(
        self,
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], tuple[LocalNonlinearJacobianTerm, ...]]:
        force = self.model.local_nonlinear_force(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            parameter,
        )
        terms = self.model.local_nonlinear_jacobian_terms(
            self.prepared.t,
            local_x,
            local_dx,
            local_ddx,
            parameter,
        )
        return _validate_local_force(force, self.prepared), tuple(terms)

    def _local_nonlinear_parameter_coefficients(
        self,
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        samples = np.asarray(
            self.model.local_nonlinear_parameter_derivative(
                self.prepared.t,
                local_x,
                local_dx,
                local_ddx,
                parameter,
            ),
            dtype=np.float64,
        )
        expected_shape = (self.prepared.t.size, len(self.prepared.nonlinear_force_dofs))
        if samples.shape != expected_shape:
            raise ValueError(f"local nonlinear parameter derivative must have shape {expected_shape}, got {samples.shape}")
        return _scatter_local_coefficients(
            _stack_local_coefficients(samples, self.prepared, self.config.sample_fft),
            self.prepared.nonlinear_force_local_indices,
            len(self.prepared.nonlinear_indices),
        )

    def _residual_jacobian_parameter(
        self,
        nonlinear_vector: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], _ContinuationLinearState]:
        linear_state = self._build_linear_state(parameter)
        return self._residual_jacobian_with_linear_state(nonlinear_vector, parameter, linear_state)

    def _residual_jacobian_with_linear_state(
        self,
        nonlinear_vector: NDArray[np.float64],
        parameter: float,
        linear_state: _ContinuationLinearState,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], _ContinuationLinearState]:
        local_x, local_dx, local_ddx = _evaluate_coordinate_states(nonlinear_vector, self.prepared)
        force, terms = self._local_nonlinear_values(local_x, local_dx, local_ddx, parameter)
        nonlinear_force = _scatter_local_coefficients(
            _stack_local_coefficients(force, self.prepared, self.config.sample_fft),
            self.prepared.nonlinear_force_local_indices,
            nonlinear_vector.size,
        )
        nonlinear_parameter = self._local_nonlinear_parameter_coefficients(local_x, local_dx, local_ddx, parameter)
        residual = linear_state.condensed_force - nonlinear_force - linear_state.condensed_linear @ nonlinear_vector
        jacobian = linear_state.condensed_linear + _local_nonlinear_jacobian(terms, self.prepared)
        parameter_column = (
            linear_state.condensed_force_derivative
            - nonlinear_parameter
            - linear_state.condensed_linear_derivative @ nonlinear_vector
        )
        return residual, jacobian, parameter_column.reshape(-1, 1), linear_state

    def _solve_initial(
        self,
        nonlinear_vector: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], _ContinuationLinearState, CondensedContinuationLog]:
        linear_state = self._build_linear_state(parameter)
        residual = np.full_like(nonlinear_vector, np.inf)
        delta = np.full_like(nonlinear_vector, np.inf)
        for epoch in range(1, self.config.max_epoch + 1):
            residual, jacobian, _, linear_state = self._residual_jacobian_with_linear_state(
                nonlinear_vector,
                parameter,
                linear_state,
            )
            delta = np.linalg.solve(jacobian, residual)
            nonlinear_vector = nonlinear_vector + delta
            if np.max(np.abs(residual)) <= self.config.res_tolerance or np.max(np.abs(delta)) <= self.config.delta_tolerance:
                break
        residual, jacobian, _, linear_state = self._residual_jacobian_with_linear_state(
            nonlinear_vector,
            parameter,
            linear_state,
        )
        log = CondensedContinuationLog(
            step=0,
            epoch=epoch,
            max_residual=float(np.max(np.abs(residual))),
            max_delta=float(np.max(np.abs(delta))),
            omega=float(parameter),
            arc_length=0.0,
            converged=bool(
                np.max(np.abs(residual)) <= self.config.res_tolerance
                or np.max(np.abs(delta)) <= self.config.delta_tolerance
            ),
        )
        return nonlinear_vector, jacobian, linear_state, log

    def _orient_initial_tangent(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        oriented = np.asarray(tangent, dtype=np.float64)
        if oriented[-1] < 0.0:
            oriented = -oriented
        return self._weighted_normalize(oriented)

    def _initial_tangent(self, jacobian: NDArray[np.float64], parameter_column: NDArray[np.float64]) -> NDArray[np.float64]:
        j_arc = np.hstack((-jacobian, parameter_column))
        basis = null_space(j_arc)
        if basis.shape[1] == 0:
            raise np.linalg.LinAlgError("arc Jacobian has no numerical null-space")
        return self._orient_initial_tangent(basis[:, 0])

    def _arc_weights(self, size: int) -> NDArray[np.float64]:
        return np.concatenate(
            (
                np.full(size - 1, 1.0 / float(self.config.q_scale) ** 2, dtype=np.float64),
                np.array([1.0 / float(self.config.omega_scale) ** 2], dtype=np.float64),
            )
        )

    def _weighted_constraint_row(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._arc_weights(tangent.size) * tangent

    def _weighted_inner(self, left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        return float(left @ (self._arc_weights(left.size) * right))

    def _weighted_normalize(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        norm = np.sqrt(self._weighted_inner(vector, vector))
        if norm == 0.0:
            raise np.linalg.LinAlgError("arc tangent has zero weighted norm")
        return vector / norm

    def _scaled_state(
        self,
        nonlinear_vector: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return np.concatenate(
            (
                nonlinear_vector / float(self.config.q_scale),
                np.array([float(parameter) / float(self.config.omega_scale)], dtype=np.float64),
            )
        )

    def _minimum_history_distance(
        self,
        nonlinear_vector: NDArray[np.float64],
        parameter: float,
        accepted_nonlinear_vectors: Sequence[NDArray[np.float64]],
        accepted_parameters: Sequence[float],
    ) -> float:
        if not accepted_parameters:
            return np.inf
        current = self._scaled_state(nonlinear_vector, parameter)
        distances = [
            np.linalg.norm(current - self._scaled_state(history_vector, history_parameter))
            for history_vector, history_parameter in zip(accepted_nonlinear_vectors, accepted_parameters)
        ]
        return float(np.min(distances))

    def _find_loop_revisit(
        self,
        nonlinear_vector: NDArray[np.float64],
        parameter: float,
        accepted_nonlinear_vectors: Sequence[NDArray[np.float64]],
        accepted_parameters: Sequence[float],
    ) -> tuple[int, float] | None:
        history_count = len(accepted_parameters)
        if history_count < _LOOP_MIN_HISTORY_STEPS:
            return None
        search_count = history_count - _LOOP_REVISIT_EXCLUSION_STEPS
        if search_count <= 0:
            return None
        current = self._scaled_state(nonlinear_vector, parameter)
        distances = np.asarray(
            [
                np.linalg.norm(current - self._scaled_state(history_vector, history_parameter))
                for history_vector, history_parameter in zip(
                    accepted_nonlinear_vectors[:search_count],
                    accepted_parameters[:search_count],
                )
            ],
            dtype=np.float64,
        )
        matched_index = int(np.argmin(distances))
        distance = float(distances[matched_index])
        if distance <= self.config.loop_revisit_tolerance:
            return matched_index, distance
        return None

    def _try_loop_restart_from_recent_history(
        self,
        accepted_nonlinear_vectors: Sequence[NDArray[np.float64]],
        accepted_parameters: Sequence[float],
    ) -> _LoopRestartResult:
        if not accepted_parameters:
            raise ValueError("cannot restart without accepted parameters")
        parameter_array = np.asarray(accepted_parameters, dtype=np.float64)
        base_parameter = float(np.max(parameter_array))
        first_candidate = max(0, len(accepted_parameters) - _LOOP_RESTART_RECENT_CANDIDATES)
        candidate_indices = tuple(range(len(accepted_parameters) - 1, first_candidate - 1, -1))
        omega_delta = float(self.config.loop_restart_omega_delta)
        restart_omega = base_parameter + omega_delta

        for _ in range(self.config.loop_restart_shrink_limit):
            if omega_delta < self.config.loop_restart_min_delta:
                break
            restart_omega = base_parameter + omega_delta
            for anchor_index in candidate_indices:
                anchor_vector = accepted_nonlinear_vectors[anchor_index]
                restart_vector, restart_jacobian, restart_linear_state, restart_log = self._solve_initial(
                    anchor_vector.copy(),
                    restart_omega,
                )
                if restart_log.converged:
                    return _LoopRestartResult(
                        anchor_index=anchor_index,
                        restart_omega=float(restart_omega),
                        restarted=True,
                        nonlinear_vector=restart_vector,
                        jacobian=restart_jacobian,
                        linear_state=restart_linear_state,
                    )
            omega_delta *= 0.5

        return _LoopRestartResult(
            anchor_index=candidate_indices[0],
            restart_omega=float(restart_omega),
            restarted=False,
        )

    def run(self, initial_coefficients: NDArray[np.float64] | None = None) -> CondensedContinuationResult:
        if initial_coefficients is None:
            raise ValueError("CondensedContinuationSolver.run requires explicit initial_coefficients")
        nonlinear_vector = self._initial_nonlinear_vector(initial_coefficients)
        parameter = float(self.config.init_omega)
        nonlinear_vector, jacobian, linear_state, initial_log = self._solve_initial(nonlinear_vector, parameter)
        if not initial_log.converged:
            raise RuntimeError(
                "initial condensed computation did not converge: "
                f"epoch={initial_log.epoch}, residual={initial_log.max_residual:.6e}, "
                f"delta={initial_log.max_delta:.6e}, omega={initial_log.omega:.10g}"
            )
        self._emit_progress(
            "Initial condensed computation, "
            f"Epoch = {initial_log.epoch}, Res = {initial_log.max_residual:.6e}, "
            f"Delta = {initial_log.max_delta:.6e}, Omega = {initial_log.omega:.10g}"
        )
        _, _, parameter_column, linear_state = self._residual_jacobian_with_linear_state(
            nonlinear_vector,
            parameter,
            linear_state,
        )
        tangent = self._initial_tangent(jacobian, parameter_column)
        y0 = np.concatenate((nonlinear_vector, np.array([parameter], dtype=np.float64)))

        parameter_history: list[float] = []
        coefficient_history: list[NDArray[np.float64]] = []
        nonlinear_coefficient_history: list[NDArray[np.float64]] = []
        accepted_nonlinear_vectors: list[NDArray[np.float64]] = []
        accepted_parameters: list[float] = []
        logs: list[CondensedContinuationLog] = []
        loop_events: list[CondensedContinuationLoopEvent] = []
        arc_length_step = float(self.config.s_initial)
        shrink_count = 0
        j_arc_v = None

        for step in range(1, self.config.max_steps + 1):
            loop_restarted = False
            stop_after_loop_event = False
            y = y0 + arc_length_step * tangent
            nonlinear_vector = y[:-1].copy()
            parameter = float(y[-1])
            residual = np.full_like(nonlinear_vector, np.inf)
            delta = np.full(nonlinear_vector.size + 1, np.inf, dtype=np.float64)
            j_arc_v = None

            for epoch in range(1, self.config.max_epoch + 1):
                residual, jacobian, parameter_column, linear_state = self._residual_jacobian_parameter(
                    nonlinear_vector,
                    parameter,
                )
                j_arc = np.hstack((-jacobian, parameter_column))
                arc_row = self._weighted_constraint_row(tangent)
                r_arc = np.concatenate(
                    (residual, np.array([arc_row @ (y - y0) - arc_length_step], dtype=np.float64))
                )
                j_arc_v = np.vstack((j_arc, arc_row[None, :]))
                delta = np.linalg.solve(j_arc_v, r_arc)
                y = y - delta
                nonlinear_vector = y[:-1].copy()
                parameter = float(y[-1])
                if np.max(np.abs(residual)) <= self.config.res_tolerance or np.max(np.abs(delta)) <= self.config.delta_tolerance:
                    break

            max_residual = float(np.max(np.abs(residual)))
            max_delta = float(np.max(np.abs(delta)))
            converged = max_residual <= self.config.res_tolerance or max_delta <= self.config.delta_tolerance
            step_log = CondensedContinuationLog(
                step=step,
                epoch=epoch,
                max_residual=max_residual,
                max_delta=max_delta,
                omega=float(parameter),
                arc_length=float(arc_length_step),
                converged=bool(converged),
            )
            logs.append(step_log)
            status = "ok" if step_log.converged else "no convergence"
            self._emit_progress(
                f"Times = {step_log.step}, Epoch = {step_log.epoch}, Res = {step_log.max_residual:.6e}, "
                f"Delta = {step_log.max_delta:.6e}, Omega = {step_log.omega:.10g}, "
                f"s = {step_log.arc_length:.6g}, {status}"
            )

            if converged:
                y0 = y.copy()
                if j_arc_v is None:
                    raise RuntimeError("converged before assembling arc Jacobian")
                tangent_candidate = np.linalg.solve(
                    j_arc_v,
                    np.concatenate((np.zeros(jacobian.shape[0]), np.array([1.0]))),
                )
                tangent = self._weighted_normalize(tangent_candidate)
                nonlinear_vector = y0[:-1].copy()
                parameter = float(y0[-1])
                linear_state = self._build_linear_state(parameter)
                coefficients = self._recover_full_coefficients(nonlinear_vector, linear_state)
                nonlinear_coefficients = unflatten_coefficients(
                    nonlinear_vector,
                    self.prepared.context.order,
                    len(self.prepared.nonlinear_dofs),
                )
                parameter_history.append(float(parameter))
                coefficient_history.append(coefficients.copy())
                nonlinear_coefficient_history.append(nonlinear_coefficients.copy())
                accepted_nonlinear_vectors.append(nonlinear_vector.copy())
                accepted_parameters.append(float(parameter))
                shrink_count = 0
                if self.config.loop_switch_enabled:
                    revisit = self._find_loop_revisit(
                        nonlinear_vector,
                        parameter,
                        accepted_nonlinear_vectors,
                        accepted_parameters,
                    )
                    if revisit is not None:
                        matched_index, distance = revisit
                        restart_result = self._try_loop_restart_from_recent_history(
                            accepted_nonlinear_vectors,
                            accepted_parameters,
                        )
                        loop_event = CondensedContinuationLoopEvent(
                            step=step,
                            current_index=len(accepted_parameters) - 1,
                            matched_index=matched_index,
                            distance=distance,
                            anchor_index=restart_result.anchor_index,
                            restart_omega=restart_result.restart_omega,
                            restarted=restart_result.restarted,
                        )
                        loop_events.append(loop_event)
                        if not restart_result.restarted:
                            self._emit_progress(
                                "Loop revisit detected, "
                                f"step = {loop_event.step}, matched = {loop_event.matched_index}, "
                                f"distance = {loop_event.distance:.6e}, "
                                f"anchor = {loop_event.anchor_index}, "
                                f"restart_omega = {loop_event.restart_omega:.10g}, restart failed"
                            )
                            stop_after_loop_event = True
                        else:
                            if (
                                restart_result.nonlinear_vector is None
                                or restart_result.jacobian is None
                                or restart_result.linear_state is None
                            ):
                                raise RuntimeError("successful loop restart did not return a complete state")
                            nonlinear_vector = restart_result.nonlinear_vector.copy()
                            parameter = float(restart_result.restart_omega)
                            linear_state = restart_result.linear_state
                            _, _, parameter_column, linear_state = self._residual_jacobian_with_linear_state(
                                nonlinear_vector,
                                parameter,
                                linear_state,
                            )
                            tangent = self._initial_tangent(restart_result.jacobian, parameter_column)
                            y0 = np.concatenate((nonlinear_vector, np.array([parameter], dtype=np.float64)))
                            coefficients = self._recover_full_coefficients(nonlinear_vector, linear_state)
                            nonlinear_coefficients = unflatten_coefficients(
                                nonlinear_vector,
                                self.prepared.context.order,
                                len(self.prepared.nonlinear_dofs),
                            )
                            parameter_history.append(float(parameter))
                            coefficient_history.append(coefficients.copy())
                            nonlinear_coefficient_history.append(nonlinear_coefficients.copy())
                            accepted_nonlinear_vectors.append(nonlinear_vector.copy())
                            accepted_parameters.append(float(parameter))
                            loop_restarted = True
                            shrink_count = 0
                            self._emit_progress(
                                "Loop revisit detected, "
                                f"step = {loop_event.step}, matched = {loop_event.matched_index}, "
                                f"distance = {loop_event.distance:.6e}, "
                                f"anchor = {loop_event.anchor_index}, "
                                f"restart_omega = {loop_event.restart_omega:.10g}, restarted"
                            )
            else:
                shrink_count += 1

            if stop_after_loop_event:
                break

            if loop_restarted:
                arc_length_step = float(self.config.s_initial)
            elif epoch >= self.config.max_epoch:
                arc_length_step = max(0.5 * arc_length_step, self.config.s_min)
            else:
                arc_length_step = min(2.0 * arc_length_step, self.config.s_max)

            if shrink_count >= self.config.shrink_limit:
                break

        coefficient_history_array = (
            np.asarray(coefficient_history, dtype=np.float64)
            if coefficient_history
            else np.empty((0, self.prepared.context.order, self.model.n_dof), dtype=np.float64)
        )
        nonlinear_history_array = (
            np.asarray(nonlinear_coefficient_history, dtype=np.float64)
            if nonlinear_coefficient_history
            else np.empty((0, self.prepared.context.order, len(self.prepared.nonlinear_dofs)), dtype=np.float64)
        )
        parameter_history_array = np.asarray(parameter_history, dtype=np.float64)
        return CondensedContinuationResult(
            parameter_history=parameter_history_array,
            coefficient_history=coefficient_history_array,
            nonlinear_coefficient_history=nonlinear_history_array,
            logs=logs,
            loop_events=loop_events,
            initial_log=initial_log,
        )


def _prepare_condensed_problem(
    model: SecondOrderTimeModel,
    *,
    harmonics: tuple[float, ...],
    nonlinear_harmonics: tuple[float, ...] | None,
    sample_fft: int,
    frequency_resolution: float,
    frequency_tolerance: float,
    s3_method: str,
    s3_quadrature_samples: int | None,
    progress_callback: Callable[[str], None] | None,
) -> _PreparedCondensedProblem:
    force_dofs = _validate_model_dofs("nonlinear_force_dofs", model.nonlinear_force_dofs, model.n_dof)
    coordinate_dofs = _validate_model_dofs(
        "nonlinear_coordinate_dofs",
        model.nonlinear_coordinate_dofs,
        model.n_dof,
    )
    nonlinear_dofs = _ordered_union(force_dofs, coordinate_dofs)
    if not nonlinear_dofs:
        raise ValueError("condensed models must define at least one nonlinear DOF")
    active_nonlinear_harmonics = nonlinear_harmonics or build_full_fft_nonlinear_harmonics(
        sample_fft,
        frequency_resolution,
    )
    context = HBContext.build(
        harmonics,
        active_nonlinear_harmonics,
        sample_fft,
        frequency_resolution,
        frequency_tolerance,
        s3_method,
        s3_quadrature_samples,
        progress_callback,
    )
    t = np.arange(sample_fft, dtype=np.float64) * (context.period / sample_fft)
    hb_item, hb_item_dt, hb_item_ddt = generate_hb_items(t, context.harmonics)
    nonlinear_set = set(nonlinear_dofs)
    linear_dofs = tuple(dof for dof in range(model.n_dof) if dof not in nonlinear_set)
    nonlinear_indices = _coefficient_indices(nonlinear_dofs, context.order)
    local_position_by_dof = {dof: position for position, dof in enumerate(nonlinear_dofs)}
    nonlinear_force_local_indices = _coefficient_indices(
        tuple(local_position_by_dof[dof] for dof in force_dofs),
        context.order,
    )
    nonlinear_coordinate_local_indices = _coefficient_indices(
        tuple(local_position_by_dof[dof] for dof in coordinate_dofs),
        context.order,
    )
    linear_indices = _coefficient_indices(linear_dofs, context.order)
    (
        operator_ll_blocks,
        operator_ln_blocks,
        operator_nl_blocks,
        operator_nn_blocks,
        forcing_l_blocks,
        forcing_n_blocks,
    ) = _prepare_structured_parameter_blocks(
        model,
        context,
        t,
        sample_fft,
        linear_indices,
        nonlinear_indices,
    )
    return _PreparedCondensedProblem(
        context=context,
        t=t,
        hb_item=hb_item,
        hb_item_dt=hb_item_dt,
        hb_item_ddt=hb_item_ddt,
        nonlinear_dofs=nonlinear_dofs,
        nonlinear_force_dofs=force_dofs,
        nonlinear_coordinate_dofs=coordinate_dofs,
        linear_dofs=linear_dofs,
        nonlinear_indices=nonlinear_indices,
        nonlinear_force_local_indices=nonlinear_force_local_indices,
        nonlinear_coordinate_local_indices=nonlinear_coordinate_local_indices,
        linear_indices=linear_indices,
        operator_ll_blocks=operator_ll_blocks,
        operator_ln_blocks=operator_ln_blocks,
        operator_nl_blocks=operator_nl_blocks,
        operator_nn_blocks=operator_nn_blocks,
        forcing_l_blocks=forcing_l_blocks,
        forcing_n_blocks=forcing_n_blocks,
    )


def _validate_model_dofs(name: str, dofs: Sequence[int], n_dof: int) -> tuple[int, ...]:
    normalized = tuple(int(dof) for dof in dofs)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique DOFs")
    if any(dof < 0 or dof >= n_dof for dof in normalized):
        raise ValueError(f"{name} contains an out-of-range DOF")
    return normalized


def _ordered_union(first: Sequence[int], second: Sequence[int]) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for dof in tuple(first) + tuple(second):
        if dof not in seen:
            values.append(int(dof))
            seen.add(int(dof))
    return tuple(values)


def _validate_positive_scale(name: str, value: float) -> None:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}")


def _prepare_structured_parameter_blocks(
    model: SecondOrderTimeModel,
    context: HBContext,
    t: NDArray[np.float64],
    sample_fft: int,
    linear_indices: NDArray[np.int64],
    nonlinear_indices: NDArray[np.int64],
) -> tuple[
    dict[float, sparse.csc_matrix],
    dict[float, sparse.csc_matrix],
    dict[float, sparse.csc_matrix],
    dict[float, sparse.csc_matrix],
    dict[float, NDArray[np.float64]],
    dict[float, NDArray[np.float64]],
]:
    linear_terms_method = getattr(model, "linear_operator_terms", None)
    if not callable(linear_terms_method):
        raise ValueError("Condensed solvers require model.linear_operator_terms()")
    forcing_terms_method = getattr(model, "forcing_terms", None)
    if not callable(forcing_terms_method):
        raise ValueError("Condensed solvers require model.forcing_terms(t)")

    mass_basis, damping_basis, stiffness_basis = harmonic_integral_matrices(context.harmonics)
    basis_by_type = {
        "ddx": sparse.csc_matrix(mass_basis),
        "dx": sparse.csc_matrix(damping_basis),
        "x": sparse.csc_matrix(stiffness_basis),
    }
    operator_blocks: dict[float, sparse.csc_matrix] = {}
    linear_terms = tuple(linear_terms_method())
    if not linear_terms:
        raise ValueError("model.linear_operator_terms() must return at least one term")
    for term in linear_terms:
        basis_type = term.basis_type
        if basis_type not in basis_by_type:
            raise ValueError(f"unsupported linear operator basis_type: {basis_type!r}")
        matrix = sparse.csc_matrix(term.matrix, dtype=np.float64)
        expected_matrix_shape = (model.n_dof, model.n_dof)
        if matrix.shape != expected_matrix_shape:
            raise ValueError(f"linear operator matrix must have shape {expected_matrix_shape}, got {matrix.shape}")
        power = _validated_parameter_power(term.parameter_power)
        block = sparse.kron(matrix, basis_by_type[basis_type], format="csc")
        _add_powered_sparse_block(operator_blocks, power, block)

    forcing_blocks: dict[float, NDArray[np.float64]] = {}
    forcing_terms = tuple(forcing_terms_method(t))
    if not forcing_terms:
        raise ValueError("model.forcing_terms(t) must return at least one term")
    for term in forcing_terms:
        samples = np.asarray(term.samples, dtype=np.float64)
        expected_samples_shape = (t.size, model.n_dof)
        if samples.shape != expected_samples_shape:
            raise ValueError(f"forcing term samples must have shape {expected_samples_shape}, got {samples.shape}")
        power = _validated_parameter_power(term.parameter_power)
        coefficients = stack_fft_coefficients(
            samples,
            context.harmonics,
            sample_fft,
            context.harmonic_indices,
        )
        _add_powered_dense_block(forcing_blocks, power, coefficients)

    return (
        _slice_powered_matrix_blocks(operator_blocks, linear_indices, linear_indices),
        _slice_powered_matrix_blocks(operator_blocks, linear_indices, nonlinear_indices),
        _slice_powered_matrix_blocks(operator_blocks, nonlinear_indices, linear_indices),
        _slice_powered_matrix_blocks(operator_blocks, nonlinear_indices, nonlinear_indices),
        _slice_powered_vector_blocks(forcing_blocks, linear_indices),
        _slice_powered_vector_blocks(forcing_blocks, nonlinear_indices),
    )


def _validated_parameter_power(power: float) -> float:
    value = float(power)
    if not np.isfinite(value):
        raise ValueError(f"parameter_power must be finite, got {power!r}")
    return value


def _add_powered_sparse_block(
    blocks: dict[float, sparse.csc_matrix],
    power: float,
    block: sparse.csc_matrix,
) -> None:
    if power in blocks:
        blocks[power] = (blocks[power] + block).tocsc()
    else:
        blocks[power] = block


def _add_powered_dense_block(
    blocks: dict[float, NDArray[np.float64]],
    power: float,
    block: NDArray[np.float64],
) -> None:
    if power in blocks:
        blocks[power] = blocks[power] + block
    else:
        blocks[power] = block


def _slice_powered_matrix_blocks(
    blocks: dict[float, sparse.csc_matrix],
    row_indices: NDArray[np.int64],
    col_indices: NDArray[np.int64],
) -> dict[float, sparse.csc_matrix]:
    return {power: block[row_indices, :][:, col_indices].tocsc() for power, block in blocks.items()}


def _slice_powered_vector_blocks(
    blocks: dict[float, NDArray[np.float64]],
    indices: NDArray[np.int64],
) -> dict[float, NDArray[np.float64]]:
    return {power: block[indices] for power, block in blocks.items()}


def _evaluate_parameter_blocks(
    prepared: _PreparedCondensedProblem,
    parameter: float,
    *,
    derivative: bool = False,
) -> _ParameterBlockSet:
    return _ParameterBlockSet(
        k_ll=_combine_powered_sparse_blocks(prepared.operator_ll_blocks, parameter, derivative=derivative),
        k_ln=_combine_powered_sparse_blocks(prepared.operator_ln_blocks, parameter, derivative=derivative),
        k_nl=_combine_powered_sparse_blocks(prepared.operator_nl_blocks, parameter, derivative=derivative),
        k_nn=_combine_powered_sparse_blocks(prepared.operator_nn_blocks, parameter, derivative=derivative),
        f_l=_combine_powered_dense_blocks(prepared.forcing_l_blocks, parameter, derivative=derivative),
        f_n=_combine_powered_dense_blocks(prepared.forcing_n_blocks, parameter, derivative=derivative),
    )


def _combine_powered_sparse_blocks(
    blocks: dict[float, sparse.csc_matrix],
    parameter: float,
    *,
    derivative: bool = False,
) -> sparse.csc_matrix:
    if not blocks:
        raise ValueError("at least one parameter block is required")
    active_parameter = float(parameter)
    result: sparse.csc_matrix | None = None
    for power, block in blocks.items():
        scale = _parameter_derivative_scale(active_parameter, power) if derivative else active_parameter**power
        if scale == 0.0:
            continue
        contribution = block * scale
        result = contribution if result is None else result + contribution
    if result is None:
        return sparse.csc_matrix(next(iter(blocks.values())).shape, dtype=np.float64)
    return result.tocsc()


def _combine_powered_dense_blocks(
    blocks: dict[float, NDArray[np.float64]],
    parameter: float,
    *,
    derivative: bool = False,
) -> NDArray[np.float64]:
    if not blocks:
        raise ValueError("at least one parameter block is required")
    active_parameter = float(parameter)
    result: NDArray[np.float64] | None = None
    for power, block in blocks.items():
        scale = _parameter_derivative_scale(active_parameter, power) if derivative else active_parameter**power
        if scale == 0.0:
            continue
        contribution = scale * block
        result = contribution if result is None else result + contribution
    if result is None:
        return np.zeros_like(next(iter(blocks.values())))
    return result


def _parameter_derivative_scale(parameter: float, power: float) -> float:
    if power == 0.0:
        return 0.0
    return power * parameter ** (power - 1.0)


def _coefficient_indices(dofs: Sequence[int], order: int) -> NDArray[np.int64]:
    if not dofs:
        return np.empty(0, dtype=np.int64)
    offsets = np.arange(order, dtype=np.int64)
    return np.concatenate([int(dof) * order + offsets for dof in dofs]).astype(np.int64)


def _recover_full_vector(
    nonlinear_vector: NDArray[np.float64],
    prepared: _PreparedCondensedProblem,
    linear_state: _ContinuationLinearState,
) -> NDArray[np.float64]:
    full = np.zeros(prepared.context.order * (len(prepared.nonlinear_dofs) + len(prepared.linear_dofs)), dtype=np.float64)
    full[prepared.nonlinear_indices] = nonlinear_vector
    full[prepared.linear_indices] = linear_state.linear_force_solution - linear_state.linear_coupling_solution @ nonlinear_vector
    return full


def _evaluate_coordinate_states(
    nonlinear_vector: NDArray[np.float64],
    prepared: _PreparedCondensedProblem,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    coordinate_vector = nonlinear_vector[prepared.nonlinear_coordinate_local_indices]
    coefficients = unflatten_coefficients(
        coordinate_vector,
        prepared.context.order,
        len(prepared.nonlinear_coordinate_dofs),
    )
    return (
        prepared.hb_item @ coefficients,
        prepared.hb_item_dt @ coefficients,
        prepared.hb_item_ddt @ coefficients,
    )


def _validate_local_force(
    force: NDArray[np.float64],
    prepared: _PreparedCondensedProblem,
) -> NDArray[np.float64]:
    force = np.asarray(force, dtype=np.float64)
    expected_force_shape = (prepared.t.size, len(prepared.nonlinear_force_dofs))
    if force.shape != expected_force_shape:
        raise ValueError(f"local nonlinear force must have shape {expected_force_shape}, got {force.shape}")
    return force


def _scatter_local_coefficients(
    local_coefficients: NDArray[np.float64],
    local_indices: NDArray[np.int64],
    size: int,
) -> NDArray[np.float64]:
    values = np.zeros(size, dtype=np.float64)
    values[local_indices] = local_coefficients
    return values


def _stack_local_coefficients(
    values: NDArray[np.float64],
    prepared: _PreparedCondensedProblem,
    sample_fft: int,
) -> NDArray[np.float64]:
    return stack_fft_coefficients(
        values,
        prepared.context.harmonics,
        sample_fft,
        prepared.context.harmonic_indices,
    )


def _local_nonlinear_jacobian(
    terms: tuple[LocalNonlinearJacobianTerm, ...],
    prepared: _PreparedCondensedProblem,
) -> NDArray[np.float64]:
    context = prepared.context
    order = context.order
    jacobian = np.zeros((len(prepared.nonlinear_dofs) * order, len(prepared.nonlinear_dofs) * order), dtype=np.float64)
    if not terms:
        return jacobian

    tensor_by_variable = {
        "x": context.s3_tensor_x,
        "dx": context.s3_tensor_dx,
        "ddx": context.s3_tensor_ddx,
    }

    force_positions = (prepared.nonlinear_force_local_indices[::order] // order).astype(np.int64)
    coordinate_positions = (prepared.nonlinear_coordinate_local_indices[::order] // order).astype(np.int64)
    variables: list[str] = []
    rows = np.empty(len(terms), dtype=np.int64)
    cols = np.empty(len(terms), dtype=np.int64)
    values = np.empty((prepared.t.size, len(terms)), dtype=np.float64)
    for index, term in enumerate(terms):
        if term.variable not in tensor_by_variable:
            raise ValueError(f"unsupported local nonlinear Jacobian variable {term.variable!r}")
        if not (0 <= term.force_index < force_positions.size):
            raise ValueError(f"local force_index out of range: {term.force_index}")
        if not (0 <= term.coordinate_index < coordinate_positions.size):
            raise ValueError(f"local coordinate_index out of range: {term.coordinate_index}")
        term_values = np.asarray(term.values, dtype=np.float64).reshape(-1)
        if term_values.shape[0] != prepared.t.size:
            raise ValueError(
                "local nonlinear Jacobian term values must have one value per time sample; "
                f"got {term_values.shape[0]}, expected {prepared.t.size}"
            )
        variables.append(term.variable)
        rows[index] = force_positions[term.force_index]
        cols[index] = coordinate_positions[term.coordinate_index]
        values[:, index] = term_values

    coeffs = coefficient_matrix_from_fft(
        values,
        context.nonlinear_harmonics,
        context.sample_count,
        context.nonlinear_harmonic_indices,
    )
    for variable, tensor in tensor_by_variable.items():
        term_indices = np.asarray([index for index, term_variable in enumerate(variables) if term_variable == variable], dtype=np.int64)
        if term_indices.size == 0:
            continue
        blocks = np.einsum("abk,kt->abt", tensor, coeffs[:, term_indices])
        for block_index, term_index in enumerate(term_indices):
            row_slice = slice(rows[term_index] * order, (rows[term_index] + 1) * order)
            col_slice = slice(cols[term_index] * order, (cols[term_index] + 1) * order)
            jacobian[row_slice, col_slice] += blocks[:, :, block_index]
    return jacobian
