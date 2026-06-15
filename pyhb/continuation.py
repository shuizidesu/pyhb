"""Newton and arc-length continuation for second-order HB models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from .harmonics import coefficient_matrix_from_fft, flatten_coefficients, generate_hb_items, stack_fft_coefficients, unflatten_coefficients
from .hb_operators import HBContext, build_full_fft_nonlinear_harmonics, harmonic_integral_matrices
from .models import SecondOrderTimeModel


@dataclass(frozen=True)
class ContinuationConfig:
    sample_fft: int = 2**11
    harmonics: tuple[float, ...] = tuple(float(v) for v in range(1, 11))
    nonlinear_harmonics: tuple[float, ...] | None = None
    frequency_resolution: float = 1.0
    frequency_tolerance: float = 1e-10
    s3_method: str = "fast"
    s3_quadrature_samples: int | None = None
    init_omega: float = 300.0
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


@dataclass
class _PreparedProblem:
    model: SecondOrderTimeModel
    context: HBContext
    t: NDArray[np.float64]
    hb_item: NDArray[np.float64]
    hb_item_dt: NDArray[np.float64]
    hb_item_ddt: NDArray[np.float64]
    operator_blocks: dict[float, sparse.csc_matrix]
    forcing_blocks: dict[float, NDArray[np.float64]]


@dataclass(frozen=True)
class _ResidualStats:
    relative_residual: float
    max_residual: float


class ContinuationSolver:
    """Generic single-parameter arc-length continuation solver."""

    def __init__(self, model: SecondOrderTimeModel, config: ContinuationConfig | None = None) -> None:
        self.model = model
        self.config = config or ContinuationConfig()
        _validate_positive_scale("q_scale", self.config.q_scale)
        _validate_positive_scale("omega_scale", self.config.omega_scale)
        _validate_optional_positive_scale("max_parameter_step", self.config.max_parameter_step)
        _validate_positive_scale("parameter_step_safety", self.config.parameter_step_safety)
        _validate_positive_scale("residual_floor", self.config.residual_floor)
        self.prepared = self._prepare()

    def _prepare(self) -> _PreparedProblem:
        config = self.config
        active_nonlinear_harmonics = config.nonlinear_harmonics or build_full_fft_nonlinear_harmonics(
            config.sample_fft,
            config.frequency_resolution,
        )
        context = HBContext.build(
            config.harmonics,
            active_nonlinear_harmonics,
            config.sample_fft,
            config.frequency_resolution,
            config.frequency_tolerance,
            config.s3_method,
            config.s3_quadrature_samples,
            config.progress_callback,
        )
        self._emit_progress(
            f"Generating HB basis... period={context.period:.12g}, samples={config.sample_fft}"
        )
        t = np.arange(config.sample_fft, dtype=np.float64) * (context.period / config.sample_fft)
        hb_item, hb_item_dt, hb_item_ddt = generate_hb_items(t, context.harmonics)
        operator_blocks, forcing_blocks = _prepare_structured_parameter_blocks(
            self.model,
            context,
            t,
            config.sample_fft,
        )
        return _PreparedProblem(
            self.model,
            context,
            t,
            hb_item,
            hb_item_dt,
            hb_item_ddt,
            operator_blocks,
            forcing_blocks,
        )

    def _emit_progress(self, message: str) -> None:
        if self.config.progress_callback is not None:
            self.config.progress_callback(message)

    def _evaluate_state(
        self,
        coeff_line: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        coeff = unflatten_coefficients(coeff_line, self.prepared.context.order, self.model.n_dof)
        x = self.prepared.hb_item @ coeff
        dx = self.prepared.hb_item_dt @ coeff
        ddx = self.prepared.hb_item_ddt @ coeff
        return coeff, x, dx, ddx

    def _linear_jacobian(self, parameter: float) -> sparse.csc_matrix:
        return _combine_powered_sparse_blocks(self.prepared.operator_blocks, parameter)

    def _linear_jacobian_derivative(self, parameter: float) -> sparse.csc_matrix:
        return _combine_powered_sparse_blocks(self.prepared.operator_blocks, parameter, derivative=True)

    def _forcing_coefficients(self, parameter: float) -> NDArray[np.float64]:
        return _combine_powered_dense_blocks(self.prepared.forcing_blocks, parameter)

    def _forcing_derivative_coefficients(self, parameter: float) -> NDArray[np.float64]:
        return _combine_powered_dense_blocks(self.prepared.forcing_blocks, parameter, derivative=True)

    def _residual(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return self._residual_terms(coeff_line, x, dx, ddx, parameter)[0]

    def _residual_terms(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        model = self.model
        nonlinear_coefficients = stack_fft_coefficients(
            model.nonlinear_force(self.prepared.t, x, dx, ddx, parameter),
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )
        forcing_coefficients = self._forcing_coefficients(parameter)
        linear_coefficients = self._linear_jacobian(parameter) @ coeff_line
        residual = forcing_coefficients - nonlinear_coefficients - linear_coefficients
        return residual, forcing_coefficients, nonlinear_coefficients, linear_coefficients

    def _jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> sparse.csc_matrix:
        return self._linear_jacobian(parameter) + self._nonlinear_jacobian(x, dx, ddx, parameter)

    def _nonlinear_jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> sparse.csc_matrix:
        context = self.prepared.context
        order = context.order
        size = self.model.n_dof * order
        terms = tuple(self.model.nonlinear_jacobian_terms(self.prepared.t, x, dx, ddx, parameter))
        if not terms:
            return sparse.csc_matrix((size, size), dtype=np.float64)

        tensor_by_variable = {
            "x": context.s3_tensor_x,
            "dx": context.s3_tensor_dx,
            "ddx": context.s3_tensor_ddx,
        }

        force_dofs = np.empty(len(terms), dtype=np.int64)
        coordinate_dofs = np.empty(len(terms), dtype=np.int64)
        variables: list[str] = []
        values = np.empty((self.prepared.t.size, len(terms)), dtype=np.float64)

        for index, term in enumerate(terms):
            if term.variable not in tensor_by_variable:
                raise ValueError(f"unsupported nonlinear Jacobian variable {term.variable!r}")
            if not (0 <= term.force_dof < self.model.n_dof):
                raise ValueError(f"force_dof out of range: {term.force_dof}")
            if not (0 <= term.coordinate_dof < self.model.n_dof):
                raise ValueError(f"coordinate_dof out of range: {term.coordinate_dof}")
            term_values = np.asarray(term.values, dtype=np.float64).reshape(-1)
            if term_values.shape[0] != self.prepared.t.size:
                raise ValueError(
                    "nonlinear Jacobian term values must have one value per time sample; "
                    f"got {term_values.shape[0]}, expected {self.prepared.t.size}"
                )
            force_dofs[index] = term.force_dof
            coordinate_dofs[index] = term.coordinate_dof
            variables.append(term.variable)
            values[:, index] = term_values

        coeffs = coefficient_matrix_from_fft(
            values,
            context.nonlinear_harmonics,
            context.sample_count,
            context.nonlinear_harmonic_indices,
        )
        row_offsets = np.arange(order, dtype=np.int64)
        col_offsets = np.arange(order, dtype=np.int64)
        row_chunks: list[NDArray[np.int64]] = []
        col_chunks: list[NDArray[np.int64]] = []
        data_chunks: list[NDArray[np.float64]] = []

        for variable, s_tensor in tensor_by_variable.items():
            term_indices = np.asarray([index for index, term_variable in enumerate(variables) if term_variable == variable], dtype=np.int64)
            if term_indices.size == 0:
                continue
            blocks = np.einsum("abk,kt->abt", s_tensor, coeffs[:, term_indices])
            term_blocks = np.moveaxis(blocks, 2, 0)
            row_indices = force_dofs[term_indices, None, None] * order + row_offsets[None, :, None]
            col_indices = coordinate_dofs[term_indices, None, None] * order + col_offsets[None, None, :]
            row_chunks.append(np.broadcast_to(row_indices, term_blocks.shape).reshape(-1))
            col_chunks.append(np.broadcast_to(col_indices, term_blocks.shape).reshape(-1))
            data_chunks.append(term_blocks.reshape(-1))

        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.concatenate(data_chunks)
        return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()

    def _parameter_jacobian(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        nonlinear_parameter = stack_fft_coefficients(
            self.model.nonlinear_parameter_derivative(self.prepared.t, x, dx, ddx, parameter),
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )
        parameter_column = (
            self._forcing_derivative_coefficients(parameter)
            - nonlinear_parameter
            - self._linear_jacobian_derivative(parameter) @ coeff_line
        )
        return parameter_column.reshape(-1, 1)

    def _solve_initial(
        self,
        coeff_line: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], StepLog]:
        config = self.config
        epoch = 1
        residual_vector = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        residual_stats = _ResidualStats(np.inf, np.inf)
        delta = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        coeff = x = dx = ddx = None
        jacobian = None

        while (
            epoch < config.max_epoch
            and residual_stats.relative_residual >= config.res_tolerance
            and np.max(np.abs(delta)) >= config.delta_tolerance
        ):
            coeff, x, dx, ddx = self._evaluate_state(coeff_line)
            jacobian = self._jacobian(x, dx, ddx, parameter)
            residual_terms = self._residual_terms(coeff_line, x, dx, ddx, parameter)
            residual_vector = residual_terms[0]
            residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
            delta = _solve_sparse(jacobian, residual_vector)
            coeff_line = coeff_line + delta
            epoch += 1

        coeff, x, dx, ddx = self._evaluate_state(coeff_line)
        jacobian = self._jacobian(x, dx, ddx, parameter)
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
        return coeff_line, coeff, x, dx, jacobian, log

    def _orient_initial_tangent(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        oriented = np.asarray(tangent, dtype=np.float64)
        if oriented[-1] < 0.0:
            oriented = -oriented
        return self._weighted_normalize(oriented)

    def _initial_tangent(self, jacobian: sparse.csc_matrix, parameter_column: NDArray[np.float64]) -> NDArray[np.float64]:
        tangent_q = _solve_sparse(jacobian, parameter_column.reshape(-1))
        return self._orient_initial_tangent(np.concatenate((tangent_q, np.array([1.0], dtype=np.float64))))

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
                raise ValueError(
                    f"initial_coefficients must have length {expected_size}, got {initial.size}"
                )
            coeff_line = initial.reshape(-1)
        else:
            raise ValueError(
                "initial_coefficients must be a 1D coefficient vector or "
                f"a 2D coefficient matrix shaped {expected_shape}, got {initial.shape}"
            )

        parameter = float(config.init_omega if initial_parameter is None else initial_parameter)
        coeff_line, coeff, x, dx, jacobian, initial_log = self._solve_initial(coeff_line, parameter)
        self._emit_progress(
            "Initial computation, "
            f"Epoch = {initial_log.epoch}, RelRes = {initial_log.relative_residual:.6e}, "
            f"MaxRes = {initial_log.max_residual:.6e}, "
            f"Delta = {initial_log.max_delta:.6e}, Omega = {initial_log.omega:.10g}"
        )
        _, _, _, ddx = self._evaluate_state(coeff_line)
        j_parameter = self._parameter_jacobian(coeff_line, x, dx, ddx, parameter)
        tangent = self._initial_tangent(jacobian, j_parameter)
        y0 = np.concatenate((coeff_line, np.array([parameter], dtype=np.float64)))

        parameter_history: list[float] = [float(parameter)] if initial_log.converged else []
        coefficient_history: list[NDArray[np.float64]] = [coeff.copy()] if initial_log.converged else []
        logs: list[StepLog] = []
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
                j_arc_lu = None

                while (
                    epoch < config.max_epoch
                    and residual_stats.relative_residual >= config.res_tolerance
                    and np.max(np.abs(delta)) >= config.delta_tolerance
                ):
                    coeff, x, dx, ddx = self._evaluate_state(coeff_line)
                    jacobian = self._jacobian(x, dx, ddx, parameter)
                    residual_terms = self._residual_terms(coeff_line, x, dx, ddx, parameter)
                    residual_vector = residual_terms[0]
                    residual_stats = _residual_stats(residual_vector, residual_terms[1:], config.residual_floor)
                    j_parameter = self._parameter_jacobian(coeff_line, x, dx, ddx, parameter)
                    r_arc = np.concatenate(
                        (
                            residual_vector,
                            np.array(
                                [self._weighted_inner(y - y0, tangent) - arc_length_step],
                                dtype=np.float64,
                            ),
                        )
                    )
                    j_arc_v = _augmented_arc_matrix(jacobian, j_parameter, self._weighted_constraint_row(tangent))
                    j_arc_lu = splu(j_arc_v)
                    delta = j_arc_lu.solve(r_arc)
                    y = y - delta
                    coeff_line = y[:-1].copy()
                    parameter = float(y[-1])
                    epoch += 1

                max_delta = float(np.max(np.abs(delta)))
                converged = residual_stats.relative_residual <= config.res_tolerance or max_delta <= config.delta_tolerance
                parameter_step = abs(float(parameter) - float(y0[-1]))
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
                    if j_arc_lu is None:
                        raise RuntimeError("converged before assembling arc Jacobian")
                    tangent_candidate = j_arc_lu.solve(
                        np.concatenate((np.zeros(jacobian.shape[0]), np.array([1.0])))
                    )
                    tangent = self._weighted_normalize(tangent_candidate)
                    coeff, _, _, _ = self._evaluate_state(coeff_line)
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

        final_coeff, _, _, _ = self._evaluate_state(y0[:-1])
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

def _validate_positive_scale(name: str, value: float) -> None:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}")


def _validate_optional_positive_scale(name: str, value: float | None) -> None:
    if value is not None:
        _validate_positive_scale(name, float(value))


def _parameter_step_too_large(max_parameter_step: float | None, parameter_step: float) -> bool:
    return max_parameter_step is not None and parameter_step > float(max_parameter_step)


def _shrink_arc_length_for_parameter_step(
    arc_length_step: float,
    parameter_step: float,
    max_parameter_step: float,
    parameter_step_safety: float,
    s_min: float,
) -> float:
    scale = parameter_step_safety * max_parameter_step / parameter_step
    return max(arc_length_step * scale, s_min)


def _prepare_structured_parameter_blocks(
    model: SecondOrderTimeModel,
    context: HBContext,
    t: NDArray[np.float64],
    sample_fft: int,
) -> tuple[dict[float, sparse.csc_matrix], dict[float, NDArray[np.float64]]]:
    mass_basis, damping_basis, stiffness_basis = harmonic_integral_matrices(context.harmonics)
    basis_by_type = {
        "ddx": sparse.csc_matrix(mass_basis),
        "dx": sparse.csc_matrix(damping_basis),
        "x": sparse.csc_matrix(stiffness_basis),
    }

    operator_blocks: dict[float, sparse.csc_matrix] = {}
    linear_terms = tuple(model.linear_operator_terms())
    if not linear_terms:
        raise ValueError("model.linear_operator_terms() must return at least one term")
    for term in linear_terms:
        if term.basis_type not in basis_by_type:
            raise ValueError(f"unsupported linear operator basis_type: {term.basis_type!r}")
        matrix = sparse.csc_matrix(term.matrix, dtype=np.float64)
        expected_matrix_shape = (model.n_dof, model.n_dof)
        if matrix.shape != expected_matrix_shape:
            raise ValueError(f"linear operator matrix must have shape {expected_matrix_shape}, got {matrix.shape}")
        power = _validated_parameter_power(term.parameter_power)
        _add_powered_sparse_block(
            operator_blocks,
            power,
            sparse.kron(matrix, basis_by_type[term.basis_type], format="csc"),
        )

    forcing_blocks: dict[float, NDArray[np.float64]] = {}
    forcing_terms = tuple(model.forcing_terms(t))
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

    return operator_blocks, forcing_blocks


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


def _solve_sparse(matrix: sparse.spmatrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    return splu(matrix.tocsc()).solve(np.asarray(rhs, dtype=np.float64))


def _rms(values: NDArray[np.float64]) -> float:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return 0.0
    return float(np.linalg.norm(vector) / np.sqrt(vector.size))


def _residual_stats(
    residual: NDArray[np.float64],
    scale_terms: tuple[NDArray[np.float64], ...],
    residual_floor: float,
) -> _ResidualStats:
    scale = max((_rms(term) for term in scale_terms), default=0.0)
    scale = max(scale, float(residual_floor))
    return _ResidualStats(
        relative_residual=_rms(residual) / scale,
        max_residual=float(np.max(np.abs(residual))) if np.asarray(residual).size else 0.0,
    )


def _augmented_arc_matrix(
    jacobian: sparse.spmatrix,
    parameter_column: NDArray[np.float64],
    arc_row: NDArray[np.float64],
) -> sparse.csc_matrix:
    size = jacobian.shape[0]
    parameter_sparse = sparse.csc_matrix(np.asarray(parameter_column, dtype=np.float64).reshape(size, 1))
    arc_sparse = sparse.csr_matrix(np.asarray(arc_row, dtype=np.float64).reshape(1, size + 1))
    top = sparse.hstack((-jacobian, parameter_sparse), format="csc")
    return sparse.vstack((top, arc_sparse), format="csc")
