"""Shared internal helpers for HB continuation solvers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from .harmonics import coefficient_matrix_from_fft, generate_hb_items, stack_fft_coefficients
from .hb_operators import FrequencyGrid, HBContext, harmonic_integral_matrices
from .models import ForcingTerm, LinearOperatorTerm, NonlinearJacobianTerm


class _StructuredHBModel(Protocol):
    @property
    def n_dof(self) -> int: ...

    def linear_operator_terms(self) -> Sequence[LinearOperatorTerm]: ...

    def forcing_terms(self, t: NDArray[np.float64]) -> Sequence[ForcingTerm]: ...


@dataclass
class _PreparedProblem:
    model: _StructuredHBModel
    context: HBContext
    t: NDArray[np.float64]
    hb_item: NDArray[np.float64]
    hb_item_dt: NDArray[np.float64]
    hb_item_ddt: NDArray[np.float64]
    operator_blocks: dict[float, sparse.csc_matrix]
    forcing_blocks: dict[float, NDArray[np.float64]]

    def evaluate_powered(self, parameter: float, *, derivative: bool) -> "_PoweredEvaluation":
        return _PoweredEvaluation(
            operator=_combine_powered_sparse_blocks(self.operator_blocks, parameter),
            forcing=_combine_powered_dense_blocks(self.forcing_blocks, parameter),
            operator_derivative=(
                _combine_powered_sparse_blocks(self.operator_blocks, parameter, derivative=True) if derivative else None
            ),
            forcing_derivative=(
                _combine_powered_dense_blocks(self.forcing_blocks, parameter, derivative=True) if derivative else None
            ),
        )


@dataclass(frozen=True)
class _PoweredEvaluation:
    operator: sparse.csc_matrix
    forcing: NDArray[np.float64]
    operator_derivative: sparse.csc_matrix | None
    forcing_derivative: NDArray[np.float64] | None


@dataclass(frozen=True)
class _ArcLengthMetric:
    weights: NDArray[np.float64]

    @classmethod
    def build(
        cls,
        coefficient_count: int,
        coefficient_scale: float,
        scalar_scales: Sequence[float],
    ) -> "_ArcLengthMetric":
        weights = np.concatenate(
            (
                np.full(coefficient_count, 1.0 / float(coefficient_scale) ** 2, dtype=np.float64),
                np.asarray([1.0 / float(scale) ** 2 for scale in scalar_scales], dtype=np.float64),
            )
        )
        return cls(weights)

    def constraint_row(self, tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = self._vector(tangent)
        return self.weights * vector

    def inner(self, left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        left_vector = self._vector(left)
        right_vector = self._vector(right)
        return float(left_vector @ (self.weights * right_vector))

    def normalize(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        values = self._vector(vector)
        norm = np.sqrt(self.inner(values, values))
        if norm == 0.0:
            raise np.linalg.LinAlgError("arc tangent has zero weighted norm")
        return values / norm

    def _vector(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != self.weights.size:
            raise ValueError(f"arc vector must have length {self.weights.size}, got {vector.size}")
        return vector


@dataclass(frozen=True)
class _ResidualStats:
    relative_residual: float
    max_residual: float


@dataclass(frozen=True)
class _BorderedArcSolve:
    delta: NDArray[np.float64]
    tangent_candidate: NDArray[np.float64]


def assemble_hb_jacobian_from_terms(
    terms: tuple[NonlinearJacobianTerm, ...],
    context: HBContext,
    sample_count: int,
    n_dof: int,
) -> sparse.csc_matrix:
    """Assemble time-domain local Jacobian samples into a global HB Jacobian."""

    order = context.order
    size = n_dof * order
    if not terms:
        return sparse.csc_matrix((size, size), dtype=np.float64)

    s3_by_variable = {
        "x": context.s3,
        "dx": context.s3_dx,
        "ddx": context.s3_ddx,
    }

    force_dofs = np.empty(len(terms), dtype=np.int64)
    coordinate_dofs = np.empty(len(terms), dtype=np.int64)
    variables: list[str] = []
    values = np.empty((sample_count, len(terms)), dtype=np.float64)

    for index, term in enumerate(terms):
        if term.variable not in s3_by_variable:
            raise ValueError(f"unsupported nonlinear Jacobian variable {term.variable!r}")
        if not (0 <= term.force_dof < n_dof):
            raise ValueError(f"force_dof out of range: {term.force_dof}")
        if not (0 <= term.coordinate_dof < n_dof):
            raise ValueError(f"coordinate_dof out of range: {term.coordinate_dof}")
        term_values = np.asarray(term.values, dtype=np.float64).reshape(-1)
        if term_values.shape[0] != sample_count:
            raise ValueError(
                "nonlinear Jacobian term values must have one value per time sample; "
                f"got {term_values.shape[0]}, expected {sample_count}"
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
    local_rows = np.tile(np.arange(order, dtype=np.int64), order)
    local_cols = np.repeat(np.arange(order, dtype=np.int64), order)
    row_chunks: list[NDArray[np.int64]] = []
    col_chunks: list[NDArray[np.int64]] = []
    data_chunks: list[NDArray[np.float64]] = []

    for variable, s3_matrix in s3_by_variable.items():
        term_indices = np.asarray(
            [index for index, term_variable in enumerate(variables) if term_variable == variable],
            dtype=np.int64,
        )
        if term_indices.size == 0:
            continue
        term_blocks_flat = (s3_matrix @ coeffs[:, term_indices]).T
        row_indices = force_dofs[term_indices, None] * order + local_rows[None, :]
        col_indices = coordinate_dofs[term_indices, None] * order + local_cols[None, :]
        row_chunks.append(row_indices.reshape(-1))
        col_chunks.append(col_indices.reshape(-1))
        data_chunks.append(term_blocks_flat.reshape(-1))

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.concatenate(data_chunks)
    return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()


def _coefficient_matrix(
    coeff_line: NDArray[np.float64],
    order: int,
    n_dof: int,
) -> NDArray[np.float64]:
    return np.asarray(coeff_line, dtype=np.float64).reshape((order, n_dof), order="F")


def _validated_dofs(name: str, dofs: Sequence[int], n_dof: int) -> tuple[int, ...]:
    validated = tuple(int(dof) for dof in dofs)
    if any(dof < 0 or dof >= n_dof for dof in validated):
        raise ValueError(f"{name} contains out-of-range DOFs for n_dof={n_dof}: {validated}")
    if len(set(validated)) != len(validated):
        raise ValueError(f"{name} contains duplicate DOFs: {validated}")
    return validated


def _evaluate_local_state(
    coefficients: NDArray[np.float64],
    coordinate_dofs: Sequence[int],
    hb_item: NDArray[np.float64],
    hb_item_dt: NDArray[np.float64],
    hb_item_ddt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    coordinates = np.asarray(tuple(coordinate_dofs), dtype=np.int64)
    local_coefficients = coefficients[:, coordinates]
    return (
        hb_item @ local_coefficients,
        hb_item_dt @ local_coefficients,
        hb_item_ddt @ local_coefficients,
    )


def _local_samples_to_global_coefficients(
    local_samples: NDArray[np.float64],
    force_dofs: Sequence[int],
    context: HBContext,
    sample_count: int,
    n_dof: int,
    label: str,
) -> NDArray[np.float64]:
    force_dofs = _validated_dofs("force_dofs", force_dofs, n_dof)
    samples = np.asarray(local_samples, dtype=np.float64)
    expected_shape = (sample_count, len(force_dofs))
    if samples.shape != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}, got {samples.shape}")
    local_coefficients = stack_fft_coefficients(
        samples,
        context.harmonics,
        sample_count,
        context.harmonic_indices,
    ).reshape((context.order, len(force_dofs)), order="F")
    return _scatter_local_coefficient_matrix(local_coefficients, force_dofs, context.order, n_dof)


def _scatter_local_coefficient_matrix(
    local_coefficients: NDArray[np.float64],
    force_dofs: Sequence[int],
    order: int,
    n_dof: int,
) -> NDArray[np.float64]:
    force_dofs = _validated_dofs("force_dofs", force_dofs, n_dof)
    coefficients = np.asarray(local_coefficients, dtype=np.float64)
    expected_shape = (order, len(force_dofs))
    if coefficients.shape != expected_shape:
        raise ValueError(f"local coefficients must have shape {expected_shape}, got {coefficients.shape}")
    full_coefficients = np.zeros((order, n_dof), dtype=np.float64)
    full_coefficients[:, list(force_dofs)] = coefficients
    return full_coefficients.reshape(-1, order="F")


def _local_jacobian_terms_to_global(
    local_terms: Sequence[object],
    force_dofs: Sequence[int],
    coordinate_dofs: Sequence[int],
    sample_count: int,
    n_dof: int,
    label: str,
) -> tuple[NonlinearJacobianTerm, ...]:
    force_dofs = _validated_dofs("force_dofs", force_dofs, n_dof)
    coordinate_dofs = _validated_dofs("coordinate_dofs", coordinate_dofs, n_dof)
    global_terms: list[NonlinearJacobianTerm] = []
    for term in local_terms:
        variable = getattr(term, "variable")
        if variable not in ("x", "dx", "ddx"):
            raise ValueError(f"unsupported {label} Jacobian variable {variable!r}")
        force_index = int(getattr(term, "force_index"))
        coordinate_index = int(getattr(term, "coordinate_index"))
        if not (0 <= force_index < len(force_dofs)):
            raise ValueError(f"{label} force_index out of range: {force_index}")
        if not (0 <= coordinate_index < len(coordinate_dofs)):
            raise ValueError(f"{label} coordinate_index out of range: {coordinate_index}")
        values = np.asarray(getattr(term, "values"), dtype=np.float64).reshape(-1)
        if values.shape[0] != sample_count:
            raise ValueError(
                f"{label} Jacobian term values must have one value per time sample; "
                f"got {values.shape[0]}, expected {sample_count}"
            )
        global_terms.append(
            NonlinearJacobianTerm(
                force_dofs[force_index],
                variable,
                coordinate_dofs[coordinate_index],
                values,
            )
        )
    return tuple(global_terms)


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


def _prepare_continuation_problem(
    model: _StructuredHBModel,
    *,
    sample_fft: int,
    harmonics: Sequence[float],
    frequency_resolution: float,
    frequency_tolerance: float,
    s3_method: str,
    s3_quadrature_samples: int | None,
    progress_callback: Callable[[str], None] | None,
) -> _PreparedProblem:
    nonlinear_harmonics = _default_nonlinear_harmonics(
        harmonics,
        frequency_resolution,
        frequency_tolerance,
    )
    context = HBContext.build(
        harmonics,
        nonlinear_harmonics,
        sample_fft,
        frequency_resolution,
        frequency_tolerance,
        s3_method,
        s3_quadrature_samples,
        progress_callback,
    )
    if progress_callback is not None:
        progress_callback(f"Generating HB basis... period={context.period:.12g}, samples={sample_fft}")
    t = np.arange(sample_fft, dtype=np.float64) * (context.period / sample_fft)
    hb_item, hb_item_dt, hb_item_ddt = generate_hb_items(t, context.harmonics)
    operator_blocks, forcing_blocks = _prepare_structured_parameter_blocks(
        model,
        context,
        t,
        sample_fft,
    )
    return _PreparedProblem(
        model=model,
        context=context,
        t=t,
        hb_item=hb_item,
        hb_item_dt=hb_item_dt,
        hb_item_ddt=hb_item_ddt,
        operator_blocks=operator_blocks,
        forcing_blocks=forcing_blocks,
    )


def _default_nonlinear_harmonics(
    harmonics: Sequence[float],
    frequency_resolution: float,
    frequency_tolerance: float,
) -> tuple[float, ...]:
    grid = FrequencyGrid(frequency_resolution, frequency_tolerance)
    hb = grid.values_for(harmonics)
    harmonic_indices = grid.indices_for(hb)
    if not harmonic_indices:
        return tuple()
    max_index = max(harmonic_indices)
    return tuple(float(frequency_resolution) * index for index in range(1, 2 * max_index + 1))


def _prepare_structured_parameter_blocks(
    model: _StructuredHBModel,
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
    for linear_term in linear_terms:
        if linear_term.basis_type not in basis_by_type:
            raise ValueError(f"unsupported linear operator basis_type: {linear_term.basis_type!r}")
        matrix = sparse.csc_matrix(linear_term.matrix, dtype=np.float64)
        expected_matrix_shape = (model.n_dof, model.n_dof)
        if matrix.shape != expected_matrix_shape:
            raise ValueError(f"linear operator matrix must have shape {expected_matrix_shape}, got {matrix.shape}")
        power = _validated_omega_power(linear_term.omega_power)
        _add_powered_sparse_block(
            operator_blocks,
            power,
            sparse.kron(matrix, basis_by_type[linear_term.basis_type], format="csc"),
        )

    forcing_blocks: dict[float, NDArray[np.float64]] = {}
    forcing_terms = tuple(model.forcing_terms(t))
    if not forcing_terms:
        raise ValueError("model.forcing_terms(t) must return at least one term")
    for forcing_term in forcing_terms:
        samples = np.asarray(forcing_term.samples, dtype=np.float64)
        expected_samples_shape = (t.size, model.n_dof)
        if samples.shape != expected_samples_shape:
            raise ValueError(f"forcing term samples must have shape {expected_samples_shape}, got {samples.shape}")
        power = _validated_omega_power(forcing_term.omega_power)
        coefficients = stack_fft_coefficients(
            samples,
            context.harmonics,
            sample_fft,
            context.harmonic_indices,
        )
        _add_powered_dense_block(forcing_blocks, power, coefficients)

    return operator_blocks, forcing_blocks


def _validated_omega_power(power: float) -> float:
    value = float(power)
    if not np.isfinite(value):
        raise ValueError(f"omega_power must be finite, got {power!r}")
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


def _array_is_finite(values: NDArray[np.float64] | float) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def _sparse_is_finite(matrix: sparse.spmatrix) -> bool:
    return bool(np.all(np.isfinite(matrix.data)))


def _solve_one_parameter_bordered_arc(
    jacobian: sparse.spmatrix,
    parameter_column: NDArray[np.float64],
    arc_row: NDArray[np.float64],
    residual_vector: NDArray[np.float64],
    arc_residual: float,
) -> _BorderedArcSolve:
    """Solve the one-parameter arc-length bordered system without factoring it.

    This is algebraically equivalent to solving ``[[-J, p], [w_q, w_a]]``,
    but only factors ``J``.  Avoiding the dense arc-length row in the sparse
    LU factorization keeps large full-system problems from suffering excessive
    fill-in.
    """

    size = jacobian.shape[0]
    parameter_column_vector = np.asarray(parameter_column, dtype=np.float64).reshape(size)
    arc_vector = np.asarray(arc_row, dtype=np.float64).reshape(size + 1)
    residual = np.asarray(residual_vector, dtype=np.float64).reshape(size)
    arc_q = arc_vector[:-1]
    arc_parameter = float(arc_vector[-1])

    try:
        lu = splu(jacobian.tocsc())
        residual_solve = lu.solve(-residual)
        parameter_solve = lu.solve(parameter_column_vector)
        denominator = float(arc_q @ parameter_solve + arc_parameter)
        tolerance = (
            100.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(np.linalg.norm(arc_q) * np.linalg.norm(parameter_solve) + abs(arc_parameter)),
            )
        )
        if abs(denominator) <= tolerance:
            raise np.linalg.LinAlgError(
                "arc bordered solve is singular or ill-conditioned; "
                f"denominator={denominator:.6e}, tolerance={tolerance:.6e}"
            )
    except (RuntimeError, np.linalg.LinAlgError):
        return _solve_augmented_one_parameter_bordered_arc(
            jacobian,
            parameter_column_vector,
            arc_vector,
            residual,
            arc_residual,
        )

    parameter_delta = (float(arc_residual) - float(arc_q @ residual_solve)) / denominator
    coefficient_delta = residual_solve + parameter_solve * parameter_delta
    tangent_parameter = 1.0 / denominator
    tangent_coefficients = parameter_solve * tangent_parameter
    return _BorderedArcSolve(
        delta=np.concatenate((coefficient_delta, np.array([parameter_delta], dtype=np.float64))),
        tangent_candidate=np.concatenate((tangent_coefficients, np.array([tangent_parameter], dtype=np.float64))),
    )


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


def _solve_augmented_one_parameter_bordered_arc(
    jacobian: sparse.spmatrix,
    parameter_column: NDArray[np.float64],
    arc_row: NDArray[np.float64],
    residual_vector: NDArray[np.float64],
    arc_residual: float,
) -> _BorderedArcSolve:
    size = jacobian.shape[0]
    augmented = _augmented_arc_matrix(jacobian, parameter_column, arc_row)
    lu = splu(augmented)
    delta_rhs = np.concatenate((np.asarray(residual_vector, dtype=np.float64).reshape(size), np.array([arc_residual])))
    tangent_rhs = np.concatenate((np.zeros(size, dtype=np.float64), np.array([1.0])))
    return _BorderedArcSolve(
        delta=lu.solve(delta_rhs),
        tangent_candidate=lu.solve(tangent_rhs),
    )
