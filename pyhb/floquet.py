"""Floquet stability postprocessing for HB periodic responses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from .continuation_core import _StructuredHBModel
from .harmonics import generate_hb_items
from .models import (
    FreeFrequencySecondOrderTimeModel,
    JacobianVariable,
    LinearOperatorTerm,
    LocalJacobianMatrices,
    SecondOrderTimeModel,
)


@dataclass(frozen=True)
class FloquetConfig:
    """Options for Hsu-style Floquet multiplier computation."""

    hsu_samples: int = 512
    method: str = "trapezoid"
    stability_tolerance: float = 1e-6
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FloquetResult:
    """Stability result for one HB solution point."""

    parameter: float
    multipliers: NDArray[np.complex128]
    spectral_radius: float
    stable: bool
    method: str
    period: float
    hsu_samples: int


def compute_floquet(
    model: SecondOrderTimeModel,
    coefficients: NDArray[np.float64],
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig | None = None,
) -> FloquetResult:
    """Compute Floquet multipliers for one analytical-model HB solution."""

    active_config = config or FloquetConfig()
    _validate_config(active_config)
    method = _resolve_method(active_config.method)
    samples = _prepare_solution_samples(
        model,
        coefficients,
        parameter,
        harmonics,
        frequency_resolution,
        active_config,
    )
    coordinate_dofs = tuple(model.nonlinear_coordinate_dofs)
    jacobian = model.local_nonlinear_jacobian(
        samples.t,
        samples.x[:, list(coordinate_dofs)],
        samples.dx[:, list(coordinate_dofs)],
        samples.ddx[:, list(coordinate_dofs)],
        float(parameter),
    )
    if method == "exponential":
        jacobians = dense_jacobians_from_local_matrices(
            model.nonlinear_force_dofs,
            coordinate_dofs,
            jacobian,
            samples.t.size,
            model.n_dof,
        )
        multipliers = _exponential_multipliers(model, samples, jacobians)
    else:
        jacobians = sampled_jacobians_from_local_matrices(
            model.nonlinear_force_dofs,
            coordinate_dofs,
            jacobian,
            samples.t.size,
            model.n_dof,
        )
        multipliers = _trapezoid_multipliers(model, samples, jacobians)
    return _build_floquet_result(samples, multipliers, active_config, method)


def compute_free_frequency_floquet(
    model: FreeFrequencySecondOrderTimeModel,
    coefficients: NDArray[np.float64],
    omega: float,
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig | None = None,
) -> FloquetResult:
    """Compute Floquet multipliers for one free-frequency HB solution."""

    active_config = config or FloquetConfig()
    _validate_config(active_config)
    method = _resolve_method(active_config.method)
    samples = _prepare_solution_samples(
        model,
        coefficients,
        omega,
        harmonics,
        frequency_resolution,
        active_config,
    )
    coordinate_dofs = tuple(model.residual_coordinate_dofs)
    jacobian = model.local_residual_jacobian(
        samples.t,
        samples.x[:, list(coordinate_dofs)],
        samples.dx[:, list(coordinate_dofs)],
        samples.ddx[:, list(coordinate_dofs)],
        float(omega),
        float(parameter),
    )
    if method == "exponential":
        jacobians = dense_jacobians_from_local_matrices(
            model.residual_force_dofs,
            coordinate_dofs,
            jacobian,
            samples.t.size,
            model.n_dof,
        )
        multipliers = _exponential_multipliers(model, samples, jacobians)
    else:
        jacobians = sampled_jacobians_from_local_matrices(
            model.residual_force_dofs,
            coordinate_dofs,
            jacobian,
            samples.t.size,
            model.n_dof,
        )
        multipliers = _trapezoid_multipliers(model, samples, jacobians)
    return _build_floquet_result(samples, multipliers, active_config, method)


@dataclass(frozen=True)
class _SolutionSamples:
    parameter: float
    period: float
    dt: float
    t: NDArray[np.float64]
    x: NDArray[np.float64]
    dx: NDArray[np.float64]
    ddx: NDArray[np.float64]


@dataclass(frozen=True)
class _DenseJacobianSamples:
    force_dofs: tuple[int, ...]
    coordinate_dofs: tuple[int, ...]
    values: dict[JacobianVariable, NDArray[np.float64]]


def prepare_solution_samples(
    model: _StructuredHBModel,
    coefficients: NDArray[np.float64],
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig,
) -> _SolutionSamples:
    """Reconstruct the sampled HB state used by Floquet analysis."""

    return _prepare_solution_samples(
        model,
        coefficients,
        parameter,
        harmonics,
        frequency_resolution,
        config,
    )


def compute_floquet_from_sparse_jacobians(
    model: _StructuredHBModel,
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    config: FloquetConfig,
) -> FloquetResult:
    """Compute trapezoid multipliers from sampled sparse Jacobians."""

    _validate_config(config)
    method = _resolve_method(config.method)
    if method != "trapezoid":
        raise ValueError("sparse sampled Jacobians are only used by the trapezoid Floquet method")
    multipliers = _trapezoid_multipliers(model, samples, jacobians)
    return _build_floquet_result(samples, multipliers, config, method)


def compute_floquet_from_dense_jacobians(
    model: _StructuredHBModel,
    samples: _SolutionSamples,
    jacobians: _DenseJacobianSamples,
    config: FloquetConfig,
) -> FloquetResult:
    """Compute exponential multipliers from compact dense Jacobians."""

    _validate_config(config)
    method = _resolve_method(config.method)
    if method != "exponential":
        raise ValueError("dense sampled Jacobians are only used by the exponential Floquet method")
    multipliers = _exponential_multipliers(model, samples, jacobians)
    return _build_floquet_result(samples, multipliers, config, method)


def _build_floquet_result(
    samples: _SolutionSamples,
    multipliers: NDArray[np.complex128],
    config: FloquetConfig,
    method: str,
) -> FloquetResult:
    spectral_radius = float(np.max(np.abs(multipliers))) if multipliers.size else 0.0
    stable = bool(spectral_radius <= 1.0 + float(config.stability_tolerance))
    _emit_progress(
        config,
        f"Floquet done, omega={samples.parameter:.10g}, rho={spectral_radius:.6e}, {_stability_label(stable)}",
    )
    return FloquetResult(
        parameter=float(samples.parameter),
        multipliers=np.asarray(multipliers, dtype=np.complex128),
        spectral_radius=spectral_radius,
        stable=stable,
        method=method,
        period=float(samples.period),
        hsu_samples=int(config.hsu_samples),
    )


def _prepare_solution_samples(
    model: _StructuredHBModel,
    coefficients: NDArray[np.float64],
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig,
) -> _SolutionSamples:
    _validate_config(config)
    coefficient_matrix = np.asarray(coefficients, dtype=np.float64)
    expected_shape = (2 * len(tuple(harmonics)) + 1, model.n_dof)
    if coefficient_matrix.shape != expected_shape:
        raise ValueError(f"coefficients must have shape {expected_shape}, got {coefficient_matrix.shape}")
    period = 2.0 * np.pi / float(frequency_resolution)
    dt = period / int(config.hsu_samples)
    t = (np.arange(int(config.hsu_samples), dtype=np.float64) + 0.5) * dt
    hb_item, hb_item_dt, hb_item_ddt = generate_hb_items(t, harmonics)
    x = hb_item @ coefficient_matrix
    dx = hb_item_dt @ coefficient_matrix
    ddx = hb_item_ddt @ coefficient_matrix
    return _SolutionSamples(
        parameter=float(parameter),
        period=float(period),
        dt=float(dt),
        t=t,
        x=x,
        dx=dx,
        ddx=ddx,
    )


def dense_jacobians_from_local_matrices(
    force_dofs: Sequence[int],
    coordinate_dofs: Sequence[int],
    jacobians: LocalJacobianMatrices,
    sample_count: int,
    n_dof: int,
) -> _DenseJacobianSamples:
    """Keep local Jacobian matrices in compact dense-coordinate form."""

    force_indices = tuple(int(value) for value in force_dofs)
    coordinate_indices = tuple(int(value) for value in coordinate_dofs)
    _validate_dense_jacobian_dofs(force_indices, coordinate_indices, n_dof)
    values = _empty_dense_jacobian_values(sample_count, len(force_indices), len(coordinate_indices))
    expected_shape = (sample_count, len(force_indices), len(coordinate_indices))
    for variable, variable_values in _local_jacobian_items(jacobians):
        array = np.asarray(variable_values, dtype=np.float64)
        if array.shape != expected_shape:
            raise ValueError(f"dN/d{variable} must have shape {expected_shape}, got {array.shape}")
        values[variable] = array
    return _DenseJacobianSamples(force_indices, coordinate_indices, values)


def _empty_dense_jacobian_values(
    sample_count: int,
    force_count: int,
    coordinate_count: int,
) -> dict[JacobianVariable, NDArray[np.float64]]:
    shape = (sample_count, force_count, coordinate_count)
    return {variable: np.zeros(shape, dtype=np.float64) for variable in ("x", "dx", "ddx")}


def _validate_dense_jacobian_dofs(
    force_dofs: tuple[int, ...],
    coordinate_dofs: tuple[int, ...],
    n_dof: int,
) -> None:
    for name, dofs in (("force_dofs", force_dofs), ("coordinate_dofs", coordinate_dofs)):
        if len(set(dofs)) != len(dofs):
            raise ValueError(f"{name} must not contain duplicates")
        invalid = [dof for dof in dofs if not 0 <= dof < n_dof]
        if invalid:
            raise ValueError(f"{name} contains out-of-range entries: {invalid}")


def sampled_jacobians_from_local_matrices(
    force_dofs: Sequence[int],
    coordinate_dofs: Sequence[int],
    jacobians: LocalJacobianMatrices,
    sample_count: int,
    n_dof: int,
) -> dict[JacobianVariable, tuple[sparse.csc_matrix, ...]]:
    """Scatter local Jacobian matrices to sampled global sparse matrices."""

    if not isinstance(jacobians, LocalJacobianMatrices):
        raise TypeError("local Jacobian method must return LocalJacobianMatrices")
    result: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]] = {}
    force_indices = tuple(int(value) for value in force_dofs)
    coordinate_indices = tuple(int(value) for value in coordinate_dofs)
    _validate_dense_jacobian_dofs(force_indices, coordinate_indices, n_dof)
    for variable in ("x", "dx", "ddx"):
        values = getattr(jacobians, variable)
        if values is None:
            result[variable] = tuple(sparse.csc_matrix((n_dof, n_dof), dtype=np.float64) for _ in range(sample_count))
            continue
        array = np.asarray(values, dtype=np.float64)
        expected_shape = (sample_count, len(force_indices), len(coordinate_indices))
        if array.shape != expected_shape:
            raise ValueError(f"dN/d{variable} must have shape {expected_shape}, got {array.shape}")
        matrices = []
        for sample_index in range(sample_count):
            sample_values = array[sample_index]
            rows, cols = np.nonzero(sample_values)
            matrices.append(
                sparse.coo_matrix(
                    (
                        sample_values[rows, cols],
                        (
                            np.asarray(force_indices, dtype=np.int64)[rows],
                            np.asarray(coordinate_indices, dtype=np.int64)[cols],
                        ),
                    ),
                    shape=(n_dof, n_dof),
                    dtype=np.float64,
                ).tocsc()
            )
        result[variable] = tuple(matrices)
    return result


def _local_jacobian_items(
    jacobians: LocalJacobianMatrices,
) -> tuple[tuple[JacobianVariable, NDArray[np.float64]], ...]:
    if not isinstance(jacobians, LocalJacobianMatrices):
        raise TypeError("local Jacobian method must return LocalJacobianMatrices")
    return tuple(
        (variable, values)
        for variable, values in (("x", jacobians.x), ("dx", jacobians.dx), ("ddx", jacobians.ddx))
        if values is not None
    )


def _exponential_multipliers(
    model: _StructuredHBModel,
    samples: _SolutionSamples,
    jacobians: _DenseJacobianSamples,
) -> NDArray[np.complex128]:
    mass_base, damping_base, stiffness_base = _combine_dense_time_operator_matrices(
        model.linear_operator_terms(),
        float(samples.parameter),
        model.n_dof,
    )
    state_dim = 2 * model.n_dof
    monodromy = np.eye(state_dim, dtype=np.float64)
    for sample_index in range(samples.t.size):
        mass = _add_dense_jacobian(mass_base, jacobians, "ddx", sample_index)
        damping = _add_dense_jacobian(damping_base, jacobians, "dx", sample_index)
        stiffness = _add_dense_jacobian(stiffness_base, jacobians, "x", sample_index)
        state_matrix = _dense_state_matrix(mass, damping, stiffness)
        monodromy = linalg.expm(state_matrix * samples.dt) @ monodromy
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _trapezoid_multipliers(
    model: _StructuredHBModel,
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
) -> NDArray[np.complex128]:
    mass_base, damping_base, stiffness_base = _combine_sparse_time_operator_matrices(
        model.linear_operator_terms(),
        float(samples.parameter),
        model.n_dof,
    )
    dt = float(samples.dt)
    identity = sparse.identity(model.n_dof, format="csc", dtype=np.float64)
    monodromy = np.eye(2 * model.n_dof, dtype=np.float64)
    for sample_index in range(samples.t.size):
        mass = mass_base + jacobians["ddx"][sample_index]
        damping = damping_base + jacobians["dx"][sample_index]
        stiffness = stiffness_base + jacobians["x"][sample_index]
        left = sparse.bmat(
            (
                (identity, -0.5 * dt * identity),
                (0.5 * dt * stiffness, mass + 0.5 * dt * damping),
            ),
            format="csc",
        )
        right = sparse.bmat(
            (
                (identity, 0.5 * dt * identity),
                (-0.5 * dt * stiffness, mass - 0.5 * dt * damping),
            ),
            format="csc",
        )
        monodromy = splu(left).solve(right @ monodromy)
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _dense_state_matrix(
    mass: NDArray[np.float64],
    damping: NDArray[np.float64],
    stiffness: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_dof = mass.shape[0]
    solved = linalg.solve(mass, np.column_stack((stiffness, damping)), assume_a="gen")
    state_matrix = np.zeros((2 * n_dof, 2 * n_dof), dtype=np.float64)
    state_matrix[:n_dof, n_dof:] = np.eye(n_dof, dtype=np.float64)
    state_matrix[n_dof:, :n_dof] = -solved[:, :n_dof]
    state_matrix[n_dof:, n_dof:] = -solved[:, n_dof:]
    return state_matrix


def _add_dense_jacobian(
    base: NDArray[np.float64],
    jacobians: _DenseJacobianSamples,
    variable: JacobianVariable,
    sample_index: int,
) -> NDArray[np.float64]:
    matrix = base.copy()
    if jacobians.force_dofs and jacobians.coordinate_dofs:
        matrix[np.ix_(jacobians.force_dofs, jacobians.coordinate_dofs)] += jacobians.values[variable][sample_index]
    return matrix


def _resolve_method(method: str) -> str:
    if method not in {"trapezoid", "exponential"}:
        raise ValueError("FloquetConfig.method must be 'trapezoid' or 'exponential'")
    return method


def _combine_sparse_time_operator_matrices(
    terms: Sequence[LinearOperatorTerm],
    parameter: float,
    n_dof: int,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
    matrices = {
        "ddx": sparse.csc_matrix((n_dof, n_dof), dtype=np.float64),
        "dx": sparse.csc_matrix((n_dof, n_dof), dtype=np.float64),
        "x": sparse.csc_matrix((n_dof, n_dof), dtype=np.float64),
    }
    for term in terms:
        if term.basis_type not in matrices:
            raise ValueError(f"unsupported linear operator basis_type: {term.basis_type!r}")
        matrix = sparse.csc_matrix(term.matrix, dtype=np.float64)
        if matrix.shape != (n_dof, n_dof):
            raise ValueError(f"linear operator matrix must have shape {(n_dof, n_dof)}, got {matrix.shape}")
        matrices[term.basis_type] = matrices[term.basis_type] + float(parameter) ** float(term.omega_power) * matrix
    return matrices["ddx"].tocsc(), matrices["dx"].tocsc(), matrices["x"].tocsc()


def _combine_dense_time_operator_matrices(
    terms: Sequence[LinearOperatorTerm],
    parameter: float,
    n_dof: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrices = {
        "ddx": np.zeros((n_dof, n_dof), dtype=np.float64),
        "dx": np.zeros((n_dof, n_dof), dtype=np.float64),
        "x": np.zeros((n_dof, n_dof), dtype=np.float64),
    }
    for term in terms:
        if term.basis_type not in matrices:
            raise ValueError(f"unsupported linear operator basis_type: {term.basis_type!r}")
        matrix = term.matrix.toarray() if sparse.issparse(term.matrix) else np.asarray(term.matrix, dtype=np.float64)
        if matrix.shape != (n_dof, n_dof):
            raise ValueError(f"linear operator matrix must have shape {(n_dof, n_dof)}, got {matrix.shape}")
        matrices[term.basis_type] += float(parameter) ** float(term.omega_power) * matrix
    return matrices["ddx"], matrices["dx"], matrices["x"]


def _emit_progress(config: FloquetConfig, message: str) -> None:
    if config.progress_callback is not None:
        config.progress_callback(message)


def _stability_label(stable: bool) -> str:
    return "stable" if stable else "unstable"


def _validate_config(config: FloquetConfig) -> None:
    if config.hsu_samples <= 0:
        raise ValueError("hsu_samples must be positive")
    if config.stability_tolerance < 0.0:
        raise ValueError("stability_tolerance must be non-negative")
