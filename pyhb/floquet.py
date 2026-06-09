"""Floquet stability postprocessing for HB periodic responses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, eigs, expm_multiply, splu

from .harmonics import generate_hb_items
from .models import JacobianVariable, SecondOrderTimeModel, _combine_time_operator_matrices


@dataclass(frozen=True)
class FloquetConfig:
    """Options for Hsu-style Floquet multiplier computation."""

    hsu_samples: int = 512
    method: str = "auto"
    explicit_state_limit: int = 800
    n_multipliers: int = 8
    stability_tolerance: float = 1e-6
    eigs_tolerance: float = 1e-8


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
    samples = _prepare_solution_samples(
        model,
        coefficients,
        parameter,
        harmonics,
        frequency_resolution,
        active_config,
    )
    jacobians = _sampled_jacobians_from_terms(
        model.nonlinear_jacobian_terms(
            samples.t,
            samples.x,
            samples.dx,
            samples.ddx,
            float(parameter),
        ),
        samples.t.size,
        model.n_dof,
    )
    return compute_floquet_from_sampled_jacobians(
        model,
        samples,
        jacobians,
        active_config,
    )


@dataclass(frozen=True)
class _SolutionSamples:
    parameter: float
    period: float
    dt: float
    t: NDArray[np.float64]
    x: NDArray[np.float64]
    dx: NDArray[np.float64]
    ddx: NDArray[np.float64]
    mass: NDArray[np.float64]
    damping: NDArray[np.float64]
    stiffness: NDArray[np.float64]


def prepare_solution_samples(
    model: SecondOrderTimeModel,
    coefficients: NDArray[np.float64],
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig,
) -> _SolutionSamples:
    """Reconstruct states and structured linear matrices for Floquet analysis."""

    return _prepare_solution_samples(
        model,
        coefficients,
        parameter,
        harmonics,
        frequency_resolution,
        config,
    )


def compute_floquet_from_sampled_jacobians(
    model: SecondOrderTimeModel,
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    config: FloquetConfig,
) -> FloquetResult:
    """Compute multipliers from precomputed time-domain nonlinear Jacobians."""

    _validate_config(config)
    state_dim = 2 * model.n_dof
    method = _resolve_method(config.method, state_dim, config.explicit_state_limit)
    if method == "explicit":
        multipliers = _explicit_multipliers(samples, jacobians)
    else:
        multipliers = _dominant_multipliers(samples, jacobians, config)
    spectral_radius = float(np.max(np.abs(multipliers))) if multipliers.size else 0.0
    return FloquetResult(
        parameter=float(samples.parameter),
        multipliers=np.asarray(multipliers, dtype=np.complex128),
        spectral_radius=spectral_radius,
        stable=bool(spectral_radius <= 1.0 + float(config.stability_tolerance)),
        method=method,
        period=float(samples.period),
        hsu_samples=int(config.hsu_samples),
    )


def _prepare_solution_samples(
    model: SecondOrderTimeModel,
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
    mass, damping, stiffness = _combine_time_operator_matrices(
        model.linear_operator_terms(),
        float(parameter),
        model.n_dof,
    )
    return _SolutionSamples(
        parameter=float(parameter),
        period=float(period),
        dt=float(dt),
        t=t,
        x=x,
        dx=dx,
        ddx=ddx,
        mass=mass,
        damping=damping,
        stiffness=stiffness,
    )


def _sampled_jacobians_from_terms(
    terms: Sequence[object],
    sample_count: int,
    n_dof: int,
) -> dict[JacobianVariable, tuple[sparse.csc_matrix, ...]]:
    rows: dict[JacobianVariable, list[list[int]]] = {variable: [[] for _ in range(sample_count)] for variable in ("x", "dx", "ddx")}
    cols: dict[JacobianVariable, list[list[int]]] = {variable: [[] for _ in range(sample_count)] for variable in ("x", "dx", "ddx")}
    data: dict[JacobianVariable, list[list[float]]] = {variable: [[] for _ in range(sample_count)] for variable in ("x", "dx", "ddx")}
    for term in terms:
        variable = term.variable
        if variable not in rows:
            raise ValueError(f"unsupported nonlinear Jacobian variable {variable!r}")
        values = np.asarray(term.values, dtype=np.float64).reshape(-1)
        if values.shape[0] != sample_count:
            raise ValueError(
                "nonlinear Jacobian term values must have one value per Hsu sample; "
                f"got {values.shape[0]}, expected {sample_count}"
            )
        for index, value in enumerate(values):
            if value == 0.0:
                continue
            rows[variable][index].append(int(term.force_dof))
            cols[variable][index].append(int(term.coordinate_dof))
            data[variable][index].append(float(value))
    return {
        variable: tuple(
            sparse.coo_matrix(
                (data[variable][index], (rows[variable][index], cols[variable][index])),
                shape=(n_dof, n_dof),
                dtype=np.float64,
            ).tocsc()
            for index in range(sample_count)
        )
        for variable in ("x", "dx", "ddx")
    }


def sampled_jacobians_from_local_arrays(
    force_dofs: Sequence[int],
    coordinate_dofs: Sequence[int],
    jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]],
    sample_count: int,
    n_dof: int,
) -> dict[JacobianVariable, tuple[sparse.csc_matrix, ...]]:
    """Scatter autodiff local Jacobian arrays to sampled global sparse matrices."""

    result: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]] = {}
    force_indices = tuple(int(value) for value in force_dofs)
    coordinate_indices = tuple(int(value) for value in coordinate_dofs)
    for variable in ("x", "dx", "ddx"):
        values = jacobian_by_variable.get(variable)
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


def _explicit_multipliers(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
) -> NDArray[np.complex128]:
    state_dim = 2 * samples.mass.shape[0]
    monodromy = np.eye(state_dim, dtype=np.float64)
    for sample_index in range(samples.t.size):
        state_matrix = _dense_state_matrix(samples, jacobians, sample_index)
        monodromy = linalg.expm(state_matrix * samples.dt) @ monodromy
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _dominant_multipliers(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    config: FloquetConfig,
) -> NDArray[np.complex128]:
    state_dim = 2 * samples.mass.shape[0]
    operators = tuple(
        _state_linear_operator(samples, jacobians, sample_index)
        for sample_index in range(samples.t.size)
    )

    def matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        propagated = np.asarray(vector, dtype=np.float64)
        for operator in operators:
            propagated = expm_multiply(_scaled_operator(operator, samples.dt), propagated, traceA=0.0)
        return np.asarray(propagated, dtype=np.float64)

    monodromy = LinearOperator(
        (state_dim, state_dim),
        matvec=matvec,
        dtype=np.float64,
    )
    k = min(int(config.n_multipliers), max(1, state_dim - 2))
    if k >= state_dim - 1:
        return _explicit_multipliers(samples, jacobians)
    values = eigs(monodromy, k=k, which="LM", tol=float(config.eigs_tolerance), return_eigenvectors=False)
    return np.asarray(values, dtype=np.complex128)


def _dense_state_matrix(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    sample_index: int,
) -> NDArray[np.float64]:
    n_dof = samples.mass.shape[0]
    mass = samples.mass + jacobians["ddx"][sample_index].toarray()
    damping = samples.damping + jacobians["dx"][sample_index].toarray()
    stiffness = samples.stiffness + jacobians["x"][sample_index].toarray()
    solved = linalg.solve(mass, np.column_stack((stiffness, damping)), assume_a="gen")
    state_matrix = np.zeros((2 * n_dof, 2 * n_dof), dtype=np.float64)
    state_matrix[:n_dof, n_dof:] = np.eye(n_dof, dtype=np.float64)
    state_matrix[n_dof:, :n_dof] = -solved[:, :n_dof]
    state_matrix[n_dof:, n_dof:] = -solved[:, n_dof:]
    return state_matrix


def _state_linear_operator(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    sample_index: int,
) -> LinearOperator:
    n_dof = samples.mass.shape[0]
    mass = _as_sparse(samples.mass) + jacobians["ddx"][sample_index]
    damping = _as_sparse(samples.damping) + jacobians["dx"][sample_index]
    stiffness = _as_sparse(samples.stiffness) + jacobians["x"][sample_index]
    mass_lu = splu(mass.tocsc())

    def matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        displacement = vector[:n_dof]
        velocity = vector[n_dof:]
        acceleration = mass_lu.solve(-(stiffness @ displacement + damping @ velocity))
        return np.concatenate((velocity, acceleration))

    def rmatvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        head = vector[:n_dof]
        tail = vector[n_dof:]
        solved_tail = mass_lu.solve(tail, trans="T")
        return np.concatenate((-(stiffness.T @ solved_tail), head - damping.T @ solved_tail))

    return LinearOperator(
        (2 * n_dof, 2 * n_dof),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )


def _scaled_operator(operator: LinearOperator, scale: float) -> LinearOperator:
    return LinearOperator(
        operator.shape,
        matvec=lambda vector: scale * operator.matvec(vector),
        rmatvec=lambda vector: scale * operator.rmatvec(vector),
        dtype=np.float64,
    )


def _as_sparse(matrix: NDArray[np.float64] | sparse.spmatrix) -> sparse.csc_matrix:
    return matrix.tocsc() if sparse.issparse(matrix) else sparse.csc_matrix(np.asarray(matrix, dtype=np.float64))


def _resolve_method(method: str, state_dim: int, explicit_state_limit: int) -> str:
    if method == "auto":
        return "explicit" if state_dim <= int(explicit_state_limit) else "dominant"
    if method not in {"explicit", "dominant"}:
        raise ValueError("FloquetConfig.method must be 'auto', 'explicit', or 'dominant'")
    return method


def _validate_config(config: FloquetConfig) -> None:
    if config.hsu_samples <= 0:
        raise ValueError("hsu_samples must be positive")
    if config.explicit_state_limit <= 0:
        raise ValueError("explicit_state_limit must be positive")
    if config.n_multipliers <= 0:
        raise ValueError("n_multipliers must be positive")
    if config.stability_tolerance < 0.0:
        raise ValueError("stability_tolerance must be non-negative")
    if config.eigs_tolerance <= 0.0:
        raise ValueError("eigs_tolerance must be positive")
