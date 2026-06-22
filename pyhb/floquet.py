"""Floquet stability postprocessing for HB periodic responses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from .harmonics import generate_hb_items
from .models import FreeFrequencySecondOrderTimeModel, JacobianVariable, SecondOrderTimeModel


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
    samples = _prepare_solution_samples(
        model,
        coefficients,
        omega,
        harmonics,
        frequency_resolution,
        active_config,
    )
    jacobians = _sampled_jacobians_from_terms(
        model.residual_jacobian_terms(
            samples.t,
            samples.x,
            samples.dx,
            samples.ddx,
            float(omega),
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
    mass: sparse.csc_matrix
    damping: sparse.csc_matrix
    stiffness: sparse.csc_matrix


@dataclass(frozen=True)
class _HsuStep:
    left_lu: object
    right: sparse.csc_matrix


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
    method = _resolve_method(config.method)
    if method == "trapezoid":
        multipliers = _trapezoid_multipliers(samples, jacobians)
    elif method == "exponential":
        multipliers = _exponential_multipliers(samples, jacobians)
    else:  # pragma: no cover - _resolve_method keeps this unreachable.
        raise ValueError(f"unsupported Floquet method {method!r}")
    spectral_radius = float(np.max(np.abs(multipliers))) if multipliers.size else 0.0
    stable = bool(spectral_radius <= 1.0 + float(config.stability_tolerance))
    _emit_progress(
        config,
        "Floquet done, "
        f"omega={samples.parameter:.10g}, rho={spectral_radius:.6e}, {_stability_label(stable)}",
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
    mass, damping, stiffness = _combine_sparse_time_operator_matrices(
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


def _exponential_multipliers(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
) -> NDArray[np.complex128]:
    state_dim = 2 * samples.mass.shape[0]
    monodromy = np.eye(state_dim, dtype=np.float64)
    for sample_index in range(samples.t.size):
        state_matrix = _dense_state_matrix(samples, jacobians, sample_index)
        monodromy = linalg.expm(state_matrix * samples.dt) @ monodromy
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _trapezoid_multipliers(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
) -> NDArray[np.complex128]:
    steps = _hsu_steps(samples, jacobians)
    return _hsu_dense_multipliers(steps, samples.mass.shape[0])


def _hsu_steps(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
) -> tuple[_HsuStep, ...]:
    mass_base = samples.mass
    damping_base = samples.damping
    stiffness_base = samples.stiffness
    dt = float(samples.dt)
    identity = sparse.identity(samples.mass.shape[0], format="csc", dtype=np.float64)
    steps = []
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
        steps.append(_HsuStep(splu(left), right))
    return tuple(steps)


def _hsu_dense_multipliers(
    steps: tuple[_HsuStep, ...],
    n_dof: int,
) -> NDArray[np.complex128]:
    state_dim = 2 * n_dof
    monodromy = np.eye(state_dim, dtype=np.float64)
    for step in steps:
        monodromy = step.left_lu.solve(step.right @ monodromy)
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _dense_state_matrix(
    samples: _SolutionSamples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    sample_index: int,
) -> NDArray[np.float64]:
    n_dof = samples.mass.shape[0]
    mass = (samples.mass + jacobians["ddx"][sample_index]).toarray()
    damping = (samples.damping + jacobians["dx"][sample_index]).toarray()
    stiffness = (samples.stiffness + jacobians["x"][sample_index]).toarray()
    solved = linalg.solve(mass, np.column_stack((stiffness, damping)), assume_a="gen")
    state_matrix = np.zeros((2 * n_dof, 2 * n_dof), dtype=np.float64)
    state_matrix[:n_dof, n_dof:] = np.eye(n_dof, dtype=np.float64)
    state_matrix[n_dof:, :n_dof] = -solved[:, :n_dof]
    state_matrix[n_dof:, n_dof:] = -solved[:, n_dof:]
    return state_matrix


def _resolve_method(method: str) -> str:
    if method not in {"trapezoid", "exponential"}:
        raise ValueError("FloquetConfig.method must be 'trapezoid' or 'exponential'")
    return method


def _combine_sparse_time_operator_matrices(
    terms: Sequence[object],
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
