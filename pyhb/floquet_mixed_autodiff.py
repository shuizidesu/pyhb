"""Mixed-order Floquet stability for autodiff HB responses.

This module handles systems that combine second-order mechanical coordinates
with first-order coordinates such as electrical voltages. It is intentionally
separate from :mod:`pyhb.floquet`, whose state-space conversion assumes an
invertible second-order mass block for every DOF.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from .autodiff_utils import _resolve_torch_device, _validate_autodiff_variables
from .floquet import (
    FloquetConfig,
    FloquetResult,
    _emit_progress,
    _resolve_method,
    _stability_label,
    _validate_config,
    prepare_solution_samples,
    sampled_jacobians_from_local_arrays,
)
from .floquet_autodiff import _autodiff_jacobians
from .models import AutodiffSecondOrderTimeModel, JacobianVariable


@dataclass(frozen=True)
class _MixedOrderLayout:
    second_order_dofs: tuple[int, ...]
    first_order_dofs: tuple[int, ...]
    n_dof: int

    @property
    def state_dim(self) -> int:
        return 2 * len(self.second_order_dofs) + len(self.first_order_dofs)


def compute_mixed_order_floquet_autodiff(
    model: AutodiffSecondOrderTimeModel,
    coefficients: NDArray[np.float64],
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    second_order_dofs: Sequence[int],
    first_order_dofs: Sequence[int],
    config: FloquetConfig | None = None,
    torch_device: str | None = None,
) -> FloquetResult:
    """Compute Floquet multipliers for mixed second-/first-order systems.

    The perturbation state is ordered as ``[dq, dq_dtau, du]``, where ``q`` are
    the second-order DOFs and ``u`` are the first-order DOFs. The sampled
    linearized equations are propagated in descriptor form, so zero mass rows
    in the original model are allowed.
    """

    active_config = config or FloquetConfig()
    _validate_config(active_config)
    method = _resolve_method(active_config.method)
    layout = _validate_layout(model.n_dof, second_order_dofs, first_order_dofs)
    device = _resolve_torch_device(torch_device)
    variables = _validate_autodiff_variables(model.autodiff_variables)
    samples = prepare_solution_samples(
        model,
        coefficients,
        parameter,
        harmonics,
        frequency_resolution,
        active_config,
    )
    jacobian_by_variable = _autodiff_jacobians(
        model,
        samples.t,
        samples.x[:, list(model.nonlinear_coordinate_dofs)],
        samples.dx[:, list(model.nonlinear_coordinate_dofs)],
        samples.ddx[:, list(model.nonlinear_coordinate_dofs)],
        float(parameter),
        variables,
        device,
    )
    sampled_jacobians = sampled_jacobians_from_local_arrays(
        model.nonlinear_force_dofs,
        model.nonlinear_coordinate_dofs,
        jacobian_by_variable,
        samples.t.size,
        model.n_dof,
    )
    if method == "trapezoid":
        multipliers = _mixed_trapezoid_multipliers(samples, sampled_jacobians, layout)
    elif method == "exponential":
        multipliers = _mixed_exponential_multipliers(samples, sampled_jacobians, layout)
    else:  # pragma: no cover - _resolve_method keeps this unreachable.
        raise ValueError(f"unsupported Floquet method {method!r}")
    spectral_radius = float(np.max(np.abs(multipliers))) if multipliers.size else 0.0
    stable = bool(spectral_radius <= 1.0 + float(active_config.stability_tolerance))
    _emit_progress(
        active_config,
        f"Floquet done, omega={float(parameter):.10g}, rho={spectral_radius:.6e}, {_stability_label(stable)}",
    )
    return FloquetResult(
        parameter=float(parameter),
        multipliers=np.asarray(multipliers, dtype=np.complex128),
        spectral_radius=spectral_radius,
        stable=stable,
        method=method,
        period=float(samples.period),
        hsu_samples=int(active_config.hsu_samples),
    )


def _mixed_trapezoid_multipliers(
    samples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    layout: _MixedOrderLayout,
) -> NDArray[np.complex128]:
    monodromy = np.eye(layout.state_dim, dtype=np.float64)
    dt = float(samples.dt)
    for sample_index in range(samples.t.size):
        descriptor, state_matrix = _mixed_descriptor_matrices(samples, jacobians, sample_index, layout)
        left = (descriptor - 0.5 * dt * state_matrix).tocsc()
        right = (descriptor + 0.5 * dt * state_matrix).tocsc()
        try:
            monodromy = splu(left).solve(right @ monodromy)
        except RuntimeError as exc:
            raise ValueError(f"mixed-order trapezoid Floquet step {sample_index} is singular") from exc
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _mixed_exponential_multipliers(
    samples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    layout: _MixedOrderLayout,
) -> NDArray[np.complex128]:
    monodromy = np.eye(layout.state_dim, dtype=np.float64)
    for sample_index in range(samples.t.size):
        descriptor, state_matrix = _mixed_descriptor_matrices(samples, jacobians, sample_index, layout)
        try:
            generator = linalg.solve(descriptor.toarray(), state_matrix.toarray(), assume_a="gen")
        except linalg.LinAlgError as exc:
            raise ValueError(f"mixed-order exponential Floquet descriptor {sample_index} is singular") from exc
        monodromy = linalg.expm(generator * float(samples.dt)) @ monodromy
    return np.asarray(linalg.eigvals(monodromy), dtype=np.complex128)


def _mixed_descriptor_matrices(
    samples,
    jacobians: dict[JacobianVariable, tuple[sparse.csc_matrix, ...]],
    sample_index: int,
    layout: _MixedOrderLayout,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    second = list(layout.second_order_dofs)
    first = list(layout.first_order_dofs)
    n_second = len(second)
    n_first = len(first)
    n_dof = layout.n_dof

    mass = samples.mass + jacobians["ddx"][sample_index]
    damping = samples.damping + jacobians["dx"][sample_index]
    stiffness = samples.stiffness + jacobians["x"][sample_index]
    _validate_no_first_order_acceleration(mass[:, first], sample_index)

    identity_second = sparse.identity(n_second, format="csc", dtype=np.float64)
    zero_ss = sparse.csc_matrix((n_second, n_second), dtype=np.float64)
    zero_sf = sparse.csc_matrix((n_second, n_first), dtype=np.float64)
    zero_ns = sparse.csc_matrix((n_dof, n_second), dtype=np.float64)

    descriptor_top = sparse.hstack((identity_second, zero_ss, zero_sf), format="csc")
    descriptor_bottom = sparse.hstack((zero_ns, mass[:, second], damping[:, first]), format="csc")
    descriptor = sparse.vstack((descriptor_top, descriptor_bottom), format="csc")

    state_top = sparse.hstack((zero_ss, identity_second, zero_sf), format="csc")
    state_bottom = sparse.hstack((-stiffness[:, second], -damping[:, second], -stiffness[:, first]), format="csc")
    state_matrix = sparse.vstack((state_top, state_bottom), format="csc")
    return descriptor, state_matrix


def _validate_layout(
    n_dof: int,
    second_order_dofs: Sequence[int],
    first_order_dofs: Sequence[int],
) -> _MixedOrderLayout:
    second = tuple(int(value) for value in second_order_dofs)
    first = tuple(int(value) for value in first_order_dofs)
    if not second:
        raise ValueError("second_order_dofs must not be empty")
    if not first:
        raise ValueError("first_order_dofs must not be empty")
    if len(set(second)) != len(second):
        raise ValueError("second_order_dofs must not contain duplicates")
    if len(set(first)) != len(first):
        raise ValueError("first_order_dofs must not contain duplicates")
    overlap = set(second) & set(first)
    if overlap:
        raise ValueError(f"second_order_dofs and first_order_dofs overlap: {sorted(overlap)}")
    all_dofs = set(second) | set(first)
    expected = set(range(int(n_dof)))
    if all_dofs != expected:
        missing = sorted(expected - all_dofs)
        extra = sorted(all_dofs - expected)
        raise ValueError(f"mixed-order DOFs must cover all model DOFs; missing={missing}, extra={extra}")
    return _MixedOrderLayout(second, first, int(n_dof))


def _validate_no_first_order_acceleration(block: sparse.spmatrix, sample_index: int) -> None:
    if block.nnz == 0:
        return
    max_abs = float(np.max(np.abs(block.data)))
    if max_abs > 1e-12:
        raise ValueError(
            "mixed-order Floquet does not support second-derivative terms on first_order_dofs; "
            f"sample={sample_index}, max_abs={max_abs:.6e}"
        )
