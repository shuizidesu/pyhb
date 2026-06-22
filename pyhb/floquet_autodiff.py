"""Autodiff nonlinear Jacobians for Floquet stability postprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.func import jacrev

from .continuation_autodiff import _resolve_torch_device, _to_numpy
from .floquet import (
    FloquetConfig,
    FloquetResult,
    compute_floquet_from_sampled_jacobians,
    prepare_solution_samples,
    sampled_jacobians_from_local_arrays,
)
from .models import AutodiffSecondOrderTimeModel, JacobianVariable
from .models import AutodiffFreeFrequencySecondOrderTimeModel


def compute_floquet_autodiff(
    model: AutodiffSecondOrderTimeModel,
    coefficients: NDArray[np.float64],
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig | None = None,
    torch_device: str | None = None,
) -> FloquetResult:
    """Compute Floquet multipliers using Torch autodiff nonlinear Jacobians."""

    active_config = config or FloquetConfig()
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
    return compute_floquet_from_sampled_jacobians(
        model,
        samples,
        sampled_jacobians,
        active_config,
    )


def compute_free_frequency_floquet_autodiff(
    model: AutodiffFreeFrequencySecondOrderTimeModel,
    coefficients: NDArray[np.float64],
    omega: float,
    parameter: float,
    harmonics: Sequence[float],
    frequency_resolution: float,
    config: FloquetConfig | None = None,
    torch_device: str | None = None,
) -> FloquetResult:
    """Compute free-frequency Floquet multipliers using Torch autodiff Jacobians."""

    active_config = config or FloquetConfig()
    device = _resolve_torch_device(torch_device)
    variables = _validate_autodiff_variables(model.autodiff_variables)
    samples = prepare_solution_samples(
        model,
        coefficients,
        omega,
        harmonics,
        frequency_resolution,
        active_config,
    )
    jacobian_by_variable = _free_frequency_autodiff_jacobians(
        model,
        samples.t,
        samples.x[:, list(model.residual_coordinate_dofs)],
        samples.dx[:, list(model.residual_coordinate_dofs)],
        samples.ddx[:, list(model.residual_coordinate_dofs)],
        float(omega),
        float(parameter),
        variables,
        device,
    )
    sampled_jacobians = sampled_jacobians_from_local_arrays(
        model.residual_force_dofs,
        model.residual_coordinate_dofs,
        jacobian_by_variable,
        samples.t.size,
        model.n_dof,
    )
    return compute_floquet_from_sampled_jacobians(
        model,
        samples,
        sampled_jacobians,
        active_config,
    )


def _autodiff_jacobians(
    model: AutodiffSecondOrderTimeModel,
    t: NDArray[np.float64],
    local_x: NDArray[np.float64],
    local_dx: NDArray[np.float64],
    local_ddx: NDArray[np.float64],
    parameter: float,
    variables: tuple[JacobianVariable, ...],
    device: torch.device,
) -> dict[JacobianVariable, NDArray[np.float64]]:
    t_tensor = _as_torch(t, device)
    x_tensor = _as_torch(local_x, device)
    dx_tensor = _as_torch(local_dx, device)
    ddx_tensor = _as_torch(local_ddx, device)
    parameter_tensor = torch.as_tensor(float(parameter), dtype=torch.float64, device=device)

    jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]] = {}
    for variable in variables:
        jacobian = jacrev(
            lambda active_variable: _force_sum(
                model,
                variable,
                active_variable,
                t_tensor,
                x_tensor,
                dx_tensor,
                ddx_tensor,
                parameter_tensor,
                device,
            )
        )(_select_variable(variable, x_tensor, dx_tensor, ddx_tensor))
        jacobian_by_variable[variable] = _to_numpy(jacobian.permute(1, 0, 2).contiguous())
    return jacobian_by_variable


def _free_frequency_autodiff_jacobians(
    model: AutodiffFreeFrequencySecondOrderTimeModel,
    t: NDArray[np.float64],
    local_x: NDArray[np.float64],
    local_dx: NDArray[np.float64],
    local_ddx: NDArray[np.float64],
    omega: float,
    parameter: float,
    variables: tuple[JacobianVariable, ...],
    device: torch.device,
) -> dict[JacobianVariable, NDArray[np.float64]]:
    t_tensor = _as_torch(t, device)
    x_tensor = _as_torch(local_x, device)
    dx_tensor = _as_torch(local_dx, device)
    ddx_tensor = _as_torch(local_ddx, device)
    omega_tensor = torch.as_tensor(float(omega), dtype=torch.float64, device=device)
    parameter_tensor = torch.as_tensor(float(parameter), dtype=torch.float64, device=device)

    jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]] = {}
    for variable in variables:
        jacobian = jacrev(
            lambda active_variable: _free_frequency_force_sum(
                model,
                variable,
                active_variable,
                t_tensor,
                x_tensor,
                dx_tensor,
                ddx_tensor,
                omega_tensor,
                parameter_tensor,
                device,
            )
        )(_select_variable(variable, x_tensor, dx_tensor, ddx_tensor))
        jacobian_by_variable[variable] = _to_numpy(jacobian.permute(1, 0, 2).contiguous())
    return jacobian_by_variable


def _force_sum(
    model: AutodiffSecondOrderTimeModel,
    variable: JacobianVariable,
    active_variable: torch.Tensor,
    t: torch.Tensor,
    local_x: torch.Tensor,
    local_dx: torch.Tensor,
    local_ddx: torch.Tensor,
    parameter: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    active_x = active_variable if variable == "x" else local_x
    active_dx = active_variable if variable == "dx" else local_dx
    active_ddx = active_variable if variable == "ddx" else local_ddx
    force = model.local_nonlinear_force_torch(t, active_x, active_dx, active_ddx, parameter)
    if not isinstance(force, torch.Tensor):
        raise TypeError("local_nonlinear_force_torch must return a torch.Tensor")
    return force.to(dtype=torch.float64, device=device).sum(dim=0)


def _free_frequency_force_sum(
    model: AutodiffFreeFrequencySecondOrderTimeModel,
    variable: JacobianVariable,
    active_variable: torch.Tensor,
    t: torch.Tensor,
    local_x: torch.Tensor,
    local_dx: torch.Tensor,
    local_ddx: torch.Tensor,
    omega: torch.Tensor,
    parameter: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    active_x = active_variable if variable == "x" else local_x
    active_dx = active_variable if variable == "dx" else local_dx
    active_ddx = active_variable if variable == "ddx" else local_ddx
    force = model.local_residual_force_torch(t, active_x, active_dx, active_ddx, omega, parameter)
    if not isinstance(force, torch.Tensor):
        raise TypeError("local_residual_force_torch must return a torch.Tensor")
    return force.to(dtype=torch.float64, device=device).sum(dim=0)


def _select_variable(
    variable: JacobianVariable,
    local_x: torch.Tensor,
    local_dx: torch.Tensor,
    local_ddx: torch.Tensor,
) -> torch.Tensor:
    if variable == "x":
        return local_x
    if variable == "dx":
        return local_dx
    if variable == "ddx":
        return local_ddx
    raise ValueError(f"unsupported autodiff variable {variable!r}")


def _validate_autodiff_variables(variables: tuple[JacobianVariable, ...]) -> tuple[JacobianVariable, ...]:
    normalized = tuple(variables)
    allowed = {"x", "dx", "ddx"}
    if len(set(normalized)) != len(normalized):
        raise ValueError("autodiff_variables must not contain duplicates")
    unsupported = set(normalized) - allowed
    if unsupported:
        raise ValueError(f"unsupported autodiff variable(s): {sorted(unsupported)}")
    return normalized


def _as_torch(values: NDArray[np.float64], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float64, device=device)
