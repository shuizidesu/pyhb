"""Autodiff nonlinear derivatives for the full arc-length continuation solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.func import jacfwd, jacrev

from .autodiff_utils import (
    _as_torch,
    _resolve_torch_device,
    _select_variable,
    _to_numpy,
    _validate_autodiff_variables,
)
from .continuation import ContinuationConfig, ContinuationSolver, _NonlinearEvaluation
from .continuation_core import (
    _local_samples_to_global_coefficients,
    _validated_dofs,
    assemble_hb_jacobian_from_local_matrices,
)
from .models import AutodiffSecondOrderTimeModel, JacobianVariable, LocalJacobianMatrices


@dataclass(frozen=True)
class ContinuationAutodiffConfig(ContinuationConfig):
    """Continuation config with an optional Torch device selector."""

    torch_device: str | None = None


@dataclass(frozen=True)
class _AutodiffLocalEvaluation:
    force_samples: NDArray[np.float64]
    jacobian: LocalJacobianMatrices
    parameter_derivative: NDArray[np.float64]


class ContinuationAutodiffSolver(ContinuationSolver):
    """Full continuation solver using Torch autodiff for local nonlinear derivatives."""

    def __init__(
        self,
        model: AutodiffSecondOrderTimeModel,
        config: ContinuationAutodiffConfig | None = None,
    ) -> None:
        if not isinstance(model, AutodiffSecondOrderTimeModel):
            raise TypeError("ContinuationAutodiffSolver requires an AutodiffSecondOrderTimeModel")
        super().__init__(model, config or ContinuationAutodiffConfig())
        self.model: AutodiffSecondOrderTimeModel
        self.config: ContinuationAutodiffConfig
        self._torch_device = _resolve_torch_device(self.config.torch_device)
        self._autodiff_variables = _validate_autodiff_variables(self.model.autodiff_variables)
        self._t_tensor = _as_torch(self.prepared.t, self._torch_device)

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
        local = self._autodiff_values(
            local_x,
            local_dx,
            local_ddx,
            parameter,
            force_count=len(force_dofs),
            coordinate_count=len(coordinate_dofs),
            include_parameter=include_parameter,
        )
        nonlinear_coefficients = _local_samples_to_global_coefficients(
            local.force_samples,
            force_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local nonlinear force",
        )
        nonlinear_jacobian = assemble_hb_jacobian_from_local_matrices(
            local.jacobian,
            force_dofs,
            coordinate_dofs,
            self.prepared.context,
            self.prepared.t.size,
            self.model.n_dof,
            "local nonlinear",
        )
        parameter_coefficients = None
        if include_parameter:
            parameter_coefficients = _local_samples_to_global_coefficients(
                local.parameter_derivative,
                force_dofs,
                self.prepared.context,
                self.prepared.t.size,
                self.model.n_dof,
                "local nonlinear parameter derivative",
            )
        return _NonlinearEvaluation(nonlinear_coefficients, nonlinear_jacobian, parameter_coefficients)

    def _autodiff_values(
        self,
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
        *,
        force_count: int,
        coordinate_count: int,
        include_parameter: bool,
    ) -> _AutodiffLocalEvaluation:
        expected_force_shape = (self.prepared.t.size, force_count)
        expected_coordinate_shape = (self.prepared.t.size, coordinate_count)

        t_tensor = self._t_tensor
        x_tensor = _as_torch(local_x, self._torch_device)
        dx_tensor = _as_torch(local_dx, self._torch_device)
        ddx_tensor = _as_torch(local_ddx, self._torch_device)
        parameter_tensor = torch.as_tensor(float(parameter), dtype=torch.float64, device=self._torch_device)

        if x_tensor.shape != expected_coordinate_shape:
            raise ValueError(f"local_x must have shape {expected_coordinate_shape}, got {tuple(x_tensor.shape)}")

        force_tensor = self._call_torch_force(t_tensor, x_tensor, dx_tensor, ddx_tensor, parameter_tensor)
        force_samples = _to_numpy(force_tensor)
        if force_samples.shape != expected_force_shape:
            raise ValueError(f"local nonlinear force must have shape {expected_force_shape}, got {force_samples.shape}")

        jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]] = {}
        for variable in self._autodiff_variables:
            jacobian_tensor = self._differentiate_variable(
                variable,
                t_tensor,
                x_tensor,
                dx_tensor,
                ddx_tensor,
                parameter_tensor,
            )
            jacobian = _to_numpy(jacobian_tensor)
            expected_jacobian_shape = (self.prepared.t.size, force_count, coordinate_count)
            if jacobian.shape != expected_jacobian_shape:
                raise ValueError(f"dN/d{variable} must have shape {expected_jacobian_shape}, got {jacobian.shape}")
            jacobian_by_variable[variable] = jacobian

        if include_parameter and self.model.autodiff_parameter_dependent:
            parameter_derivative = _to_numpy(
                self._differentiate_parameter(t_tensor, x_tensor, dx_tensor, ddx_tensor, parameter_tensor)
            )
        else:
            parameter_derivative = np.zeros(expected_force_shape, dtype=np.float64)
        if parameter_derivative.shape != expected_force_shape:
            raise ValueError(
                "local nonlinear parameter derivative must have shape "
                f"{expected_force_shape}, got {parameter_derivative.shape}"
            )
        jacobian = LocalJacobianMatrices(
            x=jacobian_by_variable.get("x"),
            dx=jacobian_by_variable.get("dx"),
            ddx=jacobian_by_variable.get("ddx"),
        )
        return _AutodiffLocalEvaluation(force_samples, jacobian, parameter_derivative)

    def _call_torch_force(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        force = self.model.local_nonlinear_force_torch(t, local_x, local_dx, local_ddx, parameter)
        if not isinstance(force, torch.Tensor):
            raise TypeError("local_nonlinear_force_torch must return a torch.Tensor")
        return force.to(dtype=torch.float64, device=self._torch_device)

    def _differentiate_variable(
        self,
        variable: JacobianVariable,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        def summed_force(active_variable: torch.Tensor) -> torch.Tensor:
            active_x = active_variable if variable == "x" else local_x
            active_dx = active_variable if variable == "dx" else local_dx
            active_ddx = active_variable if variable == "ddx" else local_ddx
            return self._call_torch_force(t, active_x, active_dx, active_ddx, parameter).sum(dim=0)

        jacobian = jacrev(summed_force)(_select_variable(variable, local_x, local_dx, local_ddx))
        return jacobian.permute(1, 0, 2).contiguous()

    def _differentiate_parameter(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        def force_for_parameter(active_parameter: torch.Tensor) -> torch.Tensor:
            return self._call_torch_force(t, local_x, local_dx, local_ddx, active_parameter)

        return jacfwd(force_for_parameter)(parameter)
