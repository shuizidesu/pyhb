"""Torch autodiff derivatives for free-frequency HB continuation."""

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
from .free_frequency import FreeFrequencyContinuationConfig, FreeFrequencyContinuationSolver
from .models import AutodiffFreeFrequencySecondOrderTimeModel, JacobianVariable, NonlinearJacobianTerm


@dataclass(frozen=True)
class FreeFrequencyContinuationAutodiffConfig(FreeFrequencyContinuationConfig):
    """Free-frequency continuation config with an optional Torch device selector."""

    torch_device: str | None = None


@dataclass
class _FreeAutodiffCache:
    key: tuple[int, int, int, float, float]
    force_samples: NDArray[np.float64]
    jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]]
    omega_derivative: NDArray[np.float64]
    parameter_derivative: NDArray[np.float64]


class FreeFrequencyContinuationAutodiffSolver(FreeFrequencyContinuationSolver):
    """Free-frequency continuation solver using Torch autodiff for local residual derivatives."""

    def __init__(
        self,
        model: AutodiffFreeFrequencySecondOrderTimeModel,
        config: FreeFrequencyContinuationAutodiffConfig | None = None,
    ) -> None:
        if not isinstance(model, AutodiffFreeFrequencySecondOrderTimeModel):
            raise TypeError(
                "FreeFrequencyContinuationAutodiffSolver requires an "
                "AutodiffFreeFrequencySecondOrderTimeModel"
            )
        self._free_autodiff_cache: _FreeAutodiffCache | None = None
        super().__init__(model, config or FreeFrequencyContinuationAutodiffConfig())
        self.model: AutodiffFreeFrequencySecondOrderTimeModel
        self.config: FreeFrequencyContinuationAutodiffConfig
        self._torch_device = _resolve_torch_device(self.config.torch_device)
        self._autodiff_variables = _validate_autodiff_variables(self.model.autodiff_variables)

    def _generalized_force_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        force = np.zeros((self.prepared.t.size, self.model.n_dof), dtype=np.float64)
        force[:, list(self.model.residual_force_dofs)] = self._autodiff_values(
            x,
            dx,
            ddx,
            omega,
            parameter,
        ).force_samples
        return force

    def _generalized_jacobian_terms(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[NonlinearJacobianTerm, ...]:
        cache = self._autodiff_values(x, dx, ddx, omega, parameter)
        force_dofs = tuple(self.model.residual_force_dofs)
        coordinate_dofs = tuple(self.model.residual_coordinate_dofs)
        terms: list[NonlinearJacobianTerm] = []
        for variable, values in cache.jacobian_by_variable.items():
            for force_index, force_dof in enumerate(force_dofs):
                for coordinate_index, coordinate_dof in enumerate(coordinate_dofs):
                    terms.append(
                        NonlinearJacobianTerm(
                            force_dof,
                            variable,
                            coordinate_dof,
                            values[:, force_index, coordinate_index],
                        )
                    )
        return tuple(terms)

    def _generalized_omega_derivative_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        derivative = np.zeros((self.prepared.t.size, self.model.n_dof), dtype=np.float64)
        derivative[:, list(self.model.residual_force_dofs)] = self._autodiff_values(
            x,
            dx,
            ddx,
            omega,
            parameter,
        ).omega_derivative
        return derivative

    def _generalized_parameter_derivative_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        derivative = np.zeros((self.prepared.t.size, self.model.n_dof), dtype=np.float64)
        derivative[:, list(self.model.residual_force_dofs)] = self._autodiff_values(
            x,
            dx,
            ddx,
            omega,
            parameter,
        ).parameter_derivative
        return derivative

    def _autodiff_values(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> _FreeAutodiffCache:
        key = (id(x), id(dx), id(ddx), float(omega), float(parameter))
        if self._free_autodiff_cache is not None and self._free_autodiff_cache.key == key:
            return self._free_autodiff_cache

        force_dofs = tuple(self.model.residual_force_dofs)
        coordinate_dofs = tuple(self.model.residual_coordinate_dofs)
        expected_force_shape = (self.prepared.t.size, len(force_dofs))
        expected_coordinate_shape = (self.prepared.t.size, len(coordinate_dofs))

        t_tensor = _as_torch(self.prepared.t, self._torch_device)
        x_tensor = _as_torch(x[:, list(coordinate_dofs)], self._torch_device)
        dx_tensor = _as_torch(dx[:, list(coordinate_dofs)], self._torch_device)
        ddx_tensor = _as_torch(ddx[:, list(coordinate_dofs)], self._torch_device)
        omega_tensor = torch.as_tensor(float(omega), dtype=torch.float64, device=self._torch_device)
        parameter_tensor = torch.as_tensor(float(parameter), dtype=torch.float64, device=self._torch_device)

        if x_tensor.shape != expected_coordinate_shape:
            raise ValueError(f"local_x must have shape {expected_coordinate_shape}, got {tuple(x_tensor.shape)}")

        force_tensor = self._call_torch_force(t_tensor, x_tensor, dx_tensor, ddx_tensor, omega_tensor, parameter_tensor)
        force_samples = _to_numpy(force_tensor)
        if force_samples.shape != expected_force_shape:
            raise ValueError(f"local residual force must have shape {expected_force_shape}, got {force_samples.shape}")

        jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]] = {}
        for variable in self._autodiff_variables:
            jacobian_tensor = self._differentiate_variable(
                variable,
                t_tensor,
                x_tensor,
                dx_tensor,
                ddx_tensor,
                omega_tensor,
                parameter_tensor,
            )
            jacobian = _to_numpy(jacobian_tensor)
            expected_jacobian_shape = (self.prepared.t.size, len(force_dofs), len(coordinate_dofs))
            if jacobian.shape != expected_jacobian_shape:
                raise ValueError(f"dG/d{variable} must have shape {expected_jacobian_shape}, got {jacobian.shape}")
            jacobian_by_variable[variable] = jacobian

        if self.model.autodiff_omega_dependent:
            omega_derivative = _to_numpy(
                self._differentiate_omega(t_tensor, x_tensor, dx_tensor, ddx_tensor, omega_tensor, parameter_tensor)
            )
        else:
            omega_derivative = np.zeros(expected_force_shape, dtype=np.float64)
        if omega_derivative.shape != expected_force_shape:
            raise ValueError(
                "local residual omega derivative must have shape "
                f"{expected_force_shape}, got {omega_derivative.shape}"
            )

        if self.model.autodiff_parameter_dependent:
            parameter_derivative = _to_numpy(
                self._differentiate_parameter(t_tensor, x_tensor, dx_tensor, ddx_tensor, omega_tensor, parameter_tensor)
            )
        else:
            parameter_derivative = np.zeros(expected_force_shape, dtype=np.float64)
        if parameter_derivative.shape != expected_force_shape:
            raise ValueError(
                "local residual parameter derivative must have shape "
                f"{expected_force_shape}, got {parameter_derivative.shape}"
            )

        self._free_autodiff_cache = _FreeAutodiffCache(
            key,
            force_samples,
            jacobian_by_variable,
            omega_derivative,
            parameter_derivative,
        )
        return self._free_autodiff_cache

    def _call_torch_force(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        omega: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        force = self.model.local_residual_force_torch(t, local_x, local_dx, local_ddx, omega, parameter)
        if not isinstance(force, torch.Tensor):
            raise TypeError("local_residual_force_torch must return a torch.Tensor")
        return force.to(dtype=torch.float64, device=self._torch_device)

    def _differentiate_variable(
        self,
        variable: JacobianVariable,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        omega: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        def summed_force(active_variable: torch.Tensor) -> torch.Tensor:
            active_x = active_variable if variable == "x" else local_x
            active_dx = active_variable if variable == "dx" else local_dx
            active_ddx = active_variable if variable == "ddx" else local_ddx
            return self._call_torch_force(t, active_x, active_dx, active_ddx, omega, parameter).sum(dim=0)

        jacobian = jacrev(summed_force)(_select_variable(variable, local_x, local_dx, local_ddx))
        return jacobian.permute(1, 0, 2).contiguous()

    def _differentiate_omega(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        omega: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        def force_for_omega(active_omega: torch.Tensor) -> torch.Tensor:
            return self._call_torch_force(t, local_x, local_dx, local_ddx, active_omega, parameter)

        return jacfwd(force_for_omega)(omega)

    def _differentiate_parameter(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        omega: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        def force_for_parameter(active_parameter: torch.Tensor) -> torch.Tensor:
            return self._call_torch_force(t, local_x, local_dx, local_ddx, omega, active_parameter)

        return jacfwd(force_for_parameter)(parameter)
