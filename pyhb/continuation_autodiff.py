"""Autodiff nonlinear derivatives for the full arc-length continuation solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import sparse
from torch.func import jacfwd, jacrev

from .continuation import ContinuationConfig, ContinuationSolver
from .harmonics import coefficient_matrix_from_fft, stack_fft_coefficients
from .models import AutodiffSecondOrderTimeModel, JacobianVariable


@dataclass(frozen=True)
class ContinuationAutodiffConfig(ContinuationConfig):
    """Continuation config with an optional Torch device selector."""

    torch_device: str | None = None


@dataclass
class _AutodiffCache:
    key: tuple[int, int, int, float]
    force_samples: NDArray[np.float64]
    jacobian_by_variable: dict[JacobianVariable, NDArray[np.float64]]
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
        self._autodiff_cache: _AutodiffCache | None = None
        super().__init__(model, config or ContinuationAutodiffConfig())
        self.model: AutodiffSecondOrderTimeModel
        self.config: ContinuationAutodiffConfig
        self._torch_device = _resolve_torch_device(self.config.torch_device)
        self._autodiff_variables = _validate_autodiff_variables(self.model.autodiff_variables)

    def _residual(
        self,
        coeff_line: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        nonlinear_coefficients = stack_fft_coefficients(
            self._global_nonlinear_force_samples(x, dx, ddx, parameter),
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )
        return (
            self._forcing_coefficients(parameter)
            - nonlinear_coefficients
            - self._linear_jacobian(parameter) @ coeff_line
        )

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
        cache = self._autodiff_values(x, dx, ddx, parameter)
        if not cache.jacobian_by_variable:
            return sparse.csc_matrix((size, size), dtype=np.float64)

        tensor_by_variable = {
            "x": context.s3_tensor_x,
            "dx": context.s3_tensor_dx,
            "ddx": context.s3_tensor_ddx,
        }
        force_dofs = np.asarray(self.model.nonlinear_force_dofs, dtype=np.int64)
        coordinate_dofs = np.asarray(self.model.nonlinear_coordinate_dofs, dtype=np.int64)
        force_count = force_dofs.size
        coordinate_count = coordinate_dofs.size
        force_columns = np.repeat(force_dofs, coordinate_count)
        coordinate_columns = np.tile(coordinate_dofs, force_count)
        row_offsets = np.arange(order, dtype=np.int64)
        col_offsets = np.arange(order, dtype=np.int64)

        row_chunks: list[NDArray[np.int64]] = []
        col_chunks: list[NDArray[np.int64]] = []
        data_chunks: list[NDArray[np.float64]] = []

        for variable, values in cache.jacobian_by_variable.items():
            flat_values = values.reshape(values.shape[0], force_count * coordinate_count)
            coeffs = coefficient_matrix_from_fft(
                flat_values,
                context.nonlinear_harmonics,
                context.sample_count,
                context.nonlinear_harmonic_indices,
            )
            blocks = np.einsum("abk,kt->abt", tensor_by_variable[variable], coeffs)
            term_blocks = np.moveaxis(blocks, 2, 0)
            row_indices = force_columns[:, None, None] * order + row_offsets[None, :, None]
            col_indices = coordinate_columns[:, None, None] * order + col_offsets[None, None, :]
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
            self._global_nonlinear_parameter_derivative(x, dx, ddx, parameter),
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

    def _global_nonlinear_force_samples(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        force = np.zeros((self.prepared.t.size, self.model.n_dof), dtype=np.float64)
        force[:, list(self.model.nonlinear_force_dofs)] = self._autodiff_values(
            x,
            dx,
            ddx,
            parameter,
        ).force_samples
        return force

    def _global_nonlinear_parameter_derivative(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        derivative = np.zeros((self.prepared.t.size, self.model.n_dof), dtype=np.float64)
        derivative[:, list(self.model.nonlinear_force_dofs)] = self._autodiff_values(
            x,
            dx,
            ddx,
            parameter,
        ).parameter_derivative
        return derivative

    def _autodiff_values(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> _AutodiffCache:
        key = (id(x), id(dx), id(ddx), float(parameter))
        if self._autodiff_cache is not None and self._autodiff_cache.key == key:
            return self._autodiff_cache

        force_dofs = tuple(self.model.nonlinear_force_dofs)
        coordinate_dofs = tuple(self.model.nonlinear_coordinate_dofs)
        expected_force_shape = (self.prepared.t.size, len(force_dofs))
        expected_coordinate_shape = (self.prepared.t.size, len(coordinate_dofs))

        t_tensor = _as_torch(self.prepared.t, self._torch_device)
        x_tensor = _as_torch(x[:, list(coordinate_dofs)], self._torch_device)
        dx_tensor = _as_torch(dx[:, list(coordinate_dofs)], self._torch_device)
        ddx_tensor = _as_torch(ddx[:, list(coordinate_dofs)], self._torch_device)
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
            expected_jacobian_shape = (self.prepared.t.size, len(force_dofs), len(coordinate_dofs))
            if jacobian.shape != expected_jacobian_shape:
                raise ValueError(
                    f"dN/d{variable} must have shape {expected_jacobian_shape}, got {jacobian.shape}"
                )
            jacobian_by_variable[variable] = jacobian

        if self.model.autodiff_parameter_dependent:
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

        self._autodiff_cache = _AutodiffCache(
            key,
            force_samples,
            jacobian_by_variable,
            parameter_derivative,
        )
        return self._autodiff_cache

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

        jacobian = jacrev(summed_force)(locals()[f"local_{variable}"])
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


def _resolve_torch_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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


def _to_numpy(values: torch.Tensor) -> NDArray[np.float64]:
    return np.asarray(values.detach().cpu().numpy(), dtype=np.float64)
