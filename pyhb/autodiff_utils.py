"""Small Torch helpers shared by pyHB autodiff paths."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .models import JacobianVariable


def _validate_autodiff_jacobian_mode(mode: str) -> str:
    normalized = str(mode)
    if normalized not in {"dense", "sparse"}:
        raise ValueError(
            "autodiff_jacobian_mode must be either 'dense' or 'sparse', "
            f"got {mode!r}"
        )
    return normalized


def _validate_rfft_harmonic_indices(
    harmonic_indices: tuple[int, ...],
    sample_count: int,
) -> None:
    nyquist_index = sample_count // 2
    if any(index < 0 or index > nyquist_index for index in harmonic_indices):
        raise ValueError(
            "sparse autodiff Jacobian projection requires all nonlinear harmonic "
            f"indices to be within the rFFT Nyquist range [0, {nyquist_index}]"
        )


def _resolve_torch_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_autodiff_variables(variables: tuple[JacobianVariable, ...]) -> tuple[JacobianVariable, ...]:
    normalized = tuple(variables)
    allowed: set[JacobianVariable] = {"x", "dx", "ddx"}
    if len(set(normalized)) != len(normalized):
        raise ValueError("autodiff_variables must not contain duplicates")
    unsupported = set(normalized) - allowed
    if unsupported:
        raise ValueError(f"unsupported autodiff variable(s): {sorted(unsupported)}")
    return normalized


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


def _as_torch(values: NDArray[np.float64], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float64, device=device)


def _to_numpy(values: torch.Tensor) -> NDArray[np.float64]:
    return np.asarray(values.detach().cpu().numpy(), dtype=np.float64)


def _compact_jacobian_fourier(
    jacobian: torch.Tensor,
    harmonic_indices: tuple[int, ...],
    sample_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Project strictly active spatial Jacobian pairs with a batched rFFT."""

    if jacobian.ndim != 3 or jacobian.shape[0] != sample_count:
        raise ValueError(
            "autodiff Jacobian must have shape "
            f"({sample_count}, force_count, coordinate_count), got {tuple(jacobian.shape)}"
        )
    _validate_rfft_harmonic_indices(harmonic_indices, sample_count)

    active_pairs = torch.nonzero(torch.any(jacobian != 0.0, dim=0), as_tuple=False)
    nonlinear_order = 2 * len(harmonic_indices) + 1
    if active_pairs.shape[0] == 0:
        empty_indices = np.empty(0, dtype=np.int64)
        empty_coefficients = np.empty((nonlinear_order, 0), dtype=np.float64)
        return empty_indices, empty_indices.copy(), empty_coefficients

    compact_samples = jacobian[:, active_pairs[:, 0], active_pairs[:, 1]]
    fft_values = torch.fft.rfft(compact_samples, dim=0) * (2.0 / sample_count)
    selected_indices = torch.as_tensor(
        harmonic_indices,
        dtype=torch.int64,
        device=jacobian.device,
    )
    selected = fft_values.index_select(0, selected_indices)
    coefficients = torch.cat(
        (
            fft_values[0:1].real / 2.0,
            selected.real,
            -selected.imag,
        ),
        dim=0,
    )
    pair_indices = active_pairs.detach().cpu().numpy()
    return (
        np.asarray(pair_indices[:, 0], dtype=np.int64),
        np.asarray(pair_indices[:, 1], dtype=np.int64),
        _to_numpy(coefficients),
    )
