"""Small Torch helpers shared by pyHB autodiff paths."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .models import JacobianVariable


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
