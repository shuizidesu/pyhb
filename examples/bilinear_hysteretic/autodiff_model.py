from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from pyhb import AutodiffSecondOrderTimeModel, ForcingTerm, LinearOperatorTerm


@dataclass(frozen=True)
class BilinearHystereticAutodiffParameters:
    xi: float = 0.05
    force_amplitude: float = 1.0
    outer_slope: float = 10.0
    inner_slope: float = 1.2
    outer_offset: float = 1.1
    inner_offset: float = 0.22
    outer_displacement: float = 0.15
    loading_switch: float = -0.10
    unloading_switch: float = 0.10


class BilinearHystereticAutodiffModel(AutodiffSecondOrderTimeModel):
    def __init__(self, parameters: BilinearHystereticAutodiffParameters | None = None) -> None:
        self.parameters = parameters or BilinearHystereticAutodiffParameters()
        parameters = self.parameters
        self.mass = np.array([[1.0]], dtype=np.float64)
        self.damping = np.array([[2.0 * float(parameters.xi)]], dtype=np.float64)
        self.stiffness = np.zeros((1, 1), dtype=np.float64)

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def nonlinear_force_dofs(self) -> tuple[int, ...]:
        return (0,)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, ...]:
        return (0,)

    @property
    def autodiff_variables(self) -> tuple[str, ...]:
        return ("x", "dx")

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, 0] = float(self.parameters.force_amplitude) * np.cos(t)
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force_torch(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        y = local_x[:, 0]
        dy = local_dx[:, 0]
        parameters = self.parameters

        outer_slope = float(parameters.outer_slope)
        inner_slope = float(parameters.inner_slope)
        outer_offset = float(parameters.outer_offset)
        inner_offset = float(parameters.inner_offset)
        outer_displacement = float(parameters.outer_displacement)
        loading_switch = float(parameters.loading_switch)
        unloading_switch = float(parameters.unloading_switch)

        loading_outer = outer_slope * y + outer_offset
        loading_inner = inner_slope * y + inner_offset
        unloading_outer = outer_slope * y - outer_offset
        unloading_inner = inner_slope * y - inner_offset

        loading_middle = torch.where(y <= loading_switch, loading_outer, loading_inner)
        unloading_middle = torch.where(y >= unloading_switch, unloading_outer, unloading_inner)
        middle = torch.where(dy > 0.0, loading_middle, unloading_middle)
        force = torch.where(
            y <= -outer_displacement,
            loading_outer,
            torch.where(y >= outer_displacement, unloading_outer, middle),
        )
        return force.reshape(-1, 1)
