from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from pyhb import AutodiffSecondOrderTimeModel, ForcingTerm, LinearOperatorTerm


@dataclass(frozen=True)
class BistableBeamStructureParameters:
    a: float = 56e-3
    b: float = 110e-3
    mass: float = 12.34e-3
    damping: float = 0.1 * 12.34e-3
    k0: float = 2.3294e9
    k1: float = 2.1243e3


class BistableBeamStructureAutodiffModel(AutodiffSecondOrderTimeModel):
    """Single-DOF structural part of the bistable beam model."""

    def __init__(self, parameters: BistableBeamStructureParameters | None = None) -> None:
        self.parameters = parameters or BistableBeamStructureParameters()
        p = self.parameters
        self.mass = np.array([[p.mass]], dtype=np.float64)
        self.damping = np.array([[p.damping]], dtype=np.float64)
        self.stiffness = np.array([[0.0]], dtype=np.float64)

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
        return ("x",)

    @property
    def autodiff_parameter_dependent(self) -> bool:
        return False

    @property
    def positive_well_displacement(self) -> float:
        p = self.parameters
        return float(np.sqrt(p.a**2 - (0.5 * p.b) ** 2))

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, 0] = -float(self.parameters.mass) * np.cos(t) * 2.0
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force_torch(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        q = local_x[:, 0]
        p = self.parameters
        geometry = torch.sqrt(float(p.a) ** 2 - q.pow(2))
        lever = 0.5 * float(p.b) - geometry
        restoring_force = 4.0 * float(p.k0) * lever.pow(3) * q / geometry + 2.0 * float(p.k1) * lever * q / geometry
        return restoring_force.reshape(-1, 1)
