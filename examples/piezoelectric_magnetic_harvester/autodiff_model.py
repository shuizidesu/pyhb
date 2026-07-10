from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from pyhb import AutodiffSecondOrderTimeModel, ForcingTerm, LinearOperatorTerm


@dataclass(frozen=True)
class PiezoelectricMagneticHarvesterParameters:
    mass_piezo: float = 4.877e-3
    damping_piezo: float = 0.01534
    stiffness_piezo: float = 148.8
    base_coupling_piezo: float = 1.04
    mass_magnetic: float = 6.167e-3
    damping_magnetic: float = 0.0178
    stiffness_magnetic: float = 144.9
    base_coupling_magnetic: float = 1.026
    electromechanical_coupling: float = 1.71e-4
    capacitance: float = 11.45e-9
    resistance: float = 400e3
    initial_gap: float = 10e-3
    magnetic_moment_1: float = 0.0192
    magnetic_moment_2: float = -0.0192
    base_acceleration: float = 2.828
    vacuum_permeability: float = 4.0 * np.pi * 1e-7


class PiezoelectricMagneticHarvesterAutodiffModel(AutodiffSecondOrderTimeModel):
    """Piezoelectric magnetic energy harvester model from Yuan et al. Eq. (16)."""

    def __init__(self, parameters: PiezoelectricMagneticHarvesterParameters | None = None) -> None:
        self.parameters = parameters or PiezoelectricMagneticHarvesterParameters()
        p = self.parameters
        self.mass = np.diag([p.mass_piezo, p.mass_magnetic, 0.0]).astype(np.float64)
        self.damping = np.array(
            [
                [p.damping_piezo, 0.0, 0.0],
                [0.0, p.damping_magnetic, 0.0],
                [-p.electromechanical_coupling, 0.0, p.capacitance],
            ],
            dtype=np.float64,
        )
        self.stiffness = np.array(
            [
                [p.stiffness_piezo, 0.0, p.electromechanical_coupling],
                [0.0, p.stiffness_magnetic, 0.0],
                [0.0, 0.0, 1.0 / p.resistance],
            ],
            dtype=np.float64,
        )
        self.magnetic_coefficient = (
            3.0 * p.vacuum_permeability * p.magnetic_moment_1 * p.magnetic_moment_2 / (2.0 * np.pi)
        )

    @property
    def n_dof(self) -> int:
        return 3

    @property
    def nonlinear_force_dofs(self) -> tuple[int, ...]:
        return (0, 1)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, ...]:
        return (0, 1)

    @property
    def autodiff_variables(self) -> tuple[str, ...]:
        return ("x",)

    @property
    def autodiff_parameter_dependent(self) -> bool:
        return False

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        p = self.parameters
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, 0] = -p.base_coupling_piezo * p.mass_piezo * p.base_acceleration * np.cos(t)
        force[:, 1] = -p.base_coupling_magnetic * p.mass_magnetic * p.base_acceleration * np.cos(t)
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force_torch(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        gap = local_x[:, 0] - local_x[:, 1] + float(self.parameters.initial_gap)
        magnetic_force = -float(self.magnetic_coefficient) / gap.pow(4)
        return torch.stack((magnetic_force, -magnetic_force), dim=1)
