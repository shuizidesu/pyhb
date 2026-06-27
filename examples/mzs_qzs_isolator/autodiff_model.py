from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from numpy.typing import NDArray

from pyhb import AutodiffSecondOrderTimeModel, ForcingTerm, LinearOperatorTerm


@dataclass(frozen=True)
class MzsQzsAutodiffParameters:
    mu1: float = 0.53
    mu2: float = 0.6
    mu4: float = 1.8
    lambda1: float = 0.83
    lambda2: float = 0.50
    force_amplitude: float = 0.06
    zeta_v: float = 0.02
    zeta1: float = 0.001
    zeta2: float = 0.001
    zeta3: float = 0.001

    @property
    def mu3(self) -> float:
        return sqrt(1.0 - float(self.mu2) ** 2)

    @property
    def mu5(self) -> float:
        denominator = self.linear_stiffness - 2.0 * (1.0 + float(self.mu1)) * float(self.mu2) ** 2
        return float(self.lambda1) * float(self.mu4) / denominator

    @property
    def linear_stiffness(self) -> float:
        return float(self.lambda1) + float(self.lambda2) + 2.0

    @property
    def origin_tangent_stiffness(self) -> float:
        return (
            self.linear_stiffness
            - 2.0 * (1.0 + float(self.mu1)) * float(self.mu2) ** 2
            - float(self.lambda1) * float(self.mu4) / self.mu5
        )


class MzsQzsAutodiffModel(AutodiffSecondOrderTimeModel):
    def __init__(self, parameters: MzsQzsAutodiffParameters | None = None) -> None:
        self.parameters = parameters or MzsQzsAutodiffParameters()
        parameters = self.parameters
        self.mass = np.array([[1.0]], dtype=np.float64)
        self.damping = np.array([[2.0 * float(parameters.zeta_v)]], dtype=np.float64)
        self.stiffness = np.array([[float(parameters.linear_stiffness)]], dtype=np.float64)

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

    @property
    def autodiff_parameter_dependent(self) -> bool:
        return True

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
        x = local_x[:, 0]
        dx = local_dx[:, 0]
        parameters = self.parameters

        mu1 = float(parameters.mu1)
        mu2 = float(parameters.mu2)
        mu3 = float(parameters.mu3)
        mu4 = float(parameters.mu4)
        mu5 = float(parameters.mu5)
        lambda1 = float(parameters.lambda1)
        zeta1 = float(parameters.zeta1)
        zeta2 = float(parameters.zeta2)
        zeta3 = float(parameters.zeta3)

        mu2_sq = mu2**2
        mu5_sq = mu5**2
        upper_denominator = (mu3 + x).pow(2) + mu2_sq
        middle_denominator = mu5_sq + x.pow(2)
        lower_denominator = (mu3 - x).pow(2) + mu2_sq

        nonlinear_damping = (
            8.0 * zeta1 * mu2_sq / upper_denominator.pow(2)
            + 8.0 * zeta2 * mu5_sq / middle_denominator.pow(2)
            + 8.0 * zeta3 * mu2_sq / lower_denominator.pow(2)
        )
        damping_force = parameter * nonlinear_damping * dx

        restoring_force = (
            -(1.0 + mu1) * (mu3 + x) / torch.sqrt(upper_denominator)
            - (1.0 + mu1) * (x - mu3) / torch.sqrt(lower_denominator)
            - lambda1 * mu4 * x / torch.sqrt(middle_denominator)
        )
        return (damping_force + restoring_force).reshape(-1, 1)
