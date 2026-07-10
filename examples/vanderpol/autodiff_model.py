from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pyhb import AutodiffFreeFrequencySecondOrderTimeModel, LinearOperatorTerm


@dataclass(frozen=True)
class VanderpolAutodiffParameters:
    lambda_value: float = 0.87


class VanderpolAutodiffModel(AutodiffFreeFrequencySecondOrderTimeModel):
    def __init__(self, parameters: VanderpolAutodiffParameters | None = None) -> None:
        self.parameters = parameters or VanderpolAutodiffParameters()
        self.mass = np.array([[1.0]], dtype=np.float64)
        self.stiffness = np.array([[1.0]], dtype=np.float64)

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def residual_force_dofs(self) -> tuple[int, ...]:
        return (0,)

    @property
    def residual_coordinate_dofs(self) -> tuple[int, ...]:
        return (0,)

    @property
    def autodiff_variables(self) -> tuple[str, ...]:
        return ("x", "dx")

    @property
    def autodiff_omega_dependent(self) -> bool:
        return True

    @property
    def autodiff_parameter_dependent(self) -> bool:
        return True

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def local_residual_force_torch(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        omega: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        lambda_value = float(self.parameters.lambda_value)
        generalized = -parameter * omega * (lambda_value - displacement.pow(2)) * velocity
        return generalized.reshape(-1, 1)
