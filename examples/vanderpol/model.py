from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyhb import (
    ForcingTerm,
    FreeFrequencySecondOrderTimeModel,
    LinearOperatorTerm,
    LocalResidualJacobianTerm,
)


@dataclass(frozen=True)
class VanderpolParameters:
    lambda_value: float = 0.87


class VanderpolModel(FreeFrequencySecondOrderTimeModel):
    def __init__(self, parameters: VanderpolParameters | None = None) -> None:
        self.parameters = parameters or VanderpolParameters()
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

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        return (ForcingTerm(np.zeros((t.size, self.n_dof), dtype=np.float64), 0.0),)

    def local_residual_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        lambda_value = float(self.parameters.lambda_value)
        generalized = -float(parameter) * float(omega) * (lambda_value - displacement**2) * velocity
        return generalized.reshape(-1, 1)

    def local_residual_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[LocalResidualJacobianTerm, ...]:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        lambda_value = float(self.parameters.lambda_value)
        return (
            LocalResidualJacobianTerm(
                0,
                "x",
                0,
                2.0 * float(parameter) * float(omega) * displacement * velocity,
            ),
            LocalResidualJacobianTerm(
                0,
                "dx",
                0,
                -float(parameter) * float(omega) * (lambda_value - displacement**2),
            ),
        )

    def local_residual_omega_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        lambda_value = float(self.parameters.lambda_value)
        derivative = -float(parameter) * (lambda_value - displacement**2) * velocity
        return derivative.reshape(-1, 1)

    def local_residual_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        lambda_value = float(self.parameters.lambda_value)
        derivative = -float(omega) * (lambda_value - displacement**2) * velocity
        return derivative.reshape(-1, 1)

