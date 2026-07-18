from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyhb import ForcingTerm, LinearOperatorTerm, LocalJacobianMatrices, SecondOrderTimeModel


@dataclass(frozen=True)
class MachineAbsorberParameters:
    mu: float = 0.1
    alpha: float = 0.177
    epsilon: float = 0.044
    lambda_ratio: float = 1.0
    damping_machine: float = 0.1
    damping_absorber: float = 0.01
    alpha1: float = 0.03


class MachineAbsorberModel(SecondOrderTimeModel):
    def __init__(self, parameters: MachineAbsorberParameters | None = None) -> None:
        self.parameters = parameters or MachineAbsorberParameters()
        parameters = self.parameters
        lambda_sq = float(parameters.lambda_ratio) ** 2
        self.mass = np.array(
            [
                [1.0 + float(parameters.mu), float(parameters.mu)],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.damping = np.array(
            [
                [2.0 * float(parameters.damping_machine), 0.0],
                [0.0, 2.0 * float(parameters.damping_absorber)],
            ],
            dtype=np.float64,
        )
        self.stiffness = np.array(
            [
                [1.0, 0.0],
                [0.0, lambda_sq],
            ],
            dtype=np.float64,
        )

    @property
    def n_dof(self) -> int:
        return 2

    @property
    def nonlinear_force_dofs(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, int]:
        return (0, 1)

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, 0] = np.cos(t)
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        x = local_x[:, 0]
        y = local_x[:, 1]
        parameters = self.parameters
        lambda_sq = float(parameters.lambda_ratio) ** 2
        f1 = float(parameters.alpha1) * x**3
        f2 = lambda_sq * (
            -4.0 * float(parameters.alpha) * float(parameters.epsilon) * y * np.abs(y)
            + float(parameters.alpha) * float(parameters.epsilon) ** 2 * y**3
        )
        return np.column_stack((f1, f2))

    def local_nonlinear_jacobian(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> LocalJacobianMatrices:
        x = local_x[:, 0]
        y = local_x[:, 1]
        parameters = self.parameters
        lambda_sq = float(parameters.lambda_ratio) ** 2
        df1_dx = 3.0 * float(parameters.alpha1) * x**2
        df2_dy = lambda_sq * (
            -8.0 * float(parameters.alpha) * float(parameters.epsilon) * np.abs(y)
            + 3.0 * float(parameters.alpha) * float(parameters.epsilon) ** 2 * y**2
        )
        jacobian_x = np.zeros((t.size, 2, 2), dtype=np.float64)
        jacobian_x[:, 0, 0] = df1_dx
        jacobian_x[:, 1, 1] = df2_dy
        return LocalJacobianMatrices(x=jacobian_x)
