from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from pyhb import ForcingTerm, LinearOperatorTerm, LocalNonlinearJacobianTerm, SecondOrderTimeModel


@dataclass(frozen=True)
class MzsQzsParameters:
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


class MzsQzsModel(SecondOrderTimeModel):
    def __init__(self, parameters: MzsQzsParameters | None = None) -> None:
        self.parameters = parameters or MzsQzsParameters()
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

    def _nonlinear_components(
        self,
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
        upper = (mu3 + x) ** 2 + mu2_sq
        middle = mu5_sq + x**2
        lower = (mu3 - x) ** 2 + mu2_sq

        damping = (
            8.0 * zeta1 * mu2_sq / upper**2
            + 8.0 * zeta2 * mu5_sq / middle**2
            + 8.0 * zeta3 * mu2_sq / lower**2
        )
        damping_derivative = (
            -32.0 * zeta1 * mu2_sq * (mu3 + x) / upper**3
            - 32.0 * zeta2 * mu5_sq * x / middle**3
            + 32.0 * zeta3 * mu2_sq * (mu3 - x) / lower**3
        )
        restoring = (
            -(1.0 + mu1) * (mu3 + x) / np.sqrt(upper)
            - (1.0 + mu1) * (x - mu3) / np.sqrt(lower)
            - lambda1 * mu4 * x / np.sqrt(middle)
        )
        restoring_derivative = (
            -(1.0 + mu1) * mu2_sq / upper ** 1.5
            - (1.0 + mu1) * mu2_sq / lower ** 1.5
            - lambda1 * mu4 * mu5_sq / middle ** 1.5
        )
        return damping, damping_derivative, restoring, restoring_derivative

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        x = local_x[:, 0]
        phase_velocity = local_dx[:, 0]
        damping, _, restoring, _ = self._nonlinear_components(x)
        force = float(parameter) * damping * phase_velocity + restoring
        return force.reshape(-1, 1)

    def local_nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[LocalNonlinearJacobianTerm, ...]:
        x = local_x[:, 0]
        phase_velocity = local_dx[:, 0]
        damping, damping_derivative, _, restoring_derivative = self._nonlinear_components(x)
        d_force_dx = float(parameter) * phase_velocity * damping_derivative + restoring_derivative
        d_force_dphase_velocity = float(parameter) * damping
        return (
            LocalNonlinearJacobianTerm(0, "x", 0, d_force_dx),
            LocalNonlinearJacobianTerm(0, "dx", 0, d_force_dphase_velocity),
        )

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        damping, _, _, _ = self._nonlinear_components(local_x[:, 0])
        derivative = damping * local_dx[:, 0]
        return derivative.reshape(-1, 1)
