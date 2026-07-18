from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from pyhb import ForcingTerm, LinearOperatorTerm, LocalJacobianMatrices, SecondOrderTimeModel

DEFAULT_MATRIX_PATH = Path(__file__).resolve().parent / "data" / "beam_parameter_matrix.mat"


@dataclass(frozen=True)
class BernoulliBeamParameters:
    force_amplitude: float = 1.0
    kappa: float = 4.0
    gamma: float = 0.001


class BernoulliBeamModel(SecondOrderTimeModel):
    def __init__(
        self,
        matrix_path: str | Path = DEFAULT_MATRIX_PATH,
        parameters: BernoulliBeamParameters | None = None,
    ) -> None:
        self.parameters = parameters or BernoulliBeamParameters()
        matrix_data = loadmat(matrix_path)
        self.mass = np.asarray(matrix_data["M"], dtype=np.float64)
        self.damping = np.asarray(matrix_data["C"], dtype=np.float64)
        self.stiffness = np.asarray(matrix_data["K"], dtype=np.float64)
        self._n_dof = int(self.mass.shape[0])
        if self.mass.shape != self.damping.shape or self.mass.shape != self.stiffness.shape:
            raise ValueError("M, C, and K must have the same shape")
        if self.mass.shape[0] != self.mass.shape[1]:
            raise ValueError("system matrices must be square")
        self.nonlinear_dof = self._n_dof - 2

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def nonlinear_force_dofs(self) -> tuple[int, ...]:
        return (self.nonlinear_dof,)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, ...]:
        return (self.nonlinear_dof,)

    @property
    def forcing_dof(self) -> int:
        return self.nonlinear_dof

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, self.forcing_dof] = float(self.parameters.force_amplitude) * np.cos(t)
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        nonlinear_force = (
            float(self.parameters.kappa) * displacement**3
            + float(self.parameters.gamma) * float(parameter) ** 3 * velocity**3
        )
        return nonlinear_force.reshape(-1, 1)

    def local_nonlinear_jacobian(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> LocalJacobianMatrices:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        return LocalJacobianMatrices(
            x=(3.0 * float(self.parameters.kappa) * displacement**2).reshape(-1, 1, 1),
            dx=(
                3.0 * float(self.parameters.gamma) * float(parameter) ** 3 * velocity**2
            ).reshape(-1, 1, 1),
        )

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        velocity = local_dx[:, 0]
        derivative = 3.0 * float(self.parameters.gamma) * float(parameter) ** 2 * velocity**3
        return derivative.reshape(-1, 1)
