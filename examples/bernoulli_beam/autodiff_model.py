from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io import loadmat

from pyhb import AutodiffSecondOrderTimeModel, ForcingTerm, LinearOperatorTerm


DEFAULT_MATRIX_PATH = Path(__file__).resolve().parent / "data" / "beam_parameter_matrix.mat"


@dataclass(frozen=True)
class BernoulliBeamAutodiffParameters:
    force_amplitude: float = 1.0
    kappa: float = 4.0
    gamma: float = 0.001


class BernoulliBeamAutodiffModel(AutodiffSecondOrderTimeModel):
    def __init__(
        self,
        matrix_path: str | Path = DEFAULT_MATRIX_PATH,
        parameters: BernoulliBeamAutodiffParameters | None = None,
    ) -> None:
        self.parameters = parameters or BernoulliBeamAutodiffParameters()
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
        force[:, self.forcing_dof] = float(self.parameters.force_amplitude) * np.cos(t)
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force_torch(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        displacement = local_x[:, 0]
        velocity = local_dx[:, 0]
        nonlinear_force = (
            float(self.parameters.kappa) * displacement.pow(3)
            + float(self.parameters.gamma) * parameter.pow(3) * velocity.pow(3)
        )
        return nonlinear_force.reshape(-1, 1)
