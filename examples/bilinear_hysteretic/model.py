from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyhb import ForcingTerm, LinearOperatorTerm, LocalJacobianMatrices, SecondOrderTimeModel


@dataclass(frozen=True)
class BilinearHystereticParameters:
    xi: float = 0.05
    force_amplitude: float = 1.0
    outer_slope: float = 10.0
    inner_slope: float = 1.2
    outer_offset: float = 1.1
    inner_offset: float = 0.22
    outer_displacement: float = 0.15
    loading_switch: float = -0.10
    unloading_switch: float = 0.10


class BilinearHystereticModel(SecondOrderTimeModel):
    def __init__(self, parameters: BilinearHystereticParameters | None = None) -> None:
        self.parameters = parameters or BilinearHystereticParameters()
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

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        y = local_x[:, 0]
        dy = local_dx[:, 0]
        force, _ = self._piecewise_force_and_slope(y, dy)
        return force.reshape(-1, 1)

    def local_nonlinear_jacobian(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> LocalJacobianMatrices:
        y = local_x[:, 0]
        dy = local_dx[:, 0]
        _, slope = self._piecewise_force_and_slope(y, dy)
        return LocalJacobianMatrices(x=slope.reshape(-1, 1, 1))

    def _piecewise_force_and_slope(
        self,
        y: NDArray[np.float64],
        dy: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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

        loading_outer_mask = y <= loading_switch
        unloading_outer_mask = y >= unloading_switch
        loading_middle = np.where(loading_outer_mask, loading_outer, loading_inner)
        unloading_middle = np.where(unloading_outer_mask, unloading_outer, unloading_inner)
        loading_slope = np.where(loading_outer_mask, outer_slope, inner_slope)
        unloading_slope = np.where(unloading_outer_mask, outer_slope, inner_slope)

        loading_mask = dy > 0.0
        middle = np.where(loading_mask, loading_middle, unloading_middle)
        middle_slope = np.where(loading_mask, loading_slope, unloading_slope)

        left_outer_mask = y <= -outer_displacement
        right_outer_mask = y >= outer_displacement
        force = np.where(
            left_outer_mask,
            loading_outer,
            np.where(right_outer_mask, unloading_outer, middle),
        )
        slope = np.where(
            left_outer_mask | right_outer_mask,
            outer_slope,
            middle_slope,
        )
        return force, slope
