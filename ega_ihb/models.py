"""Model interfaces for EGA-IHB solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


JacobianVariable = Literal["x", "dx", "ddx"]


@dataclass(frozen=True)
class NonlinearJacobianTerm:
    """One nonzero time-domain nonlinear Jacobian entry.

    ``values`` contains samples of
    ``d nonlinear_force[force_dof] / d variable[coordinate_dof]``.
    The solver projects these samples into HB space.
    """

    force_dof: int
    variable: JacobianVariable
    coordinate_dof: int
    values: NDArray[np.float64]


class SecondOrderTimeModel(ABC):
    """Interface for second-order systems solved by harmonic balance.

    Models represent systems in residual form
    ``F(t, p) - N(t, x, dx, ddx, p) - M*ddx - C(p)*dx - K(p)*x = 0``.
    The continuation solver projects this time-domain residual into HB
    coefficient space. Models provide nonlinear Jacobian entries in time
    domain and the solver assembles the HB-space blocks.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of physical degrees of freedom."""

    @abstractmethod
    def mass_matrix(self, parameter: float | None = None) -> NDArray[np.float64]:
        """Mass matrix ``M`` or effective mass matrix for the active parameter."""

    @abstractmethod
    def damping_matrix(self, parameter: float) -> NDArray[np.float64]:
        """Effective damping matrix for the active continuation parameter."""

    @abstractmethod
    def stiffness_matrix(self, parameter: float) -> NDArray[np.float64]:
        """Effective stiffness matrix for the active continuation parameter."""

    @abstractmethod
    def forcing(self, t: NDArray[np.float64], parameter: float) -> NDArray[np.float64]:
        """External force samples shaped ``(samples, n_dof)``."""

    @abstractmethod
    def nonlinear_force(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Nonlinear force samples shaped ``(samples, n_dof)``."""

    @abstractmethod
    def nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NonlinearJacobianTerm, ...]:
        """Nonzero time-domain nonlinear Jacobian entries."""

    @abstractmethod
    def parameter_derivative(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Time-domain samples of ``dR/dparameter`` shaped ``(samples, n_dof)``."""

    def rhs(self, t: float, y: NDArray[np.float64], parameter: float) -> NDArray[np.float64]:
        """Default first-order RHS for optional ODE verification."""

        n = self.n_dof
        x = y[:n]
        v = y[n:]
        x_sample = x.reshape(1, n)
        v_sample = v.reshape(1, n)
        ddx_sample = np.zeros((1, n), dtype=np.float64)
        force = self.forcing(np.array([t], dtype=np.float64), parameter)[0]
        nonlinear = self.nonlinear_force(
            np.array([t], dtype=np.float64),
            x_sample,
            v_sample,
            ddx_sample,
            parameter,
        )[0]
        acceleration = np.linalg.solve(
            self.mass_matrix(parameter),
            force - nonlinear - self.damping_matrix(parameter) @ v - self.stiffness_matrix(parameter) @ x,
        )
        return np.concatenate((v, acceleration))
