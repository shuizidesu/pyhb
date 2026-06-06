from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyhb import ForcingTerm, LinearOperatorTerm, LocalNonlinearJacobianTerm, SecondOrderTimeModel


FRACTIONAL_FREQUENCY = 1.2


@dataclass(frozen=True)
class FourDofSystem:
    n_dof: int
    mass: NDArray[np.float64]
    damping: NDArray[np.float64]
    rotational_damping: NDArray[np.float64]
    stiffness: NDArray[np.float64]
    excitation: float
    nonlinear_scale: float


def default_four_dof_system() -> FourDofSystem:
    n_dof = 4
    mass_value = 20.0
    l1 = 0.25
    l2 = 0.5
    eccentricity_length = 0.03e-3
    damping_ratio = 0.015
    polar_inertia = 0.144
    diametral_inertia = 0.072
    stiffness_value = 1.5711e6
    cubic_stiffness = 2.8e12
    static_deflection = mass_value * 9.8 / stiffness_value
    damping_value = 2.0 * mass_value * damping_ratio

    w1 = 2.0 * stiffness_value / mass_value
    w2 = (l1**2 + l2**2) * stiffness_value / diametral_inertia
    a1 = stiffness_value * (l2 - l1) / mass_value / (l1 + l2)
    b1 = stiffness_value * (l2**2 - l1**2) / diametral_inertia
    a2 = 2.0 * damping_value / mass_value
    a3 = damping_value * (l2 - l1) / mass_value / (l1 + l2)
    b2 = damping_value * (l1**2 + l2**2) / diametral_inertia
    b3 = polar_inertia / diametral_inertia
    b4 = damping_value * (l1**2 - l2**2) / diametral_inertia
    nonlinear_scale = 2.0 * static_deflection**2 * cubic_stiffness / mass_value
    excitation = eccentricity_length / static_deflection

    mass = np.eye(4, dtype=np.float64)
    damping = np.array(
        [
            [a2, 0.0, 0.0, a3],
            [0.0, a2, a3, 0.0],
            [0.0, b4, b2, 0.0],
            [b4, 0.0, 0.0, b2],
        ],
        dtype=np.float64,
    )
    rotational_damping = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, b3],
            [0.0, 0.0, -b3, 0.0],
        ],
        dtype=np.float64,
    )
    stiffness = np.array(
        [
            [w1, 0.0, 0.0, a1],
            [0.0, w1, a1, 0.0],
            [0.0, b1, w2, 0.0],
            [b1, 0.0, 0.0, w2],
        ],
        dtype=np.float64,
    )
    return FourDofSystem(n_dof, mass, damping, rotational_damping, stiffness, excitation, nonlinear_scale)


class FourDofFractionalModel(SecondOrderTimeModel):
    def __init__(self, system: FourDofSystem | None = None, fractional_frequency: float = FRACTIONAL_FREQUENCY) -> None:
        self.system = system or default_four_dof_system()
        self.fractional_frequency = float(fractional_frequency)

    @property
    def n_dof(self) -> int:
        return self.system.n_dof

    @property
    def nonlinear_force_dofs(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, int]:
        return (0, 1)

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.system.mass, "ddx", 0.0),
            LinearOperatorTerm(self.system.damping, "dx", 0.0),
            LinearOperatorTerm(self.system.rotational_damping, "dx", -1.0),
            LinearOperatorTerm(self.system.stiffness, "x", -2.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, 0] = self.system.excitation * np.cos(t)
        force[:, 1] = self.system.excitation * np.sin(t)
        fractional_force = self.system.excitation * np.cos(self.fractional_frequency * t)
        force[:, 0] += fractional_force
        force[:, 1] += fractional_force
        return (ForcingTerm(force, 0.0),)

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return (self.system.nonlinear_scale / parameter**2) * local_x**3

    def local_nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[LocalNonlinearJacobianTerm, ...]:
        factor = 3.0 * self.system.nonlinear_scale / parameter**2
        return (
            LocalNonlinearJacobianTerm(0, "x", 0, factor * local_x[:, 0] ** 2),
            LocalNonlinearJacobianTerm(1, "x", 1, factor * local_x[:, 1] ** 2),
        )

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return (-2.0 * self.system.nonlinear_scale / parameter**3) * local_x**3
