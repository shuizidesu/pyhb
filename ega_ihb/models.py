"""Model interfaces for EGA-IHB solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


JacobianVariable = Literal["x", "dx", "ddx"]
LinearBasisType = Literal["ddx", "dx", "x"]


@dataclass(frozen=True)
class LinearOperatorTerm:
    """One structured linear operator contribution.

    ``basis_type`` chooses the HB derivative basis and ``parameter_power``
    specifies the continuation-parameter multiplier.
    """

    matrix: NDArray[np.float64] | sparse.spmatrix
    basis_type: LinearBasisType
    parameter_power: float


@dataclass(frozen=True)
class ForcingTerm:
    """External force samples multiplied by a continuation-parameter power."""

    samples: NDArray[np.float64]
    parameter_power: float


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
    def linear_operator_terms(self) -> Sequence[LinearOperatorTerm]:
        """Structured linear operator terms for HB assembly."""

    @abstractmethod
    def forcing_terms(self, t: NDArray[np.float64]) -> Sequence[ForcingTerm]:
        """External force samples represented as powered forcing terms."""

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

    def nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Return ``dN/dparameter`` samples; defaults to zero."""

        return np.zeros((t.size, self.n_dof), dtype=np.float64)

    def rhs(self, t: float, y: NDArray[np.float64], parameter: float) -> NDArray[np.float64]:
        """Default first-order RHS for optional ODE verification."""

        n = self.n_dof
        x = y[:n]
        v = y[n:]
        x_sample = x.reshape(1, n)
        v_sample = v.reshape(1, n)
        ddx_sample = np.zeros((1, n), dtype=np.float64)
        force = _combine_forcing_terms(self.forcing_terms(np.array([t], dtype=np.float64)), parameter, n)[0]
        nonlinear = self.nonlinear_force(
            np.array([t], dtype=np.float64),
            x_sample,
            v_sample,
            ddx_sample,
            parameter,
        )[0]
        mass, damping, stiffness = _combine_time_operator_matrices(self.linear_operator_terms(), parameter, n)
        acceleration = np.linalg.solve(
            mass,
            force - nonlinear - damping @ v - stiffness @ x,
        )
        return np.concatenate((v, acceleration))


class CondensedSecondOrderTimeModel(ABC):
    """Interface for condensed second-order harmonic-balance models.

    Condensed models keep only the local nonlinear coordinates in the Newton
    unknowns and recover all remaining linear DOFs through Schur condensation.
    The nonlinear force and the coordinates it depends on may live on different
    physical DOFs.

    ``local_x[:, j]`` corresponds to ``nonlinear_coordinate_dofs[j]``.
    ``local_force[:, i]`` corresponds to ``nonlinear_force_dofs[i]``.
    ``local_jacobian[:, i, j]`` is
    ``d local_force_i / d local_x_j``.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of physical degrees of freedom."""

    @property
    @abstractmethod
    def nonlinear_force_dofs(self) -> tuple[int, ...]:
        """Global DOFs where local nonlinear forces are applied."""

    @property
    @abstractmethod
    def nonlinear_coordinate_dofs(self) -> tuple[int, ...]:
        """Global DOFs whose coordinates enter the local nonlinear law."""

    @abstractmethod
    def linear_operator_terms(self) -> Sequence[LinearOperatorTerm]:
        """Structured linear operator terms for the condensed solver."""

    @abstractmethod
    def forcing_terms(self, t: NDArray[np.float64]) -> Sequence[ForcingTerm]:
        """External force samples represented as powered forcing terms."""

    @abstractmethod
    def local_nonlinear_force_and_partials(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return local nonlinear force samples and local Jacobian samples."""

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Return local ``dN/dparameter`` samples; defaults to zero."""

        return np.zeros((t.size, len(self.nonlinear_force_dofs)), dtype=np.float64)


def _as_dense_matrix(matrix: NDArray[np.float64] | sparse.spmatrix) -> NDArray[np.float64]:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _combine_time_operator_matrices(
    terms: Sequence[LinearOperatorTerm],
    parameter: float,
    n_dof: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrices = {
        "ddx": np.zeros((n_dof, n_dof), dtype=np.float64),
        "dx": np.zeros((n_dof, n_dof), dtype=np.float64),
        "x": np.zeros((n_dof, n_dof), dtype=np.float64),
    }
    for term in terms:
        if term.basis_type not in matrices:
            raise ValueError(f"unsupported linear operator basis_type: {term.basis_type!r}")
        matrices[term.basis_type] += float(parameter) ** float(term.parameter_power) * _as_dense_matrix(term.matrix)
    return matrices["ddx"], matrices["dx"], matrices["x"]


def _combine_forcing_terms(
    terms: Sequence[ForcingTerm],
    parameter: float,
    n_dof: int,
) -> NDArray[np.float64]:
    result: NDArray[np.float64] | None = None
    for term in terms:
        samples = np.asarray(term.samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != n_dof:
            raise ValueError(f"forcing term samples must have shape (samples, {n_dof}), got {samples.shape}")
        contribution = float(parameter) ** float(term.parameter_power) * samples
        result = contribution if result is None else result + contribution
    if result is None:
        return np.zeros((0, n_dof), dtype=np.float64)
    return result
