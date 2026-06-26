"""Model interfaces for pyHB solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


JacobianVariable = Literal["x", "dx", "ddx"]
LinearBasisType = Literal["ddx", "dx", "x"]


@dataclass(frozen=True)
class LinearOperatorTerm:
    """One structured linear operator contribution.

    ``basis_type`` chooses the HB derivative basis and ``omega_power``
    specifies the response-frequency multiplier.
    """

    matrix: NDArray[np.float64] | sparse.spmatrix
    basis_type: LinearBasisType
    omega_power: float


@dataclass(frozen=True)
class ForcingTerm:
    """External force samples multiplied by a response-frequency power."""

    samples: NDArray[np.float64]
    omega_power: float


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


@dataclass(frozen=True)
class LocalNonlinearJacobianTerm:
    """One nonzero local nonlinear Jacobian entry.

    ``force_index`` indexes ``nonlinear_force_dofs`` and
    ``coordinate_index`` indexes ``nonlinear_coordinate_dofs``.
    """

    force_index: int
    variable: JacobianVariable
    coordinate_index: int
    values: NDArray[np.float64]


@dataclass(frozen=True)
class LocalResidualJacobianTerm:
    """One nonzero local generalized residual Jacobian entry.

    ``force_index`` indexes ``residual_force_dofs`` and ``coordinate_index``
    indexes ``residual_coordinate_dofs``.
    """

    force_index: int
    variable: JacobianVariable
    coordinate_index: int
    values: NDArray[np.float64]


@dataclass(frozen=True)
class HarmonicCoefficientConstraint:
    """Fix one HB coefficient to remove free-frequency phase ambiguity."""

    dof: int
    coefficient_index: int
    value: float = 0.0


@dataclass(frozen=True)
class ReferencePhaseCondition:
    """Use an AUTO-style reference phase condition for free-frequency continuation.

    This marker has no user parameters. After the initial fixed-coefficient
    solve, the solver builds the phase row from the previous accepted HB
    coefficients and the harmonic derivative map, then enforces
    ``g(q) = q_ref^T D q = 0`` to remove time-shift ambiguity.
    """


class SecondOrderTimeModel(ABC):
    """Interface for second-order systems solved by harmonic balance.

    Models represent systems in residual form
    ``F(t, p) - N(t, x, dx, ddx, p) - M*ddx - C(p)*dx - K(p)*x = 0``.
    The continuation solver projects this time-domain residual into HB
    coefficient space. Models describe nonlinearities locally; the base class
    scatters local forces and Jacobian terms to global DOFs for full-system
    solvers.
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
        """Global DOFs whose states enter the local nonlinear law."""

    @abstractmethod
    def linear_operator_terms(self) -> Sequence[LinearOperatorTerm]:
        """Structured linear operator terms for HB assembly."""

    @abstractmethod
    def forcing_terms(self, t: NDArray[np.float64]) -> Sequence[ForcingTerm]:
        """External force samples represented as powered forcing terms."""

    @abstractmethod
    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Local nonlinear force samples shaped ``(samples, force_count)``."""

    @abstractmethod
    def local_nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[LocalNonlinearJacobianTerm, ...]:
        """Nonzero local time-domain nonlinear Jacobian entries."""

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Return local explicit ``dN/dparameter`` samples; defaults to zero."""

        return np.zeros((t.size, len(self.nonlinear_force_dofs)), dtype=np.float64)

    def nonlinear_force(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Global nonlinear force samples shaped ``(samples, n_dof)``."""

        force_dofs = _validate_dofs("nonlinear_force_dofs", self.nonlinear_force_dofs, self.n_dof)
        local_x, local_dx, local_ddx = _extract_local_states(
            x,
            dx,
            ddx,
            _validate_dofs("nonlinear_coordinate_dofs", self.nonlinear_coordinate_dofs, self.n_dof),
        )
        local_force = np.asarray(
            self.local_nonlinear_force(t, local_x, local_dx, local_ddx, parameter),
            dtype=np.float64,
        )
        expected_shape = (t.size, len(force_dofs))
        if local_force.shape != expected_shape:
            raise ValueError(f"local nonlinear force must have shape {expected_shape}, got {local_force.shape}")
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, list(force_dofs)] = local_force
        return force

    def nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NonlinearJacobianTerm, ...]:
        """Global nonlinear Jacobian terms generated from local terms."""

        force_dofs = _validate_dofs("nonlinear_force_dofs", self.nonlinear_force_dofs, self.n_dof)
        coordinate_dofs = _validate_dofs("nonlinear_coordinate_dofs", self.nonlinear_coordinate_dofs, self.n_dof)
        local_x, local_dx, local_ddx = _extract_local_states(x, dx, ddx, coordinate_dofs)
        terms = []
        for term in self.local_nonlinear_jacobian_terms(t, local_x, local_dx, local_ddx, parameter):
            if term.variable not in ("x", "dx", "ddx"):
                raise ValueError(f"unsupported local nonlinear Jacobian variable {term.variable!r}")
            if not (0 <= term.force_index < len(force_dofs)):
                raise ValueError(f"local force_index out of range: {term.force_index}")
            if not (0 <= term.coordinate_index < len(coordinate_dofs)):
                raise ValueError(f"local coordinate_index out of range: {term.coordinate_index}")
            values = np.asarray(term.values, dtype=np.float64).reshape(-1)
            if values.shape[0] != t.size:
                raise ValueError(
                    "local nonlinear Jacobian term values must have one value per time sample; "
                    f"got {values.shape[0]}, expected {t.size}"
                )
            terms.append(
                NonlinearJacobianTerm(
                    force_dofs[term.force_index],
                    term.variable,
                    coordinate_dofs[term.coordinate_index],
                    values,
                )
            )
        return tuple(terms)

    def nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Global explicit ``dN/dparameter`` samples."""

        force_dofs = _validate_dofs("nonlinear_force_dofs", self.nonlinear_force_dofs, self.n_dof)
        local_x, local_dx, local_ddx = _extract_local_states(
            x,
            dx,
            ddx,
            _validate_dofs("nonlinear_coordinate_dofs", self.nonlinear_coordinate_dofs, self.n_dof),
        )
        local_derivative = np.asarray(
            self.local_nonlinear_parameter_derivative(t, local_x, local_dx, local_ddx, parameter),
            dtype=np.float64,
        )
        expected_shape = (t.size, len(force_dofs))
        if local_derivative.shape != expected_shape:
            raise ValueError(
                "local nonlinear parameter derivative must have shape "
                f"{expected_shape}, got {local_derivative.shape}"
            )
        derivative = np.zeros((t.size, self.n_dof), dtype=np.float64)
        derivative[:, list(force_dofs)] = local_derivative
        return derivative

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
        matrices[term.basis_type] += float(parameter) ** float(term.omega_power) * _as_dense_matrix(term.matrix)
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
        contribution = float(parameter) ** float(term.omega_power) * samples
        result = contribution if result is None else result + contribution
    if result is None:
        return np.zeros((0, n_dof), dtype=np.float64)
    return result


def _validate_dofs(name: str, dofs: Sequence[int], n_dof: int) -> tuple[int, ...]:
    normalized = tuple(int(dof) for dof in dofs)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique DOFs")
    if any(dof < 0 or dof >= n_dof for dof in normalized):
        raise ValueError(f"{name} contains an out-of-range DOF")
    return normalized


def _extract_local_states(
    x: NDArray[np.float64],
    dx: NDArray[np.float64],
    ddx: NDArray[np.float64],
    coordinate_dofs: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    indices = list(coordinate_dofs)
    return x[:, indices], dx[:, indices], ddx[:, indices]


class AutodiffSecondOrderTimeModel(SecondOrderTimeModel):
    """Second-order model whose local nonlinear derivatives are built by Torch autodiff."""

    @property
    def autodiff_variables(self) -> tuple[JacobianVariable, ...]:
        """Variables whose local nonlinear Jacobian terms should be differentiated."""

        return ("x",)

    @property
    def autodiff_parameter_dependent(self) -> bool:
        """Whether the local nonlinear force has explicit continuation-parameter dependence."""

        return False

    @abstractmethod
    def local_nonlinear_force_torch(
        self,
        t: Any,
        local_x: Any,
        local_dx: Any,
        local_ddx: Any,
        parameter: Any,
    ) -> Any:
        """Return local nonlinear force samples as a Torch tensor.

        Inputs are batched by time sample. Shapes are ``t=(samples,)`` and
        ``local_x/local_dx/local_ddx=(samples, coordinate_count)``. The return
        value must be shaped ``(samples, force_count)``.
        """

    def local_nonlinear_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Evaluate local nonlinear force through the Torch implementation."""

        import torch

        device = torch.device("cpu")
        with torch.no_grad():
            t_tensor = torch.as_tensor(t, dtype=torch.float64, device=device)
            x_tensor = torch.as_tensor(local_x, dtype=torch.float64, device=device)
            dx_tensor = torch.as_tensor(local_dx, dtype=torch.float64, device=device)
            ddx_tensor = torch.as_tensor(local_ddx, dtype=torch.float64, device=device)
            parameter_tensor = torch.as_tensor(float(parameter), dtype=torch.float64, device=device)
            force = self.local_nonlinear_force_torch(
                t_tensor,
                x_tensor,
                dx_tensor,
                ddx_tensor,
                parameter_tensor,
            )
        return np.asarray(force.detach().cpu().numpy(), dtype=np.float64)

    def local_nonlinear_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> tuple[LocalNonlinearJacobianTerm, ...]:
        """Autodiff models do not expose handwritten nonlinear Jacobian terms."""

        raise NotImplementedError("use ContinuationAutodiffSolver for AutodiffSecondOrderTimeModel")

    def local_nonlinear_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        """Autodiff models only expose explicit parameter derivatives through the autodiff solver."""

        if not self.autodiff_parameter_dependent:
            return np.zeros((t.size, len(self.nonlinear_force_dofs)), dtype=np.float64)
        raise NotImplementedError("use ContinuationAutodiffSolver for parameter-dependent autodiff nonlinearities")


class FreeFrequencySecondOrderTimeModel(ABC):
    """Interface for autonomous/free-frequency second-order HB systems.

    Models represent systems in residual form
    ``F0(t, omega) - G(t, x, dx, ddx, omega, p) - L0(omega) q = 0``.
    ``G`` is a local generalized residual contribution: it may contain
    nonlinear forces, parameter-dependent corrections, or autonomous
    self-excitation terms. The solver scatters local values and derivatives
    to global DOFs before projecting them into HB space.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of physical degrees of freedom."""

    @property
    @abstractmethod
    def residual_force_dofs(self) -> tuple[int, ...]:
        """Global DOFs where local generalized residual terms are applied."""

    @property
    @abstractmethod
    def residual_coordinate_dofs(self) -> tuple[int, ...]:
        """Global DOFs whose states enter the local generalized residual law."""

    @abstractmethod
    def linear_operator_terms(self) -> Sequence[LinearOperatorTerm]:
        """Structured linear operator terms powered by the unknown frequency."""

    def forcing_terms(self, t: NDArray[np.float64]) -> Sequence[ForcingTerm]:
        """External force samples powered by the unknown frequency; defaults to zero."""

        return (ForcingTerm(np.zeros((t.size, self.n_dof), dtype=np.float64), 0.0),)

    @abstractmethod
    def local_residual_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Local generalized residual samples shaped ``(samples, force_count)``."""

    @abstractmethod
    def local_residual_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[LocalResidualJacobianTerm, ...]:
        """Nonzero local time-domain generalized residual Jacobian entries."""

    def local_residual_omega_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Return local explicit ``dG/domega`` samples; defaults to zero."""

        return np.zeros((t.size, len(self.residual_force_dofs)), dtype=np.float64)

    def local_residual_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Return local explicit ``dG/dparameter`` samples; defaults to zero."""

        return np.zeros((t.size, len(self.residual_force_dofs)), dtype=np.float64)

    def residual_force(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Global generalized residual samples shaped ``(samples, n_dof)``."""

        force_dofs = _validate_dofs("residual_force_dofs", self.residual_force_dofs, self.n_dof)
        coordinate_dofs = _validate_dofs("residual_coordinate_dofs", self.residual_coordinate_dofs, self.n_dof)
        local_x, local_dx, local_ddx = _extract_local_states(x, dx, ddx, coordinate_dofs)
        local_force = np.asarray(
            self.local_residual_force(t, local_x, local_dx, local_ddx, omega, parameter),
            dtype=np.float64,
        )
        expected_shape = (t.size, len(force_dofs))
        if local_force.shape != expected_shape:
            raise ValueError(f"local residual force must have shape {expected_shape}, got {local_force.shape}")
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        force[:, list(force_dofs)] = local_force
        return force

    def residual_jacobian_terms(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[NonlinearJacobianTerm, ...]:
        """Global generalized residual Jacobian terms generated from local terms."""

        force_dofs = _validate_dofs("residual_force_dofs", self.residual_force_dofs, self.n_dof)
        coordinate_dofs = _validate_dofs("residual_coordinate_dofs", self.residual_coordinate_dofs, self.n_dof)
        local_x, local_dx, local_ddx = _extract_local_states(x, dx, ddx, coordinate_dofs)
        terms = []
        for term in self.local_residual_jacobian_terms(t, local_x, local_dx, local_ddx, omega, parameter):
            if term.variable not in ("x", "dx", "ddx"):
                raise ValueError(f"unsupported local residual Jacobian variable {term.variable!r}")
            if not (0 <= term.force_index < len(force_dofs)):
                raise ValueError(f"local force_index out of range: {term.force_index}")
            if not (0 <= term.coordinate_index < len(coordinate_dofs)):
                raise ValueError(f"local coordinate_index out of range: {term.coordinate_index}")
            values = np.asarray(term.values, dtype=np.float64).reshape(-1)
            if values.shape[0] != t.size:
                raise ValueError(
                    "local residual Jacobian term values must have one value per time sample; "
                    f"got {values.shape[0]}, expected {t.size}"
                )
            terms.append(
                NonlinearJacobianTerm(
                    force_dofs[term.force_index],
                    term.variable,
                    coordinate_dofs[term.coordinate_index],
                    values,
                )
            )
        return tuple(terms)

    def residual_omega_derivative(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Global explicit ``dG/domega`` samples."""

        return self._global_residual_derivative(
            "local residual omega derivative",
            self.local_residual_omega_derivative,
            t,
            x,
            dx,
            ddx,
            omega,
            parameter,
        )

    def residual_parameter_derivative(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Global explicit ``dG/dparameter`` samples."""

        return self._global_residual_derivative(
            "local residual parameter derivative",
            self.local_residual_parameter_derivative,
            t,
            x,
            dx,
            ddx,
            omega,
            parameter,
        )

    def _global_residual_derivative(
        self,
        label: str,
        local_derivative_method: Any,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        force_dofs = _validate_dofs("residual_force_dofs", self.residual_force_dofs, self.n_dof)
        coordinate_dofs = _validate_dofs("residual_coordinate_dofs", self.residual_coordinate_dofs, self.n_dof)
        local_x, local_dx, local_ddx = _extract_local_states(x, dx, ddx, coordinate_dofs)
        local_derivative = np.asarray(
            local_derivative_method(t, local_x, local_dx, local_ddx, omega, parameter),
            dtype=np.float64,
        )
        expected_shape = (t.size, len(force_dofs))
        if local_derivative.shape != expected_shape:
            raise ValueError(f"{label} must have shape {expected_shape}, got {local_derivative.shape}")
        derivative = np.zeros((t.size, self.n_dof), dtype=np.float64)
        derivative[:, list(force_dofs)] = local_derivative
        return derivative


class AutodiffFreeFrequencySecondOrderTimeModel(FreeFrequencySecondOrderTimeModel):
    """Free-frequency model whose generalized residual derivatives use Torch autodiff."""

    @property
    def autodiff_variables(self) -> tuple[JacobianVariable, ...]:
        """Variables whose local residual Jacobian terms should be differentiated."""

        return ("x",)

    @property
    def autodiff_omega_dependent(self) -> bool:
        """Whether the generalized residual has explicit unknown-frequency dependence."""

        return False

    @property
    def autodiff_parameter_dependent(self) -> bool:
        """Whether the generalized residual has explicit continuation-parameter dependence."""

        return False

    @abstractmethod
    def local_residual_force_torch(
        self,
        t: Any,
        local_x: Any,
        local_dx: Any,
        local_ddx: Any,
        omega: Any,
        parameter: Any,
    ) -> Any:
        """Return local generalized residual samples as a Torch tensor."""

    def local_residual_force(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Evaluate local generalized residual through the Torch implementation."""

        import torch

        device = torch.device("cpu")
        with torch.no_grad():
            t_tensor = torch.as_tensor(t, dtype=torch.float64, device=device)
            x_tensor = torch.as_tensor(local_x, dtype=torch.float64, device=device)
            dx_tensor = torch.as_tensor(local_dx, dtype=torch.float64, device=device)
            ddx_tensor = torch.as_tensor(local_ddx, dtype=torch.float64, device=device)
            omega_tensor = torch.as_tensor(float(omega), dtype=torch.float64, device=device)
            parameter_tensor = torch.as_tensor(float(parameter), dtype=torch.float64, device=device)
            force = self.local_residual_force_torch(
                t_tensor,
                x_tensor,
                dx_tensor,
                ddx_tensor,
                omega_tensor,
                parameter_tensor,
            )
        return np.asarray(force.detach().cpu().numpy(), dtype=np.float64)

    def local_residual_jacobian_terms(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> tuple[LocalResidualJacobianTerm, ...]:
        """Autodiff models do not expose handwritten residual Jacobian terms."""

        raise NotImplementedError("use ContinuationFreeFrequencyAutodiffSolver")

    def local_residual_omega_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Autodiff models expose omega derivatives through the autodiff solver."""

        if not self.autodiff_omega_dependent:
            return np.zeros((t.size, len(self.residual_force_dofs)), dtype=np.float64)
        raise NotImplementedError("use ContinuationFreeFrequencyAutodiffSolver for omega-dependent autodiff residuals")

    def local_residual_parameter_derivative(
        self,
        t: NDArray[np.float64],
        local_x: NDArray[np.float64],
        local_dx: NDArray[np.float64],
        local_ddx: NDArray[np.float64],
        omega: float,
        parameter: float,
    ) -> NDArray[np.float64]:
        """Autodiff models expose parameter derivatives through the autodiff solver."""

        if not self.autodiff_parameter_dependent:
            return np.zeros((t.size, len(self.residual_force_dofs)), dtype=np.float64)
        raise NotImplementedError("use ContinuationFreeFrequencyAutodiffSolver for parameter-dependent autodiff residuals")
