"""Newton and arc-length continuation for second-order HB models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import null_space

from .harmonics import coefficient_matrix_from_fft, flatten_coefficients, generate_hb_items, stack_fft_coefficients, unflatten_coefficients
from .hb_operators import HBContext, linear_jacobian
from .models import SecondOrderTimeModel


@dataclass(frozen=True)
class ContinuationConfig:
    sample_fft: int = 2**11
    harmonics: tuple[float, ...] = tuple(float(v) for v in range(1, 11))
    nonlinear_harmonics: tuple[float, ...] = tuple(float(v) for v in range(1, 21))
    frequency_resolution: float = 1.0
    frequency_tolerance: float = 1e-10
    strict_fft_grid: bool | None = None
    s3_method: str = "fast"
    s3_quadrature_samples: int | None = None
    seed: int = 0
    init_omega: float = 300.0
    max_epoch: int = 10
    res_tolerance: float = 1e-10
    delta_tolerance: float = 1e-10
    s_initial: float = 0.05
    s_max: float = 0.1
    s_min: float = 1e-6
    max_steps: int = 500
    shrink_limit: int = 20
    plot_dt: float = 0.01
    enable_ode_check: bool = False
    ode_periods: int = 400
    ode_max_step: float | None = None
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StepLog:
    step: int
    epoch: int
    max_residual: float
    max_delta: float
    omega: float
    arc_length: float
    converged: bool


@dataclass(frozen=True)
class OdeCheckResult:
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    hb_t: NDArray[np.float64]
    hb_x: NDArray[np.float64]


@dataclass(frozen=True)
class ContinuationResult:
    omega_list: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    omega: float
    coefficient_history: NDArray[np.float64]
    parameter_history: NDArray[np.float64]
    harmonics: NDArray[np.float64]
    nonlinear_harmonics: NDArray[np.float64]
    frequency_resolution: float
    period: float
    logs: list[StepLog] = field(default_factory=list)
    initial_log: StepLog | None = None
    ode_check: OdeCheckResult | None = None


@dataclass
class _PreparedProblem:
    model: SecondOrderTimeModel
    context: HBContext
    t: NDArray[np.float64]
    hb_item: NDArray[np.float64]
    hb_item_dt: NDArray[np.float64]
    hb_item_ddt: NDArray[np.float64]
    hb_item_plot: NDArray[np.float64]


class ContinuationSolver:
    """Generic single-parameter arc-length continuation solver."""

    def __init__(self, model: SecondOrderTimeModel, config: ContinuationConfig | None = None) -> None:
        self.model = model
        self.config = config or ContinuationConfig()
        self.prepared = self._prepare()

    def _prepare(self) -> _PreparedProblem:
        config = self.config
        context = HBContext.build(
            config.harmonics,
            config.nonlinear_harmonics,
            config.sample_fft,
            config.frequency_resolution,
            config.frequency_tolerance,
            config.s3_method,
            config.s3_quadrature_samples,
            config.progress_callback,
        )
        use_strict_grid = self._use_strict_fft_grid(context)
        self._emit_progress(
            f"Generating HB basis... period={context.period:.12g}, strict_fft_grid={use_strict_grid}, "
            f"samples={config.sample_fft}"
        )
        if use_strict_grid:
            t = np.arange(config.sample_fft, dtype=np.float64) * (context.period / config.sample_fft)
            t_plot = np.arange(0.0, context.period, config.plot_dt)
        else:
            t = np.linspace(0.0, context.period, config.sample_fft)
            t_plot = np.arange(0.0, context.period + 0.5 * config.plot_dt, config.plot_dt)
        hb_item, hb_item_dt, hb_item_ddt = generate_hb_items(t, context.harmonics)
        hb_item_plot, _, _ = generate_hb_items(t_plot, context.harmonics)
        return _PreparedProblem(
            self.model,
            context,
            t,
            hb_item,
            hb_item_dt,
            hb_item_ddt,
            hb_item_plot,
        )

    def _emit_progress(self, message: str) -> None:
        if self.config.progress_callback is not None:
            self.config.progress_callback(message)

    def _use_strict_fft_grid(self, context: HBContext) -> bool:
        if self.config.strict_fft_grid is not None:
            return bool(self.config.strict_fft_grid)
        integer_harmonics = all(float(value).is_integer() for value in context.harmonics)
        return not (np.isclose(context.frequency_resolution, 1.0) and integer_harmonics)

    def _evaluate_state(
        self,
        coeff_line: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        coeff = unflatten_coefficients(coeff_line, self.prepared.context.order, self.model.n_dof)
        x = self.prepared.hb_item @ coeff
        dx = self.prepared.hb_item_dt @ coeff
        ddx = self.prepared.hb_item_ddt @ coeff
        return coeff, x, dx, ddx

    def _linear_jacobian(self, parameter: float) -> NDArray[np.float64]:
        return linear_jacobian(
            self.model.mass_matrix(parameter),
            self.model.damping_matrix(parameter),
            self.model.stiffness_matrix(parameter),
            self.prepared.context.harmonics,
        )

    def _residual(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        model = self.model
        t = self.prepared.t
        samples = (
            model.forcing(t, parameter)
            - model.nonlinear_force(t, x, dx, ddx, parameter)
            - (model.mass_matrix(parameter) @ ddx.T).T
            - (model.damping_matrix(parameter) @ dx.T).T
            - (model.stiffness_matrix(parameter) @ x.T).T
        )
        return stack_fft_coefficients(
            samples,
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        )

    def _jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        return self._linear_jacobian(parameter) + self._nonlinear_jacobian(x, dx, ddx, parameter)

    def _nonlinear_jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        context = self.prepared.context
        order = context.order
        size = self.model.n_dof * order
        nl_j = np.zeros((size, size), dtype=np.float64)
        terms = tuple(self.model.nonlinear_jacobian_terms(self.prepared.t, x, dx, ddx, parameter))
        if not terms:
            return nl_j

        tensor_by_variable = {
            "x": context.s3_tensor_x,
            "dx": context.s3_tensor_dx,
            "ddx": context.s3_tensor_ddx,
        }

        force_dofs = np.empty(len(terms), dtype=np.int64)
        coordinate_dofs = np.empty(len(terms), dtype=np.int64)
        variables: list[str] = []
        values = np.empty((self.prepared.t.size, len(terms)), dtype=np.float64)

        for index, term in enumerate(terms):
            if term.variable not in tensor_by_variable:
                raise ValueError(f"unsupported nonlinear Jacobian variable {term.variable!r}")
            if not (0 <= term.force_dof < self.model.n_dof):
                raise ValueError(f"force_dof out of range: {term.force_dof}")
            if not (0 <= term.coordinate_dof < self.model.n_dof):
                raise ValueError(f"coordinate_dof out of range: {term.coordinate_dof}")
            term_values = np.asarray(term.values, dtype=np.float64).reshape(-1)
            if term_values.shape[0] != self.prepared.t.size:
                raise ValueError(
                    "nonlinear Jacobian term values must have one value per time sample; "
                    f"got {term_values.shape[0]}, expected {self.prepared.t.size}"
                )
            force_dofs[index] = term.force_dof
            coordinate_dofs[index] = term.coordinate_dof
            variables.append(term.variable)
            values[:, index] = term_values

        coeffs = coefficient_matrix_from_fft(
            values,
            context.nonlinear_harmonics,
            context.sample_count,
            context.nonlinear_harmonic_indices,
        )
        row_offsets = np.arange(order, dtype=np.int64)
        col_offsets = np.arange(order, dtype=np.int64)

        for variable, s_tensor in tensor_by_variable.items():
            term_indices = np.asarray([index for index, term_variable in enumerate(variables) if term_variable == variable], dtype=np.int64)
            if term_indices.size == 0:
                continue
            blocks = np.einsum("abk,kt->abt", s_tensor, coeffs[:, term_indices])
            row_indices = force_dofs[term_indices, None, None] * order + row_offsets[None, :, None]
            col_indices = coordinate_dofs[term_indices, None, None] * order + col_offsets[None, None, :]
            np.add.at(nl_j, (row_indices, col_indices), np.moveaxis(blocks, 2, 0))

        return nl_j

    def _parameter_jacobian(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
        ddx: NDArray[np.float64],
        parameter: float,
    ) -> NDArray[np.float64]:
        samples = self.model.parameter_derivative(self.prepared.t, x, dx, ddx, parameter)
        return stack_fft_coefficients(
            samples,
            self.prepared.context.harmonics,
            self.config.sample_fft,
            self.prepared.context.harmonic_indices,
        ).reshape(-1, 1)

    def _solve_initial(
        self,
        coeff_line: NDArray[np.float64],
        parameter: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], StepLog]:
        config = self.config
        epoch = 1
        residual_vector = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        delta = np.full(coeff_line.shape, np.inf, dtype=np.float64)
        coeff = x = dx = ddx = None
        jacobian = None

        while (
            epoch < config.max_epoch
            and np.max(np.abs(residual_vector)) >= config.res_tolerance
            and np.max(np.abs(delta)) >= config.delta_tolerance
        ):
            coeff, x, dx, ddx = self._evaluate_state(coeff_line)
            jacobian = self._jacobian(x, dx, ddx, parameter)
            residual_vector = self._residual(x, dx, ddx, parameter)
            delta = np.linalg.solve(jacobian, residual_vector)
            coeff_line = coeff_line + delta
            epoch += 1

        coeff, x, dx, ddx = self._evaluate_state(coeff_line)
        jacobian = self._jacobian(x, dx, ddx, parameter)
        log = StepLog(
            step=0,
            epoch=epoch,
            max_residual=float(np.max(np.abs(residual_vector))),
            max_delta=float(np.max(np.abs(delta))),
            omega=float(parameter),
            arc_length=0.0,
            converged=bool(
                np.max(np.abs(residual_vector)) < config.res_tolerance
                or np.max(np.abs(delta)) < config.delta_tolerance
            ),
        )
        return coeff_line, coeff, x, dx, jacobian, log

    def _initial_tangent(self, j: NDArray[np.float64], j_parameter: NDArray[np.float64]) -> NDArray[np.float64]:
        j_arc = np.hstack((-j, j_parameter))
        basis = null_space(j_arc)
        if basis.shape[1] == 0:
            raise np.linalg.LinAlgError("arc Jacobian has no numerical null-space")
        tangent = basis[:, 0]
        if tangent[-1] < 0.0:
            tangent = -tangent
        return tangent / np.linalg.norm(tangent)

    def _run_ode_check(self, coeff: NDArray[np.float64], parameter: float) -> OdeCheckResult:
        config = self.config
        t_hb = np.arange(0.0, config.ode_periods * self.prepared.context.period + 0.005, 0.01)
        hb_item, hb_item_dt, _ = generate_hb_items(t_hb, self.prepared.context.harmonics)
        hb_x = hb_item @ coeff
        hb_v = hb_item_dt @ coeff
        y0 = np.concatenate((hb_x[0, :], hb_v[0, :]))
        kwargs = {}
        if config.ode_max_step is not None:
            kwargs["max_step"] = config.ode_max_step
        sol = solve_ivp(
            lambda t, y: self.model.rhs(t, y, parameter),
            (0.0, config.ode_periods * self.prepared.context.period),
            y0,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
            **kwargs,
        )
        return OdeCheckResult(sol.t, sol.y.T, t_hb, hb_x)

    def _run_full(
        self,
        initial_coefficients: NDArray[np.float64] | None = None,
        initial_parameter: float | None = None,
    ) -> ContinuationResult:
        config = self.config
        order = self.prepared.context.order
        if initial_coefficients is None:
            rng = np.random.default_rng(config.seed)
            coeff_line = rng.random(self.model.n_dof * order, dtype=np.float64) / 100.0
        else:
            initial = np.asarray(initial_coefficients, dtype=np.float64)
            coeff_line = flatten_coefficients(initial) if initial.ndim == 2 else initial.reshape(-1)

        parameter = float(config.init_omega if initial_parameter is None else initial_parameter)
        coeff_line, coeff, x, dx, jacobian, initial_log = self._solve_initial(coeff_line, parameter)
        self._emit_progress(
            "Initial computation, "
            f"Epoch = {initial_log.epoch}, Res = {initial_log.max_residual:.6e}, "
            f"Delta = {initial_log.max_delta:.6e}, Omega = {initial_log.omega:.10g}"
        )
        _, _, _, ddx = self._evaluate_state(coeff_line)
        j_parameter = self._parameter_jacobian(x, dx, ddx, parameter)
        tangent = self._initial_tangent(jacobian, j_parameter)
        y0 = np.concatenate((coeff_line, np.array([parameter], dtype=np.float64)))

        amplitudes: list[NDArray[np.float64]] = []
        omega_list: list[float] = []
        coefficient_history: list[NDArray[np.float64]] = []
        logs: list[StepLog] = []
        arc_length_step = float(config.s_initial)
        shrink_count = 0

        for step in range(1, config.max_steps + 1):
            epoch = 1
            y = y0 + arc_length_step * tangent
            coeff_line = y[:-1].copy()
            parameter = float(y[-1])
            residual_vector = np.full(self.model.n_dof * order, np.inf, dtype=np.float64)
            delta = np.full(self.model.n_dof * order + 1, np.inf, dtype=np.float64)
            j_arc_v = None

            while (
                epoch < config.max_epoch
                and np.max(np.abs(residual_vector)) >= config.res_tolerance
                and np.max(np.abs(delta)) >= config.delta_tolerance
            ):
                coeff, x, dx, ddx = self._evaluate_state(coeff_line)
                jacobian = self._jacobian(x, dx, ddx, parameter)
                residual_vector = self._residual(x, dx, ddx, parameter)
                j_parameter = self._parameter_jacobian(x, dx, ddx, parameter)
                j_arc = np.hstack((-jacobian, j_parameter))
                r_arc = np.concatenate(
                    (residual_vector, np.array([(y - y0) @ tangent - arc_length_step], dtype=np.float64))
                )
                j_arc_v = np.vstack((j_arc, tangent[None, :]))
                delta = np.linalg.solve(j_arc_v, r_arc)
                y = y - delta
                coeff_line = y[:-1].copy()
                parameter = float(y[-1])
                epoch += 1

            max_res = float(np.max(np.abs(residual_vector)))
            max_delta = float(np.max(np.abs(delta)))
            converged = max_res <= config.res_tolerance or max_delta <= config.delta_tolerance
            step_log = StepLog(step, epoch, max_res, max_delta, float(parameter), float(arc_length_step), bool(converged))
            logs.append(step_log)
            status = "ok" if step_log.converged else "no convergence"
            self._emit_progress(
                f"Times = {step_log.step}, Epoch = {step_log.epoch}, Res = {step_log.max_residual:.6e}, "
                f"Delta = {step_log.max_delta:.6e}, Omega = {step_log.omega:.10g}, "
                f"s = {step_log.arc_length:.6g}, {status}"
            )

            if converged:
                y0 = y.copy()
                if j_arc_v is None:
                    raise RuntimeError("converged before assembling arc Jacobian")
                tangent_candidate = np.linalg.solve(
                    j_arc_v,
                    np.concatenate((np.zeros(jacobian.shape[0]), np.array([1.0]))),
                )
                tangent = tangent_candidate / np.linalg.norm(tangent_candidate)
                coeff, _, _, _ = self._evaluate_state(coeff_line)
                y_plot = self.prepared.hb_item_plot @ coeff
                amplitudes.append(rms_amplitude(y_plot))
                omega_list.append(float(parameter))
                coefficient_history.append(coeff.copy())
                shrink_count = 0
            else:
                shrink_count += 1

            if epoch >= config.max_epoch:
                arc_length_step = max(0.5 * arc_length_step, config.s_min)
            else:
                arc_length_step = min(2.0 * arc_length_step, config.s_max)

            if shrink_count >= config.shrink_limit:
                break

        final_coeff, _, _, _ = self._evaluate_state(y0[:-1])
        final_parameter = float(y0[-1])
        ode_check = self._run_ode_check(final_coeff, final_parameter) if config.enable_ode_check else None
        amplitudes_array = (
            np.asarray(amplitudes, dtype=np.float64)
            if amplitudes
            else np.empty((0, self.model.n_dof), dtype=np.float64)
        )
        coefficient_history_array = (
            np.asarray(coefficient_history, dtype=np.float64)
            if coefficient_history
            else np.empty((0, self.prepared.context.order, self.model.n_dof), dtype=np.float64)
        )
        parameter_history = np.asarray(omega_list, dtype=np.float64)
        return ContinuationResult(
            omega_list=parameter_history,
            amplitudes=amplitudes_array,
            coefficients=final_coeff,
            omega=final_parameter,
            coefficient_history=coefficient_history_array,
            parameter_history=parameter_history,
            harmonics=np.asarray(self.prepared.context.harmonics, dtype=np.float64),
            nonlinear_harmonics=np.asarray(self.prepared.context.nonlinear_harmonics, dtype=np.float64),
            frequency_resolution=float(self.prepared.context.frequency_resolution),
            period=float(self.prepared.context.period),
            logs=logs,
            initial_log=initial_log,
            ode_check=ode_check,
        )

    def run(
        self,
        initial_coefficients: NDArray[np.float64] | None = None,
        initial_parameter: float | None = None,
    ) -> ContinuationResult:
        """Run initial Newton solve followed by arc-length continuation."""

        return self._run_full(initial_coefficients, initial_parameter)


def rms_amplitude(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=0))
