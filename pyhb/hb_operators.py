"""Generic harmonic-balance operators shared by models and solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad_vec

from .harmonics import generate_hb_items


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


@dataclass(frozen=True)
class FrequencyGrid:
    """Map physical harmonic values to FFT bin indices."""

    frequency_resolution: float = 1.0
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        frequency_resolution = float(self.frequency_resolution)
        tolerance = float(self.tolerance)
        if not np.isfinite(frequency_resolution) or frequency_resolution <= 0.0:
            raise ValueError(f"frequency_resolution must be a positive finite value, got {self.frequency_resolution!r}")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"tolerance must be a non-negative finite value, got {self.tolerance!r}")

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.frequency_resolution

    def indices_for(self, harmonics: ArrayLike) -> tuple[int, ...]:
        values = np.asarray(harmonics, dtype=np.float64).reshape(-1)
        indices: list[int] = []
        for value in values:
            if value < 0.0:
                raise ValueError(f"harmonic values must be non-negative, got {value}")
            raw_index = value / self.frequency_resolution
            rounded = int(round(raw_index))
            if not np.isclose(raw_index, rounded, atol=self.tolerance, rtol=0.0):
                raise ValueError(
                    f"harmonic {value} is not aligned with frequency_resolution "
                    f"{self.frequency_resolution}; got bin {raw_index}"
                )
            indices.append(rounded)
        return tuple(indices)

    def values_for(self, harmonics: ArrayLike) -> tuple[float, ...]:
        values = np.asarray(harmonics, dtype=np.float64).reshape(-1)
        self.indices_for(values)
        return tuple(float(v) for v in values)


@dataclass(frozen=True)
class HBContext:
    """Precomputed harmonic-balance metadata passed to model Jacobians."""

    harmonics: tuple[float, ...]
    harmonic_indices: tuple[int, ...]
    nonlinear_harmonics: tuple[float, ...]
    nonlinear_harmonic_indices: tuple[int, ...]
    frequency_resolution: float
    period: float
    sample_count: int
    order: int
    s3: NDArray[np.float64]
    s3_dx: NDArray[np.float64]
    s3_ddx: NDArray[np.float64]

    @classmethod
    def build(
        cls,
        harmonics: ArrayLike,
        nonlinear_harmonics: ArrayLike,
        sample_count: int,
        frequency_resolution: float = 1.0,
        tolerance: float = 1e-10,
        s3_method: str = "fast",
        s3_quadrature_samples: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> "HBContext":
        _emit_progress(
            progress_callback,
            f"Preparing HB context... frequency_resolution={frequency_resolution}, sample_count={sample_count}",
        )
        grid = FrequencyGrid(frequency_resolution, tolerance)
        hb = grid.values_for(harmonics)
        nlhb = grid.values_for(nonlinear_harmonics)
        hb_indices = grid.indices_for(hb)
        nlhb_indices = grid.indices_for(nlhb)
        _emit_progress(
            progress_callback,
            f"Precomputing S3... method={s3_method}, harmonics={len(hb)}, nonlinear_harmonics={len(nlhb)}",
        )
        s3 = compute_s3(
            hb,
            nlhb,
            hb_indices,
            nlhb_indices,
            period=grid.period,
            sample_count=int(sample_count),
            method=s3_method,
            quadrature_samples=s3_quadrature_samples,
        )
        order = 2 * len(hb) + 1
        coefficient_dt_map, coefficient_ddt_map = coefficient_derivative_maps(hb)
        s3_dx = _differentiate_s3(s3, coefficient_dt_map, order)
        s3_ddx = _differentiate_s3(s3, coefficient_ddt_map, order)
        _emit_progress(
            progress_callback,
            f"Precompute finished. order={order}, period={grid.period:.12g}, s3_shape={s3.shape}",
        )
        return cls(
            harmonics=hb,
            harmonic_indices=hb_indices,
            nonlinear_harmonics=nlhb,
            nonlinear_harmonic_indices=nlhb_indices,
            frequency_resolution=float(frequency_resolution),
            period=grid.period,
            sample_count=int(sample_count),
            order=order,
            s3=s3,
            s3_dx=s3_dx,
            s3_ddx=s3_ddx,
        )


def coefficient_derivative_maps(harmonics: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hb = np.asarray(harmonics, dtype=np.float64).reshape(-1)
    n_harmonics = hb.size
    order = 2 * n_harmonics + 1
    first = np.zeros((order, order), dtype=np.float64)
    second = np.zeros((order, order), dtype=np.float64)

    for index, harmonic in enumerate(hb):
        cos_index = 1 + index
        sin_index = 1 + n_harmonics + index
        first[sin_index, cos_index] = -harmonic
        first[cos_index, sin_index] = harmonic
        second[cos_index, cos_index] = -(harmonic**2)
        second[sin_index, sin_index] = -(harmonic**2)

    return first, second


def _differentiate_s3(
    s3: ArrayLike,
    derivative_map: ArrayLike,
    order: int,
) -> NDArray[np.float64]:
    s3_matrix = np.asarray(s3, dtype=np.float64)
    derivative = np.asarray(derivative_map, dtype=np.float64)
    if s3_matrix.ndim != 2 or s3_matrix.shape[0] != order * order:
        raise ValueError("s3 must have shape (order * order, nonlinear_order)")
    if derivative.shape != (order, order):
        raise ValueError(f"derivative_map must have shape {(order, order)}, got {derivative.shape}")

    nonlinear_order = s3_matrix.shape[1]
    tensor = s3_matrix.reshape((order, order, nonlinear_order), order="F")
    differentiated = np.einsum("abk,bc->ack", tensor, derivative, optimize=True)
    return np.asarray(differentiated, dtype=np.float64).reshape(
        (order * order, nonlinear_order),
        order="F",
    )


def harmonic_integral_matrices(
    harmonics: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Analytic equivalents of the MATLAB ``integral(...)/pi`` matrices."""

    hb = np.asarray(harmonics, dtype=np.float64).reshape(-1)
    n = hb.size
    order = 2 * n + 1
    mass_basis = np.zeros((order, order), dtype=np.float64)
    damping_basis = np.zeros_like(mass_basis)
    stiffness_basis = np.zeros_like(mass_basis)

    stiffness_basis[0, 0] = 2.0
    for idx, harmonic in enumerate(hb):
        cos_idx = 1 + idx
        sin_idx = 1 + n + idx
        stiffness_basis[cos_idx, cos_idx] = 1.0
        stiffness_basis[sin_idx, sin_idx] = 1.0
        mass_basis[cos_idx, cos_idx] = -(harmonic**2)
        mass_basis[sin_idx, sin_idx] = -(harmonic**2)
        damping_basis[cos_idx, sin_idx] = harmonic
        damping_basis[sin_idx, cos_idx] = -harmonic

    return mass_basis, damping_basis, stiffness_basis


def integrate_s3(
    harmonics: ArrayLike,
    nonlinear_harmonics: ArrayLike,
    period: float = 2.0 * np.pi,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute the flattened nonlinear Jacobian projection matrix ``S3``."""

    hb = np.asarray(harmonics, dtype=np.float64).reshape(-1)
    nlhb = np.asarray(nonlinear_harmonics, dtype=np.float64).reshape(-1)
    order = 2 * hb.size + 1

    def integrand(tau: float) -> NDArray[np.float64]:
        hb_item, _, _ = generate_hb_items([tau], hb)
        nlhb_item, _, _ = generate_hb_items([tau], nlhb)
        outer_flat = np.outer(hb_item[0], hb_item[0]).reshape(order * order, order="F")
        return outer_flat[:, None] * nlhb_item[0][None, :]

    value, _ = quad_vec(integrand, 0.0, period, epsabs=epsabs, epsrel=epsrel)
    return np.asarray(value, dtype=np.float64) * (2.0 / period)


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def choose_s3_sample_count(
    sample_count: int,
    harmonic_indices: ArrayLike,
    nonlinear_harmonic_indices: ArrayLike,
) -> int:
    """Choose a safe periodic grid size for fast S3 projection."""

    hb_idx = np.asarray(harmonic_indices, dtype=np.int64).reshape(-1)
    nlhb_idx = np.asarray(nonlinear_harmonic_indices, dtype=np.int64).reshape(-1)
    max_hb = int(np.max(hb_idx)) if hb_idx.size else 0
    max_nlhb = int(np.max(nlhb_idx)) if nlhb_idx.size else 0
    max_combo = 2 * max_hb + max_nlhb
    return _next_power_of_two(max(int(sample_count), 2 * max_combo + 1, 16))


def integrate_s3_fast(
    harmonics: ArrayLike,
    nonlinear_harmonics: ArrayLike,
    period: float,
    sample_count: int,
    harmonic_indices: ArrayLike,
    nonlinear_harmonic_indices: ArrayLike,
) -> NDArray[np.float64]:
    """Compute flattened S3 by exact periodic-grid projection for aligned harmonics."""

    n_samples = choose_s3_sample_count(sample_count, harmonic_indices, nonlinear_harmonic_indices)
    tau = np.arange(n_samples, dtype=np.float64) * (period / n_samples)
    hb_item, _, _ = generate_hb_items(tau, harmonics)
    order = hb_item.shape[1]
    products = (hb_item[:, :, None] * hb_item[:, None, :]).reshape(n_samples, order * order, order="F")
    product_fft = np.fft.rfft(products, axis=0) * (2.0 / n_samples)
    nonlinear_indices = np.asarray(nonlinear_harmonic_indices, dtype=np.int64).reshape(-1)
    if np.any(nonlinear_indices < 0) or np.any(nonlinear_indices >= product_fft.shape[0]):
        raise ValueError(
            "nonlinear harmonic FFT indices exceed the fast S3 projection grid; "
            f"maximum supported index is {product_fft.shape[0] - 1}"
        )
    return np.asarray(
        np.hstack(
            (
                product_fft[0:1, :].real.T,
                product_fft[nonlinear_indices, :].real.T,
                -product_fft[nonlinear_indices, :].imag.T,
            )
        ),
        dtype=np.float64,
    )


def compute_s3(
    harmonics: ArrayLike,
    nonlinear_harmonics: ArrayLike,
    harmonic_indices: ArrayLike,
    nonlinear_harmonic_indices: ArrayLike,
    *,
    period: float,
    sample_count: int,
    method: str = "fast",
    quadrature_samples: int | None = None,
) -> NDArray[np.float64]:
    """Compute S3 using the selected method."""

    if method == "fast":
        return integrate_s3_fast(
            harmonics,
            nonlinear_harmonics,
            period,
            sample_count if quadrature_samples is None else quadrature_samples,
            harmonic_indices,
            nonlinear_harmonic_indices,
        )
    if method == "quad":
        return integrate_s3(harmonics, nonlinear_harmonics, period=period)
    raise ValueError(f"unsupported s3_method {method!r}; expected 'fast' or 'quad'")
