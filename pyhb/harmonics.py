"""Harmonic-balance basis utilities.

The basis order matches the MATLAB helper ``GenerateHBItem``:
``[1, cos(HB[0] tau), ..., cos(HB[-1] tau), sin(HB[0] tau), ...]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_hb_items(
    tau: ArrayLike,
    harmonics: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return harmonic basis, first derivative, and second derivative matrices."""

    tau_arr = np.asarray(tau, dtype=np.float64).reshape(-1)
    hb = np.asarray(harmonics, dtype=np.float64).reshape(-1)
    n_samples = tau_arr.size
    n_harmonics = hb.size
    order = 2 * n_harmonics + 1

    item = np.ones((n_samples, order), dtype=np.float64)
    item_dt = np.zeros_like(item)
    item_ddt = np.zeros_like(item)

    angles = tau_arr[:, None] * hb[None, :]
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)

    item[:, 1 : 1 + n_harmonics] = cos_terms
    item[:, 1 + n_harmonics :] = sin_terms

    item_dt[:, 1 : 1 + n_harmonics] = -hb[None, :] * sin_terms
    item_dt[:, 1 + n_harmonics :] = hb[None, :] * cos_terms

    hb_sq = hb[None, :] ** 2
    item_ddt[:, 1 : 1 + n_harmonics] = -hb_sq * cos_terms
    item_ddt[:, 1 + n_harmonics :] = -hb_sq * sin_terms

    return item, item_dt, item_ddt


def coefficient_matrix_from_fft(
    values: ArrayLike,
    harmonics: ArrayLike,
    sample_count: int | None = None,
    harmonic_indices: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Extract nonlinear-Jacobian Fourier coefficients for many terms.

    The input is shaped ``(samples, n_terms)`` and the output is shaped
    ``(2 * n_harmonics + 1, n_terms)``. The DC row is halved to match
    nonlinear Jacobian projection conventions.
    """

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2:
        raise ValueError("values must be shaped (samples, n_terms)")

    n = _validated_sample_count(sample_count, samples.shape[0])
    hb = _validated_fft_indices(harmonic_indices if harmonic_indices is not None else harmonics, n)
    fft_values = _real_sample_fft(samples, hb) * (2.0 / n)
    return np.asarray(
        np.vstack((fft_values[0:1, :].real / 2.0, fft_values[hb, :].real, -fft_values[hb, :].imag)),
        dtype=np.float64,
    )


def stack_fft_coefficients(
    sample_by_dof: ArrayLike,
    harmonics: ArrayLike,
    sample_count: int | None = None,
    harmonic_indices: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return stacked residual coefficients for an array shaped ``(samples, dof)``.

    This follows ``CalculateRes.m`` and the arc-length ``J_lambda`` assembly:
    the DC row is ``real(fft[0])`` after the ``2 / sampleFFT`` scaling, not
    ``real(fft[0]) / 2``. Nonlinear Jacobian coefficient matrices intentionally
    halve the DC term instead.
    """

    values = np.asarray(sample_by_dof, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("sample_by_dof must be a 2D array shaped (samples, dof)")

    n = _validated_sample_count(sample_count, values.shape[0])
    hb = _validated_fft_indices(harmonic_indices if harmonic_indices is not None else harmonics, n)
    fft_values = _real_sample_fft(values, hb) * (2.0 / n)
    coeff = np.vstack((fft_values[0:1, :].real, fft_values[hb, :].real, -fft_values[hb, :].imag))
    return np.asarray(coeff, dtype=np.float64).reshape(-1, order="F")


def flatten_coefficients(coefficients: ArrayLike) -> NDArray[np.float64]:
    """Flatten ``(basis, dof)`` coefficients with MATLAB column-major order."""

    return np.asarray(coefficients, dtype=np.float64).reshape(-1, order="F")


def unflatten_coefficients(vector: ArrayLike, order: int, dof: int) -> NDArray[np.float64]:
    """Inverse of :func:`flatten_coefficients`."""

    return np.asarray(vector, dtype=np.float64).reshape((order, dof), order="F")


def _validated_sample_count(sample_count: int | None, actual_count: int) -> int:
    count = int(actual_count if sample_count is None else sample_count)
    if count != actual_count:
        raise ValueError(f"sample_count must match the number of sample rows; got {count}, expected {actual_count}")
    if count < 1:
        raise ValueError("at least one time sample is required")
    return count


def _validated_fft_indices(indices: ArrayLike, fft_size: int) -> NDArray[np.int64]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values >= fft_size):
        raise ValueError(f"harmonic FFT indices must be in [0, {fft_size - 1}], got {tuple(values)}")
    return values


def _real_sample_fft(values: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.complex128]:
    sample_count = values.shape[0]
    if indices.size == 0 or int(np.max(indices)) <= sample_count // 2:
        return np.asarray(np.fft.rfft(values, axis=0), dtype=np.complex128)
    return np.asarray(np.fft.fft(values, axis=0), dtype=np.complex128)
