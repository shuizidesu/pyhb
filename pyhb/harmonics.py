"""Harmonic-balance basis utilities.

The basis order matches the MATLAB helper ``GenerateHBItem``:
``[1, cos(HB[0] tau), ..., cos(HB[-1] tau), sin(HB[0] tau), ...]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_hb_items(tau: ArrayLike, harmonics: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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


def coefficient_vector_from_fft(
    values: ArrayLike,
    harmonics: ArrayLike,
    sample_count: int | None = None,
    harmonic_indices: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Extract MATLAB-style Fourier coefficients from samples.

    MATLAB code uses ``fft(values) * 2 / sampleFFT`` and then stacks
    ``[real(fft[0])/2, real(fft[h]), -imag(fft[h])]``.
    """

    samples = np.asarray(values)
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    n = int(sample_count if sample_count is not None else samples.size)
    hb = np.asarray(harmonic_indices if harmonic_indices is not None else harmonics, dtype=np.int64).reshape(-1)
    fft_values = np.fft.fft(samples) * (2.0 / n)
    return np.concatenate(
        (
            np.array([fft_values[0].real / 2.0], dtype=np.float64),
            fft_values[hb].real.astype(np.float64),
            (-fft_values[hb].imag).astype(np.float64),
        )
    )


def coefficient_matrix_from_fft(
    values: ArrayLike,
    harmonics: ArrayLike,
    sample_count: int | None = None,
    harmonic_indices: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Extract nonlinear-Jacobian Fourier coefficients for many terms.

    The input is shaped ``(samples, n_terms)`` and the output is shaped
    ``(2 * n_harmonics + 1, n_terms)``. The DC row follows
    :func:`coefficient_vector_from_fft` and is halved.
    """

    samples = np.asarray(values)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2:
        raise ValueError("values must be shaped (samples, n_terms)")

    n = int(sample_count if sample_count is not None else samples.shape[0])
    hb = np.asarray(harmonic_indices if harmonic_indices is not None else harmonics, dtype=np.int64).reshape(-1)
    fft_values = np.fft.fft(samples, axis=0) * (2.0 / n)
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
    ``real(fft[0]) / 2``. Nonlinear Jacobian helper coefficients use
    :func:`coefficient_vector_from_fft`, which intentionally halves the DC term.
    """

    values = np.asarray(sample_by_dof)
    if values.ndim != 2:
        raise ValueError("sample_by_dof must be a 2D array shaped (samples, dof)")

    n = int(sample_count if sample_count is not None else values.shape[0])
    hb = np.asarray(harmonic_indices if harmonic_indices is not None else harmonics, dtype=np.int64).reshape(-1)
    fft_values = np.fft.fft(values, axis=0) * (2.0 / n)
    coeff = np.vstack((fft_values[0:1, :].real, fft_values[hb, :].real, -fft_values[hb, :].imag))
    return np.asarray(coeff, dtype=np.float64).reshape(-1, order="F")


def flatten_coefficients(coefficients: ArrayLike) -> NDArray[np.float64]:
    """Flatten ``(basis, dof)`` coefficients with MATLAB column-major order."""

    return np.asarray(coefficients, dtype=np.float64).reshape(-1, order="F")


def unflatten_coefficients(vector: ArrayLike, order: int, dof: int) -> NDArray[np.float64]:
    """Inverse of :func:`flatten_coefficients`."""

    return np.asarray(vector, dtype=np.float64).reshape((order, dof), order="F")
