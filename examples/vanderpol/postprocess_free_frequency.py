from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.vanderpol.model import VanderpolModel
from pyhb import FloquetConfig, compute_free_frequency_floquet
from pyhb.harmonics import generate_hb_items

DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "free_frequency.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "free_frequency.png"
DEFAULT_RMS_OUTPUT = Path(__file__).resolve().parent / "results" / "free_frequency_rms.npz"
DEFAULT_STABILITY_OUTPUT = Path(__file__).resolve().parent / "results" / "free_frequency_floquet.npz"
DEFAULT_SAMPLE_COUNT = 2048
HARMONICS = tuple(float(value) for value in range(1, 51))
FREQUENCY_RESOLUTION = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess the van der Pol free-frequency result.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--rms-output", type=Path, default=DEFAULT_RMS_OUTPUT)
    parser.add_argument("--no-stability", action="store_true")
    parser.add_argument("--stability-output", type=Path, default=DEFAULT_STABILITY_OUTPUT)
    parser.add_argument("--hsu-samples", type=int, default=512)
    parser.add_argument("--floquet-method", choices=("trapezoid", "exponential"), default="trapezoid")
    parser.add_argument("--stability-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def compute_rms_history(coefficient_history: NDArray[np.float64], sample_count: int) -> NDArray[np.float64]:
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    t = np.arange(sample_count, dtype=np.float64) * (period / sample_count)
    hb_item, _, _ = generate_hb_items(t, HARMONICS)
    response = np.einsum("to,sod->std", hb_item, coefficient_history)
    return np.sqrt(np.mean(response * response, axis=1))


def compute_stability_history(
    parameter_history: NDArray[np.float64],
    omega_history: NDArray[np.float64],
    coefficient_history: NDArray[np.float64],
    hsu_samples: int,
    floquet_method: str,
    stability_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.complex128]]:
    model = VanderpolModel()
    config = FloquetConfig(
        hsu_samples=hsu_samples,
        method=floquet_method,
        stability_tolerance=stability_tolerance,
    )
    spectral_radius = []
    stable = []
    multipliers = []
    total = int(parameter_history.size)
    print(f"Computing Floquet stability... points={total}, method={floquet_method}, samples={hsu_samples}")
    for index, (parameter, omega, coefficients) in enumerate(
        zip(parameter_history, omega_history, coefficient_history, strict=True),
        start=1,
    ):
        result = compute_free_frequency_floquet(
            model,
            coefficients,
            float(omega),
            float(parameter),
            HARMONICS,
            FREQUENCY_RESOLUTION,
            config,
        )
        status = "stable" if result.stable else "unstable"
        print(
            f"Floquet {index}/{total} done, parameter={float(parameter):.10g}, "
            f"omega={float(omega):.10g}, rho={result.spectral_radius:.6e}, {status}"
        )
        spectral_radius.append(result.spectral_radius)
        stable.append(result.stable)
        multipliers.append(result.multipliers)
    return (
        np.asarray(spectral_radius, dtype=np.float64),
        np.asarray(stable, dtype=np.bool_),
        np.asarray(multipliers, dtype=np.complex128),
    )


def save_plot(
    parameter_history: NDArray[np.float64],
    omega_history: NDArray[np.float64],
    rms_history: NDArray[np.float64],
    output: Path,
    stable_history: NDArray[np.bool_] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_amplitude, ax_frequency) = plt.subplots(2, 1, sharex=True)
    _plot_curve(ax_amplitude, parameter_history, rms_history[:, 0], "RMS", stable_history)
    ax_amplitude.set_ylabel("RMS amplitude")
    ax_amplitude.grid(True, alpha=0.3)
    ax_frequency.plot(parameter_history, omega_history)
    ax_frequency.set_xlabel("epsilon")
    ax_frequency.set_ylabel("omega")
    ax_frequency.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {output}")


def _plot_curve(
    ax,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    label: str,
    stable_history: NDArray[np.bool_] | None,
) -> None:
    if stable_history is None or stable_history.size != x.size:
        ax.plot(x, y, label=label)
        return
    (handle,) = ax.plot([], [], label=label)
    color = handle.get_color()
    handle.remove()
    start = 0
    first = True
    for index in range(1, x.size + 1):
        if index == x.size or bool(stable_history[index]) != bool(stable_history[start]):
            segment = slice(start, index)
            ax.plot(
                x[segment],
                y[segment],
                color=color,
                linestyle="-" if bool(stable_history[start]) else "--",
                marker="." if index - start == 1 else None,
                label=label if first else None,
            )
            first = False
            start = index


def run_from_args(args: argparse.Namespace) -> None:
    with np.load(args.input) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        omega_history = np.asarray(data["omega_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    rms_history = compute_rms_history(coefficient_history, args.sample_count)
    spectral_radius = None
    stable_history = None
    multiplier_history = None
    if not getattr(args, "no_stability", False):
        spectral_radius, stable_history, multiplier_history = compute_stability_history(
            parameter_history,
            omega_history,
            coefficient_history,
            getattr(args, "hsu_samples", 512),
            getattr(args, "floquet_method", "trapezoid"),
            getattr(args, "stability_tolerance", 1e-4),
        )
    save_plot(parameter_history, omega_history, rms_history, args.output, stable_history)
    if args.rms_output is not None:
        args.rms_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.rms_output,
            parameter=parameter_history,
            omega=omega_history,
            rms_history=rms_history,
        )
        print(f"Saved RMS table to {args.rms_output}")
    stability_output = getattr(args, "stability_output", None)
    if (
        stability_output is not None
        and spectral_radius is not None
        and stable_history is not None
        and multiplier_history is not None
    ):
        stability_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            stability_output,
            parameter=parameter_history,
            omega=omega_history,
            spectral_radius=spectral_radius,
            stable_flag=stable_history,
            multipliers=multiplier_history,
        )
        print(f"Saved stability table to {stability_output}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
