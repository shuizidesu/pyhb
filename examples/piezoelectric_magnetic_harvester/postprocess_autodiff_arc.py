from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.piezoelectric_magnetic_harvester.autodiff_model import (
    PiezoelectricMagneticHarvesterAutodiffModel,
)
from examples.piezoelectric_magnetic_harvester.run_autodiff_arc import (
    DEFAULT_PLOT_DOFS,
    FREQUENCY_RESOLUTION,
    HARMONICS,
)
from pyhb import FloquetConfig, compute_mixed_order_floquet_autodiff
from pyhb.harmonics import generate_hb_items

DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc.npz"
DEFAULT_OUTPUT_FIG = Path(__file__).resolve().parent / "results" / "autodiff_arc.png"
DEFAULT_OUTPUT_RMS = Path(__file__).resolve().parent / "results" / "autodiff_arc_rms.npz"
DEFAULT_OUTPUT_FLOQUET = Path(__file__).resolve().parent / "results" / "autodiff_arc_floquet.npz"
DEFAULT_SAMPLE_COUNT = 2048
SECOND_ORDER_DOFS = (0, 1)
FIRST_ORDER_DOFS = (2,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess the piezoelectric magnetic harvester autodiff continuation result."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--rms-output", type=Path, default=DEFAULT_OUTPUT_RMS)
    parser.add_argument("--no-stability", action="store_true")
    parser.add_argument("--stability-output", type=Path, default=DEFAULT_OUTPUT_FLOQUET)
    parser.add_argument("--hsu-samples", type=int, default=512)
    parser.add_argument("--floquet-method", choices=("trapezoid", "exponential"), default="exponential")
    parser.add_argument("--stability-tolerance", type=float, default=1e-4)
    parser.add_argument("--torch-device", type=str, default=None)
    return parser.parse_args()


def compute_rms_history(
    coefficient_history: NDArray[np.float64],
    dofs: tuple[int, ...],
    sample_count: int,
) -> NDArray[np.float64]:
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    t = np.arange(sample_count, dtype=np.float64) * (period / sample_count)
    hb_item, _, _ = generate_hb_items(t, HARMONICS)
    selected_coefficients = coefficient_history[:, :, list(dofs)]
    response = np.einsum("to,sod->std", hb_item, selected_coefficients)
    return np.sqrt(np.mean(response * response, axis=1))


def save_plot(
    parameter_history: NDArray[np.float64],
    rms_history: NDArray[np.float64],
    dofs: tuple[int, ...],
    output: Path,
    stable_history: NDArray[np.bool_] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(dofs), 1, sharex=True, figsize=(6.4, 3.0 * len(dofs)))
    axes_array = np.atleast_1d(axes)
    for axis, column, dof in zip(axes_array, range(len(dofs)), dofs, strict=True):
        values = rms_history[:, column]
        if dof == 1:
            values = values * 1e3
            ylabel = "z2 RMS [mm]"
        elif dof == 2:
            ylabel = "u RMS [V]"
        else:
            ylabel = f"DOF {dof + 1} RMS"
        _plot_curve(axis, parameter_history, values, ylabel, stable_history)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes_array[-1].set_xlabel("omega")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {output}")


def compute_stability_history(
    parameter_history: NDArray[np.float64],
    coefficient_history: NDArray[np.float64],
    hsu_samples: int,
    floquet_method: str,
    stability_tolerance: float,
    torch_device: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.complex128]]:
    model = PiezoelectricMagneticHarvesterAutodiffModel()
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
    for index, (parameter, coefficients) in enumerate(
        zip(parameter_history, coefficient_history, strict=True),
        start=1,
    ):
        result = compute_mixed_order_floquet_autodiff(
            model,
            coefficients,
            float(parameter),
            HARMONICS,
            FREQUENCY_RESOLUTION,
            SECOND_ORDER_DOFS,
            FIRST_ORDER_DOFS,
            config,
            torch_device=torch_device,
        )
        status = "stable" if result.stable else "unstable"
        print(
            f"Floquet {index}/{total} done, omega={float(parameter):.10g}, rho={result.spectral_radius:.6e}, {status}"
        )
        spectral_radius.append(result.spectral_radius)
        stable.append(result.stable)
        multipliers.append(result.multipliers)
    return (
        np.asarray(spectral_radius, dtype=np.float64),
        np.asarray(stable, dtype=np.bool_),
        np.asarray(multipliers, dtype=np.complex128),
    )


def _plot_curve(
    ax,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    label: str,
    stable_history: NDArray[np.bool_] | None,
) -> None:
    if stable_history is None or stable_history.size != x.size:
        ax.plot(x, y)
        return
    (handle,) = ax.plot([], [], label=label)
    color = handle.get_color()
    handle.remove()
    start = 0
    for index in range(1, x.size + 1):
        if index == x.size or bool(stable_history[index]) != bool(stable_history[start]):
            segment = slice(start, index)
            ax.plot(
                x[segment],
                y[segment],
                color=color,
                linestyle="-" if bool(stable_history[start]) else "--",
                marker="." if index - start == 1 else None,
            )
            start = index


def run_from_args(args: argparse.Namespace) -> None:
    with np.load(args.input) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    dofs = tuple(int(dof) for dof in args.dofs)
    rms_history = compute_rms_history(coefficient_history, dofs, args.sample_count)
    spectral_radius = None
    stable_history = None
    multiplier_history = None
    if not getattr(args, "no_stability", False):
        spectral_radius, stable_history, multiplier_history = compute_stability_history(
            parameter_history,
            coefficient_history,
            getattr(args, "hsu_samples", 512),
            getattr(args, "floquet_method", "exponential"),
            getattr(args, "stability_tolerance", 1e-4),
            getattr(args, "torch_device", None),
        )
    save_plot(parameter_history, rms_history, dofs, args.output, stable_history)
    if args.rms_output is not None:
        args.rms_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.rms_output,
            omega=parameter_history,
            rms_history=rms_history,
            dofs=np.asarray(dofs, dtype=np.int64),
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
            omega=parameter_history,
            spectral_radius=spectral_radius,
            stable_flag=stable_history,
            multipliers=multiplier_history,
        )
        print(f"Saved stability table to {stability_output}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
