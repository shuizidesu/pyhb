from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb import FloquetConfig, compute_floquet_autodiff
from pyhb.harmonics import generate_hb_items
from examples.bernoulli_beam.autodiff_model import BernoulliBeamAutodiffModel
from examples.bernoulli_beam.run_autodiff_arc import DEFAULT_PLOT_DOFS, FREQUENCY_RESOLUTION, HARMONICS


DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "autodiff_arc.png"
DEFAULT_SAMPLE_COUNT = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess the Bernoulli beam autodiff continuation result.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--rms-output", type=Path, default=None)
    parser.add_argument("--no-stability", action="store_true")
    parser.add_argument("--stability-output", type=Path, default=None)
    parser.add_argument("--hsu-samples", type=int, default=512)
    parser.add_argument("--floquet-method", choices=("auto", "explicit", "dominant"), default="auto")
    parser.add_argument("--n-multipliers", type=int, default=8)
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
    fig, ax = plt.subplots()
    for column, dof in enumerate(dofs):
        _plot_curve(ax, parameter_history, rms_history[:, column], f"DOF {dof + 1}", stable_history)
    ax.set_xlabel("omega")
    ax.set_ylabel("RMS amplitude")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {output}")


def compute_stability_history(
    parameter_history: NDArray[np.float64],
    coefficient_history: NDArray[np.float64],
    hsu_samples: int,
    floquet_method: str,
    n_multipliers: int,
    torch_device: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    model = BernoulliBeamAutodiffModel()
    config = FloquetConfig(
        hsu_samples=hsu_samples,
        method=floquet_method,
        n_multipliers=n_multipliers,
        progress_callback=print,
    )
    spectral_radius = []
    stable = []
    total = int(parameter_history.size)
    print(f"Computing Floquet stability... points={total}")
    for index, (parameter, coefficients) in enumerate(
        zip(parameter_history, coefficient_history, strict=True),
        start=1,
    ):
        print(f"Floquet {index}/{total}, omega={float(parameter):.10g}")
        result = compute_floquet_autodiff(
            model,
            coefficients,
            float(parameter),
            HARMONICS,
            FREQUENCY_RESOLUTION,
            config,
            torch_device=torch_device,
        )
        status = "stable" if result.stable else "unstable"
        print(f"Floquet {index}/{total} done, rho={result.spectral_radius:.6e}, {status}")
        spectral_radius.append(result.spectral_radius)
        stable.append(result.stable)
    return np.asarray(spectral_radius, dtype=np.float64), np.asarray(stable, dtype=np.bool_)


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
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    dofs = tuple(int(dof) for dof in args.dofs)
    rms_history = compute_rms_history(coefficient_history, dofs, args.sample_count)
    spectral_radius = None
    stable_history = None
    if not getattr(args, "no_stability", False):
        spectral_radius, stable_history = compute_stability_history(
            parameter_history,
            coefficient_history,
            getattr(args, "hsu_samples", 512),
            getattr(args, "floquet_method", "auto"),
            getattr(args, "n_multipliers", 8),
            getattr(args, "torch_device", None),
        )
    save_plot(parameter_history, rms_history, dofs, args.output, stable_history)
    if args.rms_output is not None:
        args.rms_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.rms_output, np.column_stack((parameter_history, rms_history)))
        print(f"Saved RMS table to {args.rms_output}")
    stability_output = getattr(args, "stability_output", None)
    if stability_output is not None and spectral_radius is not None and stable_history is not None:
        stability_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(stability_output, np.column_stack((parameter_history, spectral_radius, stable_history.astype(np.float64))))
        print(f"Saved stability table to {stability_output}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
