from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.piezoelectric_magnetic_harvester.run_arc import (
    DEFAULT_PLOT_DOFS,
    FREQUENCY_RESOLUTION,
    HARMONICS,
)
from pyhb.harmonics import generate_hb_items

DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_OUTPUT_FIG = Path(__file__).resolve().parent / "results" / "arc.png"
DEFAULT_OUTPUT_RMS = Path(__file__).resolve().parent / "results" / "arc_rms.npz"
DEFAULT_SAMPLE_COUNT = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess the piezoelectric magnetic harvester analytical continuation result."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--rms-output", type=Path, default=DEFAULT_OUTPUT_RMS)
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
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(dofs), 1, sharex=True, figsize=(6.4, 3.0 * len(dofs)))
    axes_array = np.atleast_1d(axes)
    for axis, column, dof in zip(axes_array, range(len(dofs)), dofs, strict=True):
        values = rms_history[:, column]
        if dof in (0, 1):
            values = values * 1e3
            ylabel = f"DOF {dof + 1} RMS [mm]"
        elif dof == 2:
            ylabel = "u RMS [V]"
        else:
            ylabel = f"DOF {dof + 1} RMS"
        axis.plot(parameter_history, values)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes_array[-1].set_xlabel("omega")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {output}")


def run_from_args(args: argparse.Namespace) -> None:
    with np.load(args.input) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    dofs = tuple(int(dof) for dof in args.dofs)
    rms_history = compute_rms_history(coefficient_history, dofs, args.sample_count)
    save_plot(parameter_history, rms_history, dofs, args.output)
    if args.rms_output is not None:
        args.rms_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.rms_output,
            omega=parameter_history,
            rms_history=rms_history,
            dofs=np.asarray(dofs, dtype=np.int64),
        )
        print(f"Saved RMS table to {args.rms_output}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
