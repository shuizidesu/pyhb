from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhb.harmonics import generate_hb_items
from examples.four_dof_integer.run_arc import DEFAULT_PLOT_DOFS, FREQUENCY_RESOLUTION, HARMONICS


DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "arc.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "arc.png"
DEFAULT_SAMPLE_COUNT = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess the 4DOF integer-harmonic continuation result.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--dofs", type=int, nargs="+", default=DEFAULT_PLOT_DOFS)
    parser.add_argument("--rms-output", type=Path, default=None)
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
    fig, ax = plt.subplots()
    for column, dof in enumerate(dofs):
        ax.plot(parameter_history, rms_history[:, column], label=f"DOF {dof + 1}")
    ax.set_xlabel("omega")
    ax.set_ylabel("RMS amplitude")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
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
        np.save(args.rms_output, np.column_stack((parameter_history, rms_history)))
        print(f"Saved RMS table to {args.rms_output}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
