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
from examples.vanderpol.run_free_frequency import FREQUENCY_RESOLUTION, HARMONICS


DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "free_frequency.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "free_frequency.png"
DEFAULT_RMS_OUTPUT = Path(__file__).resolve().parent / "results" / "free_frequency_rms.npz"
DEFAULT_SAMPLE_COUNT = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess the van der Pol free-frequency result.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--rms-output", type=Path, default=DEFAULT_RMS_OUTPUT)
    return parser.parse_args()


def compute_rms_history(coefficient_history: NDArray[np.float64], sample_count: int) -> NDArray[np.float64]:
    period = 2.0 * np.pi / FREQUENCY_RESOLUTION
    t = np.arange(sample_count, dtype=np.float64) * (period / sample_count)
    hb_item, _, _ = generate_hb_items(t, HARMONICS)
    response = np.einsum("to,sod->sd", hb_item, coefficient_history)
    return np.sqrt(np.mean(response * response, axis=1))


def save_plot(
    parameter_history: NDArray[np.float64],
    omega_history: NDArray[np.float64],
    rms_history: NDArray[np.float64],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_amplitude, ax_frequency) = plt.subplots(2, 1, sharex=True)
    ax_amplitude.plot(parameter_history, rms_history)
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


def run_from_args(args: argparse.Namespace) -> None:
    with np.load(args.input) as data:
        parameter_history = np.asarray(data["parameter_history"], dtype=np.float64)
        omega_history = np.asarray(data["omega_history"], dtype=np.float64)
        coefficient_history = np.asarray(data["coefficient_history"], dtype=np.float64)
    rms_history = compute_rms_history(coefficient_history, args.sample_count)
    save_plot(parameter_history, omega_history, rms_history, args.output)
    if args.rms_output is not None:
        args.rms_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.rms_output,
            parameter=parameter_history,
            omega=omega_history,
            rms_history=rms_history,
        )
        print(f"Saved RMS table to {args.rms_output}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
