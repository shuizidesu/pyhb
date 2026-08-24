# pyHB

pyHB provides generic harmonic-balance and arc-length continuation utilities for nonlinear dynamic response analysis.

The project is licensed under the BSD 3-Clause License. The current package
version is available as `pyhb.__version__`.

The reusable library code lives in `pyhb`. System-specific models and runnable scripts live in `examples`.

## Install

Install the core library dependencies with:

```powershell
pip install .
```

Optional extras are available for example plotting, Torch autodiff models, and local development:

```powershell
pip install .[examples]
pip install .[autodiff]
pip install .[dev]
```

The `dev` extra provides Ruff, Mypy, and distribution build tooling. The full
local regression additionally requires the `examples` and `autodiff` extras.

The core package depends only on NumPy and SciPy. Torch is optional and is loaded only when autodiff solvers or autodiff Floquet helpers are used.

## Package Layout

- `pyhb/` contains the publishable Python package.
- `pyhb.models` defines the public model interfaces and structured term dataclasses used by the solvers.
- `examples/` contains runnable reference systems. These models are part of the source release examples, not public library API.

The wheel built from this project installs only `pyhb/*.py` and package metadata. Source checkouts include the tracked examples so the reference systems can be run and inspected.

## Public API

The main solver and postprocessing entry points are:

- `ContinuationSolver` for full-system analytical nonlinear Jacobians.
- `ContinuationAutodiffSolver` for full-system Torch autodiff nonlinear derivatives.
- `ContinuationFreeFrequencySolver` for autonomous systems where the response frequency is unknown.
- `ContinuationFreeFrequencyAutodiffSolver` for Torch autodiff derivatives in free-frequency systems.
- `compute_floquet` for analytical-model Floquet stability postprocessing.
- `compute_floquet_autodiff` for Torch autodiff Floquet stability postprocessing.
- `compute_free_frequency_floquet` and `compute_free_frequency_floquet_autodiff` for free-frequency
  Floquet stability postprocessing.
- `compute_mixed_order_floquet_autodiff` for systems combining second- and
  first-order coordinates.

## Minimal Use

Implement a model by subclassing `SecondOrderTimeModel`, then pass explicit initial harmonic coefficients to a solver:

```python
import numpy as np

from pyhb import ContinuationConfig, ContinuationSolver

model = ...  # SecondOrderTimeModel implementation
config = ContinuationConfig(sample_fft=512, harmonics=(1.0,), max_steps=10)
initial_coefficients = np.zeros((2 * len(config.harmonics) + 1, model.n_dof))

result = ContinuationSolver(model, config).run(initial_coefficients=initial_coefficients)
print(result.parameter_history)
```

The Newton backend is selected in the solver config:

```python
# Analytical Jacobian model
config = ContinuationConfig(linear_solver="sparse")  # or "dense"

# Torch autodiff model
from pyhb import ContinuationAutodiffConfig

config = ContinuationAutodiffConfig(
    linear_solver="sparse",
    autodiff_jacobian_mode="dense",
)
```

`linear_solver` controls the final Newton system: `"sparse"` uses CSC matrices
and SuperLU, while `"dense"` keeps the structured linear operators sparse but
assembles the complete Newton Jacobian as a dense matrix and uses LAPACK LU.
`autodiff_jacobian_mode` controls only the nonlinear Jacobian projection after
Torch AD. The default combination is `linear_solver="sparse"` with
`autodiff_jacobian_mode="dense"`.

| Linear solver | Autodiff Jacobian | Intended use |
| --- | --- | --- |
| `sparse` | `dense` | Default; local nonlinearities and ordinary small-to-large sparse HB systems. |
| `sparse` | `sparse` | Many nonlinear-force DOFs with diagonal, banded, or otherwise sparse coordinate dependence. |
| `dense` | `dense` | Moderate HB dimension with a globally dense nonlinear and final Newton Jacobian. |
| `dense` | `sparse` | Unsupported: compact sparse AD projection would be expanded back to dense and lose its benefit. |

For Torch-based nonlinear derivatives, install the `autodiff` extra and use
`ContinuationAutodiffSolver` or `ContinuationFreeFrequencyAutodiffSolver`.

## Examples

Run scripts live under `examples/<system>/` and write results under each system's `results/` directory:

```powershell
python -m examples.machine_absorber.run_arc
python -m examples.machine_absorber.postprocess_arc
```

Example model files define system-specific data and nonlinear laws. Solver settings, output paths, and plotting choices belong in run or postprocess scripts.

## Citation

If pyHB is useful in your research, please cite:

Yuhong Jin, Qi Liu, Lei Hou, et al. *pyHB: an open-source
automatic-differentiation-enhanced semi-analytical solver for nonlinear
dynamics*, 2026, [**arXiv:2607.17577**](https://arxiv.org/abs/2607.17577).
