# pyHB

pyHB provides generic harmonic-balance and arc-length continuation utilities for nonlinear dynamic response analysis.

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
- `CondensedContinuationSolver` for localized nonlinearities with Schur condensation.
- `compute_floquet` for analytical-model Floquet stability postprocessing.
- `compute_floquet_autodiff` for Torch autodiff Floquet stability postprocessing.

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

For localized nonlinearities, use `CondensedContinuationSolver`. For Torch-based nonlinear derivatives, install the `autodiff` extra and use `ContinuationAutodiffSolver`.

## Examples

Run scripts live under `examples/<system>/` and write results under each system's `results/` directory:

```powershell
python -m examples.machine_absorber.run_arc
python -m examples.machine_absorber.postprocess_arc
```

Example model files define system-specific data and nonlinear laws. Solver settings, output paths, and plotting choices belong in run or postprocess scripts.
