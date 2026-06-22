# pyHB API Guide

pyHB provides reusable harmonic-balance tools for nonlinear second-order dynamic
systems. The installable library lives under `pyhb/`. The `examples/` directory
contains runnable reference systems that show how to define models, run
continuation, and postprocess periodic responses; those example models are not
part of the public library API.

The core package depends on NumPy and SciPy. Torch is optional: `import pyhb`
and `from pyhb import *` do not load Torch. Torch is imported only when an
autodiff solver or autodiff Floquet helper is explicitly accessed.

## Installation

Install the core package from a checkout:

```bash
pip install .
```

Optional extras:

```bash
pip install .[examples]   # plotting dependencies used by examples
pip install .[autodiff]   # Torch-based nonlinear derivatives
```

## Harmonic Coefficients

pyHB represents a periodic response with a harmonic basis ordered as

```text
[1, cos(h1*tau), ..., cos(hm*tau), sin(h1*tau), ..., sin(hm*tau)]
```

where `tau` is the normalized periodic time and `harmonics=(h1, ..., hm)`.
For `m` retained harmonics, the coefficient order is `2*m + 1`. Coefficient
matrices are shaped `(order, n_dof)` and flattened in column-major order so that
all coefficients for one physical degree of freedom stay contiguous.

The public helpers in `pyhb.harmonics` can be used when an application needs to
reconstruct or reshape coefficients:

- `generate_hb_items(tau, harmonics)` returns basis, first-derivative basis,
  and second-derivative basis matrices.
- `flatten_coefficients(coefficients)` flattens an `(order, n_dof)` matrix.
- `unflatten_coefficients(vector, order, dof)` restores the matrix layout.
- `stack_fft_coefficients(sample_by_dof, harmonics, ...)` projects sampled
  residual-like arrays into stacked HB coefficients.

## Frequency Grid

`FrequencyGrid` maps physical harmonic values to aligned FFT bin indices.

```python
from pyhb import FrequencyGrid

grid = FrequencyGrid(frequency_resolution=0.1, tolerance=1e-10)
indices = grid.indices_for((1.0, 1.2))
period = grid.period
```

Parameters:

- `frequency_resolution`: positive finite frequency spacing between FFT bins.
- `tolerance`: non-negative finite tolerance used when checking that each
  harmonic is aligned with the frequency grid.

Use `build_full_fft_nonlinear_harmonics(sample_count, frequency_resolution)` to
construct all positive FFT-bin harmonics through Nyquist for nonlinear force
projection. This is the default nonlinear harmonic grid used by the solvers
when `nonlinear_harmonics` is `None`.

## Model Interface

All analytical solvers consume a `SecondOrderTimeModel`. A model represents the
time-domain residual

```text
F(t, p) - N(t, x, dx, ddx, p) - M(p)*ddx - C(p)*dx - K(p)*x = 0
```

where `p` is the continuation parameter, usually the excitation frequency
`omega`.

Required model properties and methods:

- `n_dof`: number of physical degrees of freedom.
- `nonlinear_force_dofs`: global DOFs where local nonlinear forces are applied.
- `nonlinear_coordinate_dofs`: global DOFs whose states enter the local
  nonlinear law.
- `linear_operator_terms()`: returns `LinearOperatorTerm` objects.
- `forcing_terms(t)`: returns `ForcingTerm` objects sampled at the solver time
  grid.
- `local_nonlinear_force(t, local_x, local_dx, local_ddx, parameter)`: returns
  nonlinear force samples shaped `(samples, len(nonlinear_force_dofs))`.
- `local_nonlinear_jacobian_terms(...)`: returns local analytical nonlinear
  Jacobian terms.

Optional method:

- `local_nonlinear_parameter_derivative(...)`: returns explicit `dN/dparameter`
  samples. The default is zero.

The base class scatters local nonlinear force and Jacobian data to global DOFs
for the full-system solvers. The condensed solver also uses the local DOF
metadata directly.

### LinearOperatorTerm

```python
from pyhb import LinearOperatorTerm

LinearOperatorTerm(matrix=M, basis_type="ddx", parameter_power=0.0)
LinearOperatorTerm(matrix=C, basis_type="dx", parameter_power=1.0)
LinearOperatorTerm(matrix=K, basis_type="x", parameter_power=0.0)
```

Fields:

- `matrix`: dense NumPy array or SciPy sparse matrix.
- `basis_type`: one of `"ddx"`, `"dx"`, or `"x"`, selecting the HB derivative
  basis used for this contribution.
- `parameter_power`: exponent applied to the continuation parameter before the
  term is assembled. For example, `parameter_power=2.0` multiplies the matrix by
  `parameter**2`.

### ForcingTerm

```python
from pyhb import ForcingTerm

ForcingTerm(samples=force_samples, parameter_power=0.0)
```

Fields:

- `samples`: sampled external force array shaped `(samples, n_dof)`.
- `parameter_power`: exponent applied to the continuation parameter before the
  force contribution is assembled.

### LocalNonlinearJacobianTerm

```python
from pyhb import LocalNonlinearJacobianTerm

LocalNonlinearJacobianTerm(
    force_index=0,
    variable="x",
    coordinate_index=0,
    values=dndx_samples,
)
```

Fields:

- `force_index`: index inside `nonlinear_force_dofs`.
- `variable`: one of `"x"`, `"dx"`, or `"ddx"`.
- `coordinate_index`: index inside `nonlinear_coordinate_dofs`.
- `values`: one sampled scalar derivative per solver time sample.

## Autodiff Models

`AutodiffSecondOrderTimeModel` is a `SecondOrderTimeModel` variant whose local
nonlinear force is implemented with Torch and whose nonlinear derivatives are
generated by Torch autodiff.

Additional API:

- `local_nonlinear_force_torch(t, local_x, local_dx, local_ddx, parameter)`:
  returns a Torch tensor shaped `(samples, force_count)`.
- `autodiff_variables`: tuple of variables to differentiate, defaulting to
  `("x",)`.
- `autodiff_parameter_dependent`: set to `True` when the nonlinear force has
  explicit continuation-parameter dependence.

The NumPy `local_nonlinear_force(...)` method is provided by the base autodiff
class and evaluates the Torch implementation on CPU without gradient tracking.

## Full Analytical Continuation

Use `ContinuationSolver` for full-system harmonic-balance continuation when the
model provides analytical local nonlinear Jacobian terms.

```python
import numpy as np
from pyhb import ContinuationConfig, ContinuationSolver

config = ContinuationConfig(
    sample_fft=2048,
    harmonics=tuple(float(v) for v in range(1, 11)),
    frequency_resolution=1.0,
    init_omega=300.0,
)

solver = ContinuationSolver(model, config)
initial = np.zeros((2 * len(config.harmonics) + 1, model.n_dof))
result = solver.run(initial_coefficients=initial)
```

`ContinuationConfig` fields:

- `sample_fft`: number of strict FFT time samples.
- `harmonics`: retained response harmonics.
- `nonlinear_harmonics`: nonlinear projection harmonics. If `None`, all
  positive FFT bins through Nyquist are used.
- `frequency_resolution` and `frequency_tolerance`: define harmonic-to-FFT-bin
  alignment.
- `s3_method`: `"fast"` for periodic-grid projection or `"quad"` for numerical
  quadrature.
- `s3_quadrature_samples`: optional sample override for S3 construction.
- `init_omega`: initial continuation parameter for the preliminary Newton solve.
- `max_epoch`, `res_tolerance`, `delta_tolerance`: Newton iteration controls.
- `s_initial`, `s_max`, `s_min`: arc-length step controls.
- `q_scale`, `omega_scale`: weighted arc-length scales for coefficient and
  parameter components.
- `max_parameter_step` and `parameter_step_safety`: optional accepted-point
  parameter-step guard.
- `max_steps`, `shrink_limit`, `residual_floor`: continuation and diagnostic
  controls.
- `progress_callback`: optional callback receiving progress strings. Library
  code is silent when this is `None`.

`ContinuationSolver.run(initial_coefficients)` requires explicit initial
coefficients shaped `(order, n_dof)` or as an equivalent flattened vector.

`ContinuationResult` contains:

- `coefficients`: final accepted flattened coefficient vector.
- `omega`: final continuation parameter.
- `coefficient_history`: accepted coefficient history.
- `parameter_history`: accepted parameter history.
- `harmonics`, `nonlinear_harmonics`, `frequency_resolution`, `period`.
- `logs` and `initial_log`: Newton and continuation diagnostics.

## Torch Autodiff Continuation

Use `ContinuationAutodiffSolver` when the nonlinear force is easier to define
with Torch than with analytical Jacobian samples.

```python
from pyhb import ContinuationAutodiffConfig, ContinuationAutodiffSolver

config = ContinuationAutodiffConfig(
    harmonics=(1.0, 2.0, 3.0),
    torch_device=None,
)

result = ContinuationAutodiffSolver(model, config).run(initial_coefficients)
```

`ContinuationAutodiffConfig` extends `ContinuationConfig` with:

- `torch_device`: optional Torch device selector such as `"cpu"` or `"cuda"`.
  `None` lets Torch use CPU.

The continuation loop, residual definition, linear assembly, S3 tensors, and
result object match the full analytical solver. Only the nonlinear derivative
generation path is replaced by Torch autodiff.

## Condensed Continuation

Use `CondensedContinuationSolver` for localized nonlinearities where the
linear-only DOFs can be eliminated by Schur condensation.

```python
from pyhb import CondensedContinuationConfig, CondensedContinuationSolver

config = CondensedContinuationConfig(
    sample_fft=32768,
    harmonics=(1.0, 1.2),
    frequency_resolution=0.1,
)

result = CondensedContinuationSolver(model, config).run(initial_coefficients)
```

The model interface is still `SecondOrderTimeModel`, but the DOF metadata is
especially important:

- `nonlinear_force_dofs` selects where local nonlinear forces enter.
- `nonlinear_coordinate_dofs` selects the coordinates that drive the nonlinear
  law.
- The condensed nonlinear state is formed from the ordered union of those DOFs.

`CondensedContinuationConfig` has the same conceptual controls as
`ContinuationConfig`, with defaults tuned for condensed localized problems.

`CondensedContinuationResult` contains:

- `parameter_history`: accepted continuation parameters.
- `coefficient_history`: recovered full coefficient history.
- `nonlinear_coefficient_history`: condensed nonlinear-DOF coefficient history.
- `logs` and `initial_log`: diagnostics.

## Floquet Stability

Floquet stability is postprocessing for one accepted HB solution point. It does
not change continuation results.

```python
from pyhb import FloquetConfig, compute_floquet

floquet = compute_floquet(
    model,
    coefficients=result.coefficient_history[-1],
    parameter=result.parameter_history[-1],
    harmonics=result.harmonics,
    frequency_resolution=result.frequency_resolution,
    config=FloquetConfig(method="trapezoid", hsu_samples=512),
)
```

`FloquetConfig` fields:

- `hsu_samples`: time samples used for the periodic linearized system.
- `method`: `"trapezoid"` for the Hsu/trapezoid monodromy method or
  `"exponential"` for a piecewise matrix exponential reference method.
- `stability_tolerance`: tolerance used when classifying near-unit multipliers.
- `progress_callback`: optional progress callback.

`FloquetResult` contains:

- `parameter`, `multipliers`, `spectral_radius`, `stable`.
- `method`, `period`, `hsu_samples`.

For autodiff models, use `compute_floquet_autodiff(...)` with the same public
configuration object. Torch is imported only when this helper is explicitly
used.

## Examples

Runnable reference systems live under `examples/<system>/`. Typical scripts are:

- `run_arc.py`: full analytical continuation.
- `run_autodiff_arc.py`: Torch autodiff continuation.
- `run_condensed_arc.py`: condensed continuation when available.
- `postprocess_arc.py`, `postprocess_autodiff_arc.py`,
  `postprocess_condensed_arc.py`: response reconstruction, plots, and optional
  stability tables.

Example run scripts save minimal `.npz` payloads with `parameter_history` and
`coefficient_history`. Postprocess scripts reconstruct time responses from the
same harmonic and sampling constants used by the corresponding run script.

The current source examples include aero-engine, Bernoulli beam, bilinear
hysteretic, and machine-absorber systems.

