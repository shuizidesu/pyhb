# pyHB API Guide

pyHB provides harmonic-balance tools for nonlinear second-order dynamic systems.
The reusable package is `pyhb/`. The `examples/` directory contains runnable
reference systems and is useful source material, but those system models are not
part of the stable library API.

The core package depends on NumPy and SciPy. Torch is optional: `import pyhb`
and `from pyhb import *` do not load Torch. Torch is imported only when an
autodiff continuation solver or autodiff Floquet helper is explicitly used.

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

## Module Map

Most users start from objects exported by `pyhb.__init__`:

- `SecondOrderTimeModel` and `AutodiffSecondOrderTimeModel` define model
  interfaces.
- `ContinuationSolver`, `ContinuationAutodiffSolver`, and
  `CondensedContinuationSolver` run harmonic-balance continuation.
- `compute_floquet` and `compute_floquet_autodiff` compute Floquet multipliers
  for accepted HB solutions.
- `FrequencyGrid` and `build_full_fft_nonlinear_harmonics` expose the frequency
  grid tools that users commonly need when configuring solvers.

Lower-level modules provide the assembly building blocks used by the solvers:

- `pyhb.models` defines structured linear terms, forcing terms, local nonlinear
  Jacobian terms, DOF validation, and local-to-global nonlinear scattering.
- `pyhb.harmonics` defines the HB basis order and FFT projection helpers.
- `pyhb.hb_operators` builds reusable HB metadata: frequency grids, derivative
  maps, linear projection matrices, and the S3 tensors used for nonlinear
  Jacobian projection.
- `pyhb.continuation` assembles the full analytical residual and Newton
  correction matrix, then runs weighted arc-length continuation.
- `pyhb.continuation_autodiff` reuses the full continuation loop but obtains
  nonlinear derivatives from Torch autodiff.
- `pyhb.condensed_continuation` eliminates linear-only DOFs by Schur
  condensation and continues the reduced nonlinear coordinates.
- `pyhb.floquet` reconstructs one HB solution, samples analytical nonlinear
  Jacobians in time, and computes monodromy multipliers.
- `pyhb.floquet_autodiff` uses Torch autodiff to build the same sampled
  nonlinear Jacobians before calling the shared Floquet backend.

The typical data flow is:

```text
model samples
  -> HB/FFT projection
  -> precomputed frequency grid, basis blocks, and S3 tensors
  -> residual and Newton correction matrix
  -> continuation history
  -> optional Floquet postprocessing
```

## Harmonic Coefficients

pyHB represents a periodic response with the basis

```text
[1, cos(h1*tau), ..., cos(hm*tau), sin(h1*tau), ..., sin(hm*tau)]
```

where `tau` is periodic time and `harmonics=(h1, ..., hm)`. With `m` retained
harmonics, the coefficient order is `2*m + 1`.

Coefficient matrices are shaped `(order, n_dof)`. Flattened vectors use
column-major order, so all coefficients for one physical degree of freedom are
contiguous.

Useful helpers in `pyhb.harmonics`:

- `generate_hb_items(tau, harmonics)` returns the basis, first derivative
  basis, and second derivative basis matrices.
- `flatten_coefficients(coefficients)` flattens an `(order, n_dof)` matrix.
- `unflatten_coefficients(vector, order, dof)` restores the matrix layout.
- `stack_fft_coefficients(sample_by_dof, harmonics, ...)` projects sampled
  residual-like arrays into stacked HB coefficients.
- `coefficient_matrix_from_fft(values, harmonics, ...)` projects sampled
  nonlinear Jacobian entries into a coefficient matrix used with S3 tensors.

The residual projection and nonlinear-Jacobian projection have different DC
conventions internally. Users normally do not need to manage this manually;
the solvers call the appropriate helper for each quantity.

## HB Operators

`pyhb.hb_operators` contains the shared harmonic-balance metadata used by all
continuation solvers.

### FrequencyGrid

`FrequencyGrid` maps physical harmonic values to FFT bin indices.

```python
from pyhb import FrequencyGrid

grid = FrequencyGrid(frequency_resolution=0.1, tolerance=1e-10)
indices = grid.indices_for((1.0, 1.2))
period = grid.period
```

Parameters:

- `frequency_resolution`: positive finite frequency spacing between FFT bins.
- `tolerance`: non-negative finite tolerance used when checking that each
  harmonic is aligned with the grid.

`period` is `2*pi/frequency_resolution`.

`build_full_fft_nonlinear_harmonics(sample_count, frequency_resolution)` returns
all positive FFT-bin harmonics through Nyquist. This is the default nonlinear
projection grid when a solver config leaves `nonlinear_harmonics=None`.

### HBContext

`HBContext` is built during solver initialization. It is not usually constructed
directly by end users, but it explains what is precomputed before Newton
iterations begin:

- validated response harmonics and their FFT bin indices;
- validated nonlinear projection harmonics and their FFT bin indices;
- the fundamental period and sample count;
- `order = 2*len(harmonics) + 1`;
- the raw S3 matrix;
- `s3_tensor_x`, `s3_tensor_dx`, and `s3_tensor_ddx`.

The full and condensed solvers keep this context in their prepared problem
objects. The autodiff continuation solver uses the same context; only the source
of nonlinear derivative samples changes.

### Linear Projection Matrices

`harmonic_integral_matrices(harmonics)` returns three HB projection matrices:

- mass-like matrix for `"ddx"` terms;
- damping-like matrix for `"dx"` terms;
- stiffness-like matrix for `"x"` terms.

The continuation solvers combine these matrices with each model
`LinearOperatorTerm` using Kronecker products. The result is a frequency-powered
linear block in coefficient space.

`coefficient_derivative_maps(harmonics)` returns two coefficient maps. They
convert a coefficient vector into the corresponding first- and second-derivative
basis contribution. These maps are also used to convert the base S3 tensor into
the derivative tensors for `dx` and `ddx` nonlinear Jacobian terms.

### S3 Construction

The S3 tensor is the common projection tensor for nonlinear Jacobians. It couples
one residual test basis function, one response basis function, and one nonlinear
Jacobian Fourier basis function.

`compute_s3(...)` selects the implementation:

- `integrate_s3_fast(...)` uses an aligned periodic FFT grid and is the default
  `s3_method="fast"` path.
- `integrate_s3(...)` uses numerical quadrature and is selected by
  `s3_method="quad"`.

After S3 is computed, `s3_to_tensor(...)` reshapes it into
`(order, order, nonlinear_order)`. `HBContext.build(...)` then constructs:

- `s3_tensor_x`: used for derivatives of nonlinear force with respect to `x`;
- `s3_tensor_dx`: used for derivatives with respect to `dx`;
- `s3_tensor_ddx`: used for derivatives with respect to `ddx`.

These tensors are precomputed once for the configured harmonic grid.

## Model Interface

All analytical solvers consume a `SecondOrderTimeModel`. A model represents the
time-domain residual

```text
F(t, p) - N(t, x, dx, ddx, p) - M(p)*ddx - C(p)*dx - K(p)*x = 0
```

where `p` is the continuation parameter. In forced-response solvers this is
usually the excitation frequency `omega`.

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
  Jacobian samples.

Optional method:

- `local_nonlinear_parameter_derivative(...)`: returns explicit
  `dN/dparameter` samples. The default is zero.

The base class scatters local nonlinear force and Jacobian data to global DOFs
for full-system solvers. The condensed solver also uses the local DOF metadata
directly.

### LinearOperatorTerm

```python
from pyhb import LinearOperatorTerm

LinearOperatorTerm(matrix=M, basis_type="ddx", omega_power=0.0)
LinearOperatorTerm(matrix=C, basis_type="dx", omega_power=1.0)
LinearOperatorTerm(matrix=K, basis_type="x", omega_power=0.0)
```

Fields:

- `matrix`: dense NumPy array or SciPy sparse matrix shaped `(n_dof, n_dof)`.
- `basis_type`: one of `"ddx"`, `"dx"`, or `"x"`.
- `omega_power`: exponent applied to the response frequency before the
  term is assembled.

### ForcingTerm

```python
from pyhb import ForcingTerm

ForcingTerm(samples=force_samples, omega_power=0.0)
```

Fields:

- `samples`: sampled external force array shaped `(samples, n_dof)`.
- `omega_power`: exponent applied to the response frequency before the
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

Use `ContinuationSolver` for full-system HB continuation when the model provides
analytical local nonlinear Jacobian terms.

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
- `frequency_resolution` and `frequency_tolerance`: harmonic-to-FFT-bin
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
coefficients shaped `(order, n_dof)` or an equivalent flattened vector.

`ContinuationResult` contains:

- `coefficients`: final accepted flattened coefficient vector.
- `omega`: final continuation parameter.
- `coefficient_history`: accepted coefficient history.
- `parameter_history`: accepted parameter history.
- `harmonics`, `nonlinear_harmonics`, `frequency_resolution`, `period`.
- `logs` and `initial_log`: Newton and continuation diagnostics.

The full solver assembles:

- projected residual coefficients from forcing, nonlinear force, and linear
  terms;
- a linear correction block from precomputed structured operator blocks;
- a nonlinear correction block from model Jacobian samples, FFT coefficients,
  and S3 tensors;
- a parameter column from forcing, nonlinear parameter derivative, and linear
  parameter derivative terms.

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
  `None` uses CUDA when available, otherwise CPU.

The continuation loop, residual definition, linear assembly, S3 tensors, and
result object match the full analytical solver. Only the nonlinear derivative
generation path is replaced by Torch autodiff.

## Free-Frequency Continuation

Use `FreeFrequencyContinuationSolver` for autonomous systems where the response
frequency is unknown and the continuation parameter is a separate scalar, such
as van der Pol limit-cycle continuation.

Free-frequency models implement `FreeFrequencySecondOrderTimeModel` and write a
local generalized residual contribution `G`:

```text
R = F0(t, omega) - G(t, x, dx, ddx, omega, parameter) - L0(omega) q
```

The structured `LinearOperatorTerm` and `ForcingTerm` APIs are reused, but their
`omega_power` is always interpreted as a power of the unknown response
frequency. Parameter-dependent nonlinear forces, self-excitation, or other
localized corrections should be included in `G`.

Required free-frequency model metadata:

- `residual_force_dofs`: global equation rows where `G` is applied.
- `residual_coordinate_dofs`: global coordinates whose `x/dx/ddx` enter `G`.
- `local_residual_force(...)`: local samples of `G`.
- `local_residual_jacobian_terms(...)`: local samples of `dG/dx`,
  `dG/ddx`, and `dG/dddx`.

Optional derivatives default to zero:

- `local_residual_omega_derivative(...)`
- `local_residual_parameter_derivative(...)`

`FreeFrequencyContinuationConfig` extends `ContinuationConfig` with:

- `init_parameter`: initial value of the true continuation parameter.
- `parameter_scale`: weighted arc-length scale for the true continuation
  parameter.
- `constraint`: a `HarmonicCoefficientConstraint` fixing one HB coefficient to
  remove phase ambiguity.
- `constraint_tolerance`: convergence tolerance for that constraint.

The preliminary Newton solve uses unknowns `[q, omega]` at fixed `parameter`.
Arc-length continuation uses `[q, omega, parameter]`. The constraint row is
assembled by the solver and contains a single `1` at the constrained coefficient
column.

`FreeFrequencyContinuationResult` stores:

- `parameter_history`: accepted true continuation parameter values.
- `omega_history`: accepted response frequencies.
- `coefficient_history`: accepted HB coefficient history.

`FreeFrequencyContinuationAutodiffSolver` uses
`AutodiffFreeFrequencySecondOrderTimeModel`. The model supplies
`local_residual_force_torch(...)`, and Torch autodiff builds the local
derivatives controlled by `autodiff_variables`, `autodiff_omega_dependent`, and
`autodiff_parameter_dependent`.

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

### Analytical and Autodiff Floquet

`compute_floquet(...)` and `compute_floquet_autodiff(...)` share the same
Floquet backend after nonlinear Jacobian samples have been prepared.

`compute_floquet(...)`:

- reconstructs `x`, `dx`, and `ddx` at Hsu sample midpoints;
- calls `model.nonlinear_jacobian_terms(...)`;
- scatters the analytical sampled Jacobian terms into sparse matrices for
  derivatives with respect to `x`, `dx`, and `ddx`.

`compute_floquet_autodiff(...)`:

- reconstructs the same sampled states;
- evaluates `model.local_nonlinear_force_torch(...)`;
- uses Torch autodiff to obtain local sampled Jacobian arrays;
- scatters those arrays to the same sparse sampled-Jacobian representation.

After this point both paths call the same monodromy routines. Torch is imported
only by the autodiff path.

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
