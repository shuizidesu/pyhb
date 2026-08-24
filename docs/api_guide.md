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
- `ContinuationSolver`, `ContinuationAutodiffSolver`,
  `ContinuationFreeFrequencySolver`, and
  `ContinuationFreeFrequencyAutodiffSolver` run harmonic-balance continuation.
- `compute_floquet`, `compute_free_frequency_floquet`, and their autodiff
  counterparts compute Floquet multipliers for accepted HB solutions.
- `FrequencyGrid` exposes the frequency-grid checks used by the solvers.

Lower-level modules provide the assembly building blocks used by the solvers:

- `pyhb.models` defines structured linear and forcing terms,
  `LocalJacobianMatrices`, DOF validation, and local-to-global nonlinear
  scattering.
- `pyhb.harmonics` defines the HB basis order and FFT projection helpers.
- `pyhb.hb_operators` builds reusable HB metadata: frequency grids, derivative
  maps, linear projection matrices, and the flattened S3 operators used for nonlinear
  Jacobian projection.
- `pyhb.continuation_core` orchestrates shared continuation preparation and
  provides powered-block, arc-metric, Jacobian-assembly, and bordered-solve
  utilities. HB operator construction remains in `pyhb.hb_operators`.
- `pyhb.continuation` assembles the full analytical residual and Newton
  correction matrix, then runs weighted arc-length continuation.
- `pyhb.continuation_autodiff` reuses the full continuation loop but obtains
  nonlinear derivatives from Torch autodiff.
- `pyhb.continuation_free_frequency` handles autonomous/free-frequency systems where the
  response frequency is an unknown.
- `pyhb.continuation_free_frequency_autodiff` applies Torch autodiff to the same
  free-frequency continuation loop.
- `pyhb.floquet` reconstructs one HB solution, samples analytical nonlinear
  Jacobians in time, and computes monodromy multipliers.
- `pyhb.floquet_autodiff` uses Torch autodiff to build the same sampled
  nonlinear Jacobians before calling the shared Floquet backend.
- `pyhb.floquet_mixed_autodiff` handles mixed second-/first-order descriptor
  systems whose mass matrix is singular in the original coordinate layout.

The typical data flow is:

```text
model samples
  -> HB/FFT projection
  -> precomputed frequency grid, basis blocks, and flattened S3 matrices
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
  nonlinear Jacobian entries into a coefficient matrix used with S3 operators.

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

Continuation solvers choose the nonlinear Jacobian projection grid internally.
For response harmonics with maximum FFT-bin index `hmax`, the nonlinear
projection harmonics are the positive bins through `2*hmax`. Higher S3 columns
are theoretically zero because products of retained HB basis functions cannot
exceed the sum of two retained harmonic frequencies.

### HBContext

`HBContext` is built during solver initialization. It is not usually constructed
directly by end users, but it explains what is precomputed before Newton
iterations begin:

- validated response harmonics and their FFT bin indices;
- solver-generated nonlinear projection harmonics and their FFT bin indices;
- the fundamental period and sample count;
- `order = 2*len(harmonics) + 1`;
- the S3 projection matrix for `x` derivatives;
- the derivative S3 projection matrices `s3_dx` and `s3_ddx`.

The continuation solvers keep this context in their prepared problem objects.
The autodiff continuation solvers use the same context; only the source of
nonlinear derivative samples changes.

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
basis contribution. These maps are also used to convert the base S3 projection
matrix into the derivative projection matrices for `dx` and `ddx` nonlinear
Jacobian terms.

### S3 Construction

The S3 matrix is the common projection operator for nonlinear Jacobians. It couples
one residual test basis function, one response basis function, and one nonlinear
Jacobian Fourier basis function. It is stored as a two-dimensional CSR sparse
matrix with shape `(order*order, nonlinear_order)`. Values below `1e-12` are
discarded during preprocessing to remove FFT roundoff from theoretically zero
entries before the sparse representation is built.

Mathematically this is an `(order, order, nonlinear_order)` tensor. `s3`,
`s3_dx`, and `s3_ddx` all use the same flattened shape and Fortran ordering.
The derivative variants are formed by tensor contraction with the coefficient
derivative maps; no `(order*order, order*order)` Kronecker matrix is constructed.

`compute_s3(...)` selects the implementation:

- `integrate_s3_fast(...)` uses an aligned periodic FFT grid and is the default
  `s3_method="fast"` path.
- `integrate_s3(...)` uses numerical quadrature and is selected by
  `s3_method="quad"`.

After S3 is computed, `HBContext.build(...)` constructs:

- `s3`: used for derivatives of nonlinear force with respect to `x`;
- `s3_dx`: used for derivatives with respect to `dx`;
- `s3_ddx`: used for derivatives with respect to `ddx`.

These matrices are precomputed once for the configured harmonic grid. Runtime
nonlinear Jacobian assembly multiplies the selected sparse S3 matrix by the dense
Fourier coefficients of local derivative samples. Contributions associated with
`x`, `dx`, and `ddx` are summed in local HB blocks before those blocks are
scattered once into the global sparse Jacobian.

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
- `local_nonlinear_jacobian(...)`: returns local analytical Jacobian matrices
  as `LocalJacobianMatrices`.

Optional method:

- `local_nonlinear_parameter_derivative(...)`: returns explicit
  `dN/dparameter` samples. The default is zero.

The solver projects and scatters local nonlinear force and Jacobian data to
global DOFs.

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

### LocalJacobianMatrices

```python
from pyhb import LocalJacobianMatrices

LocalJacobianMatrices(
    x=dndx_samples,
    dx=dnddx_samples,
    ddx=None,
)
```

Fields:

- `x`: optional batched `dN/dx` matrix.
- `dx`: optional batched `dN/d(dx)` matrix.
- `ddx`: optional batched `dN/d(ddx)` matrix.

Every non-`None` array has shape
`(samples, len(nonlinear_force_dofs), len(nonlinear_coordinate_dofs))`.
The `dx` and `ddx` fields refer to derivatives on the dimensionless HB
phase-time grid. Omit a field with `None` when the nonlinear law does not
depend on that variable.

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
- `linear_solver`: `"sparse"` keeps the default CSC/SuperLU Newton path;
  `"dense"` assembles the nonlinear HB Jacobian directly as a dense matrix and
  uses LAPACK LU for systems whose nonlinear Jacobian is globally dense.
- `progress_callback`: optional callback receiving progress strings. Library
  code is silent when this is `None`.

`ContinuationSolver.run(initial_coefficients)` requires explicit initial
coefficients shaped `(order, n_dof)` or an equivalent flattened vector.

`ContinuationResult` contains:

- `coefficients`: final accepted coefficient matrix shaped `(order, n_dof)`.
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
  and S3 operators;
- a parameter column from forcing, nonlinear parameter derivative, and linear
  parameter derivative terms.

Both linear-solver modes retain the structured linear HB operators as CSC
matrices. In dense mode, the S3 contraction produces a dense nonlinear HB
Jacobian, the sparse linear entries are accumulated into that dense matrix,
and `scipy.linalg.lu_factor` / `lu_solve` solve the Newton systems. This mode is
intended only for moderate-size systems whose nonlinear Jacobian makes the
complete Newton matrix dense. Its matrix storage alone requires approximately
`8 * (n_dof * order)**2` bytes. Existing local-nonlinearity examples should
keep the default sparse solver.

For an analytical model, only `linear_solver` needs to be selected:

```python
from pyhb import ContinuationConfig, ContinuationSolver

config = ContinuationConfig(
    linear_solver="sparse",  # "sparse" or "dense"
)
result = ContinuationSolver(model, config).run(initial_coefficients)
```

The same field is inherited by `ContinuationFreeFrequencyConfig`.

## Torch Autodiff Continuation

Use `ContinuationAutodiffSolver` when the nonlinear force is easier to define
with Torch than with analytical Jacobian samples.

```python
from pyhb import ContinuationAutodiffConfig, ContinuationAutodiffSolver

config = ContinuationAutodiffConfig(
    harmonics=(1.0, 2.0, 3.0),
    torch_device=None,
    linear_solver="sparse",
    autodiff_jacobian_mode="dense",
)

result = ContinuationAutodiffSolver(model, config).run(initial_coefficients)
```

`ContinuationAutodiffConfig` extends `ContinuationConfig` with:

- `torch_device`: optional Torch device selector such as `"cpu"` or `"cuda"`.
  `None` uses CUDA when available, otherwise CPU.
- `autodiff_jacobian_mode`: `"dense"` preserves the standard batched Jacobian
  path. `"sparse"` detects strictly active local force-coordinate pairs after
  each dense `jacrev`, performs their batched `rfft` on the Torch device, and
  transfers only compact Fourier coefficients for S3 contraction and sparse
  global assembly.

The sparse mode is useful when nonlinear forces are distributed over many DOFs
but each force depends on only a few coordinates. It does not remove the peak
memory required by the dense `jacrev` result itself; it reduces subsequent
device transfer, FFT, S3 contraction, and COO/CSC assembly from all spatial
pairs to only strictly nonzero pairs. No numerical threshold is applied. The
configured `sample_fft` must place every internally generated nonlinear
harmonic index within the rFFT Nyquist range.

The two options control different stages:

- `autodiff_jacobian_mode` controls processing after Torch has evaluated the
  local time-sampled Jacobian. It does not select the Newton linear solver.
- `linear_solver` controls the representation and factorization of the final
  Newton Jacobian after the linear and nonlinear HB contributions are combined.

Their four possible combinations have the following status and intended use:

| `linear_solver` | `autodiff_jacobian_mode` | Status and intended use |
| --- | --- | --- |
| `"sparse"` | `"dense"` | **Supported and default.** The local AD Jacobian, FFT coefficients, and S3 contraction use dense arrays; the resulting HB blocks are scattered to CSC and combined with the sparse linear operator. Use this for localized nonlinearities and the existing pyHB examples. |
| `"sparse"` | `"sparse"` | **Supported.** After dense `jacrev`, pyHB retains only strictly active force-coordinate pairs, performs compact rFFT/S3 projection, and assembles CSC directly. Use this when nonlinear forces cover many DOFs but their coordinate dependence is diagonal, banded, or otherwise sparse. |
| `"dense"` | `"dense"` | **Supported.** The nonlinear HB Jacobian is assembled directly as a dense matrix, sparse linear entries are added to it, and LAPACK LU solves the final system. Use this only when the complete Newton Jacobian is genuinely dense and `n_dof * order` is moderate. |
| `"dense"` | `"sparse"` | **Unsupported.** Compact sparse AD projection is intended to produce a sparse final Jacobian. Expanding it into a dense Newton matrix would add sparse assembly and conversion cost without retaining its memory advantage. |

Accordingly, `linear_solver="dense"` requires
`autodiff_jacobian_mode="dense"`; the solver raises `ValueError` for the fourth
combination. The dense mode also requires approximately
`8 * (n_dof * order)**2` bytes for each full float64 matrix, in addition to LU
storage and temporary arrays. It should not be selected merely because the
physical system has many DOFs.

The same two fields and combination rules apply to
`ContinuationFreeFrequencyAutodiffConfig`:

```python
from pyhb import ContinuationFreeFrequencyAutodiffConfig

config = ContinuationFreeFrequencyAutodiffConfig(
    linear_solver="sparse",
    autodiff_jacobian_mode="sparse",
    torch_device="cuda",
)
```

The continuation loop, residual definition, linear assembly, S3 operators, and
result object match the full analytical solver. Only the nonlinear derivative
generation path is replaced by Torch autodiff.

## Free-Frequency Continuation

Use `ContinuationFreeFrequencySolver` for autonomous systems where the response
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
- `local_residual_jacobian(...)`: returns `LocalJacobianMatrices` containing
  local samples of `dG/dx`, `dG/d(dx)`, and `dG/d(ddx)`.

Optional derivatives default to zero:

- `local_residual_omega_derivative(...)`
- `local_residual_parameter_derivative(...)`

`ContinuationFreeFrequencyConfig` extends `ContinuationConfig` with:

- `init_parameter`: initial value of the true continuation parameter.
- `parameter_scale`: weighted arc-length scale for the true continuation
  parameter.
- `initial_constraint`: a fixed `HarmonicCoefficientConstraint` used for the
  preliminary Newton solve.
- `phase_condition`: a `ReferencePhaseCondition` by default, or an optional
  fixed `HarmonicCoefficientConstraint`, used during continuation.
- `constraint`: a compatibility alias for a fixed constraint configuration.
- `constraint_tolerance`: convergence tolerance for that constraint.

The preliminary Newton solve uses unknowns `[q, omega]` at fixed `parameter`.
Arc-length continuation uses `[q, omega, parameter]`. The preliminary constraint
row contains a single `1`; the default continuation phase row is generated from
the previous accepted reference solution.

`ContinuationFreeFrequencyResult` stores:

- `parameter_history`: accepted true continuation parameter values.
- `omega_history`: accepted response frequencies.
- `coefficient_history`: accepted HB coefficient history.

`ContinuationFreeFrequencyAutodiffSolver` uses
`AutodiffFreeFrequencySecondOrderTimeModel`. The model supplies
`local_residual_force_torch(...)`, and Torch autodiff builds the local
derivatives controlled by `autodiff_variables`, `autodiff_omega_dependent`, and
`autodiff_parameter_dependent`. Its config supports the same
`autodiff_jacobian_mode="dense"|"sparse"` projection choice as forced-response
autodiff continuation.

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
- calls `model.local_nonlinear_jacobian(...)`;
- consumes the same compact local Jacobian matrices used by the continuation
  solver.

`compute_floquet_autodiff(...)`:

- reconstructs the same sampled states;
- evaluates `model.local_nonlinear_force_torch(...)`;
- uses Torch autodiff to obtain local sampled Jacobian arrays;
- scatters those arrays to the same sparse sampled-Jacobian representation.

For autonomous/free-frequency systems use
`compute_free_frequency_floquet(...)` or
`compute_free_frequency_floquet_autodiff(...)`. These functions take both the
solved response frequency `omega` and the true continuation `parameter`; linear
operator powers are evaluated with `omega`, while residual Jacobian samples are
evaluated with `(omega, parameter)`.

After this point both paths call the same monodromy routines. Torch is imported
only by the autodiff path.

The trapezoid backend factors and applies one sampled step at a time, updating
the dense monodromy immediately so LU factorizations are not retained for the
whole period.

## Examples

Runnable reference systems live under `examples/<system>/`. Typical scripts are:

- `run_arc.py`: full analytical continuation.
- `run_autodiff_arc.py`: Torch autodiff continuation.
- `run_free_frequency.py`: free-frequency continuation for autonomous systems.
- `postprocess_arc.py`, `postprocess_autodiff_arc.py`, and related
  postprocess scripts: response reconstruction, plots, and optional stability
  tables.

Example run scripts save minimal `.npz` payloads with `parameter_history` and
`coefficient_history`. Postprocess scripts reconstruct time responses from the
same harmonic and sampling constants used by the corresponding run script.

The current source examples include aero-engine, Bernoulli beam, bilinear
hysteretic, and machine-absorber systems.
