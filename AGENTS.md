# AGENTS.md

This file captures project-local guidance for future coding agents working on pyHB.

## Project Shape

- The Python package is `pyhb`; do not use the old `ega_ihb` name.
- Reusable library code lives only under `pyhb/`.
- System-specific code belongs under `examples/<system>/`.
- MATLAB files and generated reference material are not part of the publishable Python library.
- `tests/`, `matlab_references/`, and all `examples/*/results/` directories are ignored by git in this repository. They are still useful locally.

## Environment And Commands

- Preferred Python executable:
  `D:\anaconda3\envs\pytorch_main\python.exe`
- Full local regression:
  `D:\anaconda3\envs\pytorch_main\python.exe -m unittest discover -s tests -v`
- Package linting is configured through Ruff in `pyproject.toml`, but Ruff may not always be installed.
- Use PowerShell-compatible commands on Windows.
- Before committing, inspect:
  `git status --short --branch`

## Library Boundaries

- Keep `pyhb/` generic. Do not add 4DOF, aero-engine, Bernoulli beam, or other system-specific logic to the library.
- Public model interfaces are in `pyhb/models.py`.
- Public solvers are:
  - `ContinuationSolver` for full-system analytical Jacobians.
  - `ContinuationAutodiffSolver` for full-system Torch autodiff nonlinear derivatives.
  - `CondensedContinuationSolver` for localized nonlinearities with Schur condensation.
- `pyhb.__init__` intentionally uses lazy autodiff exports. Do not add top-level imports of `continuation_autodiff` or `floquet_autodiff`, because Torch is an optional dependency.
- Preserve the structured linear interface:
  - `LinearOperatorTerm(matrix, basis_type, parameter_power)`
  - `ForcingTerm(samples, parameter_power)`
  - `basis_type` is one of `"ddx"`, `"dx"`, or `"x"`.
- Nonlinear model definitions should use local DOF APIs:
  - `nonlinear_force_dofs`
  - `nonlinear_coordinate_dofs`
  - `local_nonlinear_force(...)`
  - `local_nonlinear_jacobian_terms(...)`
  - optional `local_nonlinear_parameter_derivative(...)`
- For autodiff models, implement `local_nonlinear_force_torch(...)` and set `autodiff_variables` /
  `autodiff_parameter_dependent` as needed.

## Solver Behavior To Preserve

- Do not change residual signs, Newton update signs, Jacobian assembly, FFT projection, S3 tensors, or continuation formulas unless explicitly asked.
- Full and autodiff solvers share the `ContinuationSolver` continuation loop. Condensed continuation has its own loop.
- Initial coefficients must be supplied explicitly by callers.
- Nonlinear harmonics default to the full FFT nonlinear harmonic grid; do not reintroduce low-order nonlinear harmonic helpers.
- The time grid is the strict FFT grid:
  `t = arange(sample_fft) * period / sample_fft`.
- Weighted arc length uses:
  `||[q, omega]||_W^2 = ||q / q_scale||^2 + (omega / omega_scale)^2`.
  The weight is applied only to the arc-length constraint row, not to the physical residual Jacobian.
- `max_parameter_step` is an accepted-point sampling guard. It must apply only after Newton has converged. Unconverged steps should report `no convergence` and use the normal shrink path.
- Keep solver stdout optional through `progress_callback`; library code should be silent by default.

## Examples

- Each system folder is intentionally self-contained. Avoid shared cross-system example helpers.
- Typical system layout:
  - `model.py` for analytical model definitions.
  - `autodiff_model.py` when Torch autodiff is demonstrated.
  - `run_arc.py`, `run_condensed_arc.py`, `run_autodiff_arc.py` as applicable.
  - matching `postprocess_*.py` scripts for RMS plots and stability tables.
- Model files should define only model/data loading code. Solver parameters such as harmonics, frequency resolution,
  step sizes, output paths, and plotting DOFs belong in run or postprocess scripts.
- Run scripts save minimal NPZ payloads:
  - `parameter_history`
  - `coefficient_history`
- Postprocess scripts reconstruct time responses from the run script constants and may compute RMS, plots, and Floquet stability.
- Results should be written under each system's `results/` directory; these directories are ignored and scripts should create them.
- Preserve existing example defaults unless the user explicitly asks to change numerical settings.

## Floquet Stability

- Floquet computation is postprocessing, not continuation solver logic.
- Current method names are:
  - `trapezoid` for the Hsu/trapezoid monodromy method.
  - `exponential` for the piecewise matrix exponential reference method.
- Do not reintroduce removed `auto` or `dominant` methods without a new design discussion.
- Stability output format is a three-column NumPy table:
  `omega, spectral_radius, stable_flag`.
- Use `stability_tolerance` to avoid classifying near-unit numerical noise as clearly unstable.
- Floquet logging should remain concise: one summary line in examples and one completion line per point.

## Git And Generated Files

- Do not commit generated `results/`, caches, `.idea/`, MATLAB references, or local test artifacts.
- Because `tests/` is ignored in this repository, local test edits may not show in git status. Mention this explicitly if relevant.
- Do not use destructive git commands such as `git reset --hard` or `git checkout --` unless the user clearly requests them.
- If a working tree has user changes, preserve them and work around them; do not revert unrelated local edits.

## Safe Change Policy

- The current numerical behavior is considered validated. Prefer minimal, localized edits.
- For library changes, run the full unittest suite before reporting success.
- For example changes, run at least a small smoke test for the touched example, using a temporary output path.
- When adding a new example, follow an existing system folder pattern and keep the saved NPZ minimal.
- When adding a new public API, update `pyhb/__init__.py` carefully and preserve optional dependency boundaries.
