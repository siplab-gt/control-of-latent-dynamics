# Agent Guide: Control of Latent Dynamics

This repository contains a Snakemake workflow and Python source code for "Closed-loop optogenetic control of latent dynamics". It involves factor-based SNN (fSNN) training, system identification, dynamics modeling (DPAD/LDS), and closed-loop control simulation using Cleo.

## Project Structure

- `src/`: Core Python packages.
  - `ctrl/`: Control implementations (LQMPC, OL, NULL).
  - `fsnn/`: Factor-based SNN models and task-specific logic (cycling, reaching).
  - `model_fit/`: Dynamics modeling (BRAID/DPAD, LDS).
  - `scripts/`: Entry points for Snakemake rules.
  - `helpers.py`: Shared utilities for seeding, logging, and parameter preprocessing.
- `workflow/`: Snakemake orchestration.
  - `Snakefile`: Main workflow definition.
  - `envs/`: Conda environment definitions.
- `config/`: Configuration files (YAML).
- `results/`: Generated data, models, and figures (ignored by git, but structure is parameter-dependent).

## Essential Commands

### Environment Setup
```bash
mamba env create -f workflow/envs/main.yaml
mamba activate cold
```

### Running the Workflow
```bash
# Run everything
snakemake --use-conda

# Run a specific rule
snakemake results/figs/diagnostics.pdf

# Debugging (attaches debugger on port 5678)
snakemake [target] --config debug=true
```

### HPC Execution (PACE)
```bash
# Using the pace profile
snakemake --profile pace

# Or via sbatch
sbatch workflow/smk-submit-slurm.sbatch
```

## Naming Conventions & Patterns

- **Parameters and Wildcards**: The project uses a complex path structure for results based on a large number of parameters (e.g., `results/task_reaching/setting_optimal_stim_rec/...`).
- **Brian2 Units**: Many parameters in `config.yaml` or script arguments use unit suffixes like `_ms`, `_s`, `_um`, `_mm`. The `preprocess_params` function in `src/helpers.py` automatically converts these to `brian2.Quantity` objects.
- **Dill + Gzip**: Large objects (cleo sims, models, datasets) are typically saved as `.pkl.gz` using `dill` for robust serialization of complex objects/lambdas.
- **Controller Interface**: All controllers should inherit from `src.ctrl.Controller`.
- **Seeding**: Use `src.helpers.seed_everything(seed, "optional_string")` to ensure reproducibility across NumPy, Brian2, Cleo, and TensorFlow.

## Key Libraries
- **Cleo**: For closed-loop optogenetics simulation.
- **Brian2 / brian2cuda**: For spiking neural network simulation.
- **BRAID / DPAD**: For latent dynamics modeling.
- **Snakemake**: For workflow management.

## Gotchas

- **Path Wildcards**: Adding a new parameter to the sweep often requires updating `Snakefile`'s `cond_params` and `paths_for_params` to ensure it's correctly reflected in output paths.
- **Memory/Runtime**: SNN simulations and BRAID training can be resource-intensive. `Snakefile` contains several lambda functions to estimate `mem_mb` and `runtime` based on network scale.
- **Cython Cache**: Multiple concurrent jobs can sometimes conflict when compiling Cython extensions for Brian2. Snakemake rules for experiments have `retries: 1` to mitigate this.
- **Latent Dims**: The workflow handles cases where `n_latent_dims` is varied relative to a baseline using `n_latent_dims_offset`.

## Testing
There are no explicit `pytest` tests; the workflow itself acts as the integration test suite. Check `.github/workflows/main.yml` for CI details.
