# Snakemake workflow: `control-of-latent-dynamics`

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![GitHub actions status](https://github.com/siplab-gt/control-of-latent-dynamics/workflows/Tests/badge.svg?branch=main)](https://github.com/siplab-gt/control-of-latent-dynamics/actions?query=branch%3Amain+workflow%3ATests)

A Snakemake workflow for running all simulations and producing figures for "Closed-loop optogenetic control of latent dynamics".
This entails:

- training of [factor-based SNNs (fSNNs)](https://www.sciencedirect.com/science/article/pii/S0896627322010807)
- wrapping fSNNs with Cleo and producing system identification (training) input/output data via stimulation and recording
- fitting [DPAD](https://www.nature.com/articles/s41593-024-01731-2) dynamics models to said data to support latent factor inference and inform optimal control
- constructing optimal controllers from those fit dynamics models
- incorporating those controllers into Cleo to simulate complete closed-loop optogenetics experiments
- performing data analysis on results, computing error, latency, causal inference power, etc.
- exploring a large parameter space of different network and experimental features
- integrating results into final figures

## Installing dependencies

```bash
mamba env create -f workflow/envs/main.yaml
mamba activate cold
```

Since Snakemake's conda method straight from the file struggles with the relative path to install the source package here, we just make sure to prepare the environment first.
Keep this environment up-to-date with the env file like so:

```bash
mamba env update -n cold -f workflow/envs/main.yaml --prune 
```

## Usage

- Simply run `snakemake`.
- To run on GT's PACE, include the `--profile pace` flag---better yet, copy or link this file to `~/.config/snakemake/default` so it's automatic.
- To debug a given output, run `snakemake [target] --config debug=true`
- The recommended way to run is from a login node, not as a SLURM job---since this is long-running, make sure to do so from a tmux session.
- Though sometimes local rules get killed, I presume from using too much memory.
    - Better to use a small job instead---`sbatch --account {...} workflow/smk-submit-slurm.sbatch`.
    - Snakemake bails when not run from a main process though...I commented out the `assert workflow.is_main_process` line that did that.

