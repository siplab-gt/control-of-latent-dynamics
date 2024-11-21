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


## Usage

The usage of this workflow is described in the [Snakemake Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/?usage=siplab-gt%2Fcontrol-of-latent-dynamics).

If you use this workflow in a paper, don't forget to give credits to the authors by citing the URL of this (original) control-of-latent-dynamics repository and its DOI (see above).
