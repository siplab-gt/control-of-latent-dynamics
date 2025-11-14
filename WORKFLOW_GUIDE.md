# Workflow Guide: Control of Latent Dynamics

This guide explains which Python files to run, what they do, and how they connect in the workflow.

## Overview

This is a **Snakemake workflow** for simulating closed-loop optogenetic control experiments. The workflow follows this pipeline:

1. **Train fSNN** (factor-based Spiking Neural Network)
2. **System Identification** - Collect stimulation/recording data
3. **Model Fitting** - Fit dynamics models to the data
4. **Control Experiments** - Run closed-loop control simulations
5. **Analysis & Plotting** - Analyze results and generate figures

## Main Entry Point: Snakemake

**You typically don't run Python files directly!** Instead, use Snakemake:

```bash
# Activate conda environment
conda activate cold

# Run the entire workflow (generates all figures)
snakemake -c1

# Run specific targets
snakemake results/figs/sufficient_stim_reaching.pdf
```

The `Snakefile` orchestrates everything and handles dependencies automatically.

## Workflow Pipeline (in order)

### 1. **fSNN Preparation** (`prep_fsnn.py`)
**What it does:**
- Builds and trains a factor-based Spiking Neural Network (fSNN)
- Runs the network in different modes: `demean`, `train`, `test`
- Saves trained network and/or recorded data

**Key outputs:**
- `fsnn_post_demean.pkl.gz` - Network after demeaning
- `fsnn_post_train.pkl.gz` - Trained network
- `fsnn_test_data.pkl.gz` - Test data recordings

**When to run directly:**
- Usually run via Snakemake, but you can test with:
```python
python src/scripts/prep_fsnn.py
```

---

### 2. **CLEO Preparation** (`cleo_prep.py`)
**What it does:**
- Wraps the fSNN with CLEO (simulation framework for optogenetics)
- Sets up the interface array (stimulation/recording channels)
- Prepares the simulation environment

**Key outputs:**
- `cleo_prep.pkl.gz` - Prepared CLEO simulation object

**Dependencies:**
- Requires `fsnn_post_train.pkl.gz` from step 1

---

### 3. **System Identification Experiments** (`sys_id_expts.py`)
**What it does:**
- Runs stimulation experiments to collect input/output data
- Records spikes, latent factors, and behavior
- This data is used to fit the dynamics model

**Key outputs:**
- `sys_id_data.pkl.gz` - System identification data

**Dependencies:**
- Requires `cleo_prep.pkl.gz` from step 2

**Visualization:**
- `plot_expt_data_trials.py` - Plots individual trials
- `plot_expt_data_wide.py` - Wide-format plots

---

### 4. **Model Fitting** (`model_fit.py`)
**What it does:**
- Fits a Linear Dynamical System (LDS) model to the sys_id data
- Uses either `known_factors.py` or `braid.py` depending on setting
- Learns the dynamics: how stimulation affects latent factors and behavior

**Key outputs:**
- `model_fit_lds.pkl.gz` - Fitted LDS model
- `model_fit_pred_data.pkl.gz` - Model predictions

**Dependencies:**
- Requires `sys_id_data.pkl.gz` from step 3

**Analysis:**
- `analyze_model_fit_data.py` - Computes R², RMSE, etc. for model fit quality

**Visualization:**
- `plot_model_fit_pred.py` - Compares model predictions to actual data

---

### 5. **Control Experiments** (`ctrl_expts.py`)
**What it does:**
- Runs closed-loop control experiments
- Uses controllers (OL=Open Loop, LQMPC=Linear Quadratic Model Predictive Control)
- Applies optimal stimulation based on the fitted model

**Key outputs:**
- `ctrl_data.pkl.gz` - Control experiment results

**Dependencies:**
- Requires `cleo_prep.pkl.gz` from step 2
- Requires `model_fit_lds.pkl.gz` from step 4

**Controllers available:**
- `OL` - Open loop (no feedback)
- `LQMPC` - Linear Quadratic MPC (optimal control)
- `NULL` - Null controller (baseline)

**Visualization:**
- `plot_ctrl_data.py` - Plots control experiment results

---

### 6. **Analysis** (`analyze_ctrl_data.py`, `analyze_fsnn_data.py`)
**What it does:**
- Computes performance metrics (R², RMSE, latency, etc.)
- Compares control performance to baselines
- Generates summary statistics

**Key outputs:**
- `summary.csv` - Summary metrics for each condition
- `fsnn_test_data_analysis.csv` - fSNN performance metrics

**Dependencies:**
- Requires `ctrl_data.pkl.gz` from step 5
- Requires `fsnn_test_data.pkl.gz` from step 1

---

### 7. **Figure Generation**
**What it does:**
- Generates publication-quality figures
- Combines results across parameter sweeps

**Key scripts:**
- `sufficient_stim_fig.py` - Figure for "sufficient stimulation" setting
- `optimal_stim_rec_fig.py` - Figure for "optimal stimulation/recording" setting
- `plot_schematic.py` - Schematic diagrams

**Key outputs:**
- `results/figs/{setting}_{task}.pdf` - Main figures

**Dependencies:**
- Requires aggregated `summary.csv` files from step 6

---

## Special Purpose Scripts

### **Impulse Responses** (`impulse_responses.py`)
- Tests system response to impulse stimulations
- Useful for understanding system dynamics

### **Plotting Scripts** (various `plot_*.py`)
- `plot_fsnn.py` - Visualize fSNN activity
- `plot_cleo.py` - Visualize CLEO setup
- `plot_fsnn_data.py` - Plot fSNN recordings

---

## How to Run Things

### Option 1: Use Snakemake (Recommended)
```bash
# Run everything
snakemake -c1

# Run specific figure
snakemake results/figs/sufficient_stim_reaching.pdf

# Run with more cores
snakemake -c4

# Dry run (see what would be executed)
snakemake -n
```

### Option 2: Run Individual Scripts (for testing/debugging)
Most scripts are designed to work with Snakemake, but you can test them:

```python
# Example: test prep_fsnn
from src.scripts.prep_fsnn import main
# ... set up parameters ...
main(cond_params, base_params, device, ...)
```

### Option 3: Use Test Rules
The Snakefile has convenient test rules:
```bash
# Test Q1 (sufficient_stim, reaching)
snakemake testq1

# Test Q2 (optimal_stim_rec, reaching)
snakemake testq2
```

---

## Configuration

All parameters are in `config/config.yaml`:
- `grid_sweep_params` - Parameters to sweep (all combinations)
- `star_sweep_params` - Parameters to explore one at a time
- `manual_sweep_params` - Parameters handled manually (task, setting, etc.)

---

## Data Flow Summary

```
prep_fsnn.py
    ↓ (fsnn_post_train.pkl.gz)
cleo_prep.py
    ↓ (cleo_prep.pkl.gz)
sys_id_expts.py
    ↓ (sys_id_data.pkl.gz)
model_fit.py
    ↓ (model_fit_lds.pkl.gz)
ctrl_expts.py
    ↓ (ctrl_data.pkl.gz)
analyze_ctrl_data.py
    ↓ (summary.csv)
{setting}_fig.py
    ↓ (results/figs/{setting}_{task}.pdf)
```

---

## Key Directories

- `results/` - All output data and figures
- `logs/` - Log files from each step
- `src/scripts/` - Main workflow scripts
- `src/ctrl/` - Controller implementations
- `src/model_fit/` - Model fitting methods
- `src/fsnn/` - fSNN implementation

---

## Tips

1. **Start with test rules**: Use `snakemake testq1` or `testq2` to run a single condition
2. **Check logs**: Look in `logs/` directory if something fails
3. **Use dry run**: `snakemake -n` shows what would run without executing
4. **Parallel execution**: Use `-c N` to run N jobs in parallel
5. **Incremental runs**: Snakemake only reruns what's needed based on file timestamps


