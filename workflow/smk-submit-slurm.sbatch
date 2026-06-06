#!/bin/bash
#SBATCH -Jcold-snakemake-coord                  # Job name (coordinator)
#SBATCH -N1 --ntasks-per-node=1                 # Coordinator only needs 1 core
#SBATCH --mem-per-cpu=4G                        # Memory for coordinator
#SBATCH -t20:00:00                              # Time limit (fit before 2026-05-12T06:00 maintenance)
#SBATCH -ologs/slurm-%j.out                         # Combined output and error messages file
#SBATCH -A gts-crozell3                  # Account (primary recovered to 75.6 SU 2026-06-01; ece fallback)
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3                           # Load module dependencies
module load cuda/12.6.1                         # CUDA 12.6 required by torch 2.10+cu126
eval "$(conda shell.bash hook)"
conda activate ~/r-crozell3-0/envs/USID

# USID repo path (required for model_fit_method=usid)
export USID_REPO_ROOT="/storage/project/r-crozell3-0/zbeddingfield3/projects/usid"
export PYTHONPATH="${PYTHONPATH}:${USID_REPO_ROOT}"

# Config file selection (default to testconfig for faster runs).
# Two configfiles: one for params (defaults + rule blocks), one for sweeps
# (Cartesian-product axes). Both are merged by Snakemake.
CONFIG_PARAMS="${1:-../config/testconfig_params.yaml}"
CONFIG_SWEEPS="${2:-../config/testconfig_sweeps.yaml}"
echo "Using params: $CONFIG_PARAMS"
echo "Using sweeps: $CONFIG_SWEEPS"

mkdir -p "$(readlink -f results 2>/dev/null || echo results)" logs

# --unlock needs the same configfiles because the Snakefile's parse-time
# validation (lines 28-32) requires `params:` + `sweeps:` blocks before any
# subcommand can run. Without them, --unlock fails with ValueError on a
# stale lock from a previously-scancel'd coordinator.
snakemake --configfiles "$CONFIG_PARAMS" "$CONFIG_SWEEPS" --unlock

# Batch label for USID results grouping: YYYYMMDD_<coordinator_job_id>
COLD_BATCH_LABEL="$(date +%Y%m%d)-${SLURM_JOB_ID:-local}"

# SLURM executor mode: submits each job to SLURM queue
# --jobs 100: up to 100 parallel SLURM jobs
# --default-resources: set defaults for jobs that don't specify
# SLURM executor with PACE settings (adapted from profiles/pace)
# NOTE: Removed sdm=conda due to PACE conda version (24.1.2) not meeting
# Snakemake 8.x requirement (24.7.1+). Using already-activated USID env instead.
snakemake \
    --configfiles "$CONFIG_PARAMS" "$CONFIG_SWEEPS" \
    --config cold_batch_label="$COLD_BATCH_LABEL" \
    --executor slurm \
    --jobs 100 \
    --max-jobs-per-second 1 \
    --max-status-checks-per-second 2 \
    --default-resources slurm_account=gts-crozell3 tasks_per_gpu=0 \
    --envvars LD_LIBRARY_PATH PYTHONPATH USID_REPO_ROOT \
    --rerun-incomplete \
    --keep-going
