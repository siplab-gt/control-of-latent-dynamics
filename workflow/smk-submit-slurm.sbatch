#!/bin/bash
#SBATCH -Jcold-snakemake-coord                  # Job name (coordinator)
#SBATCH -N1 --ntasks-per-node=1                 # Coordinator only needs 1 core
#SBATCH --mem-per-cpu=4G                        # Memory for coordinator
#SBATCH -t2-00:00:00                           # Time limit D-HH:MM:SS
#SBATCH -ologs/slurm-%j.out                         # Combined output and error messages file
#SBATCH -A gts-crozell3-ece              # Account
#SBATCH --mail-type=END,FAIL                    # Notify when job ends or fails
#SBATCH --mail-user=jrauf3@gatech.edu       # Replace with your email
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3                           # Load module dependencies
module load cuda/12.1.1                         # CUDA for Brian2CUDA (supports sm_70/V100)
eval "$(conda shell.bash hook)"
conda activate ~/r-crozell3-0/envs/USID

# Add USID repo to PYTHONPATH for model_fit_method=usid
USID_REPO="$(cd ../../.. && pwd)"
export PYTHONPATH="${PYTHONPATH}:${USID_REPO}"

# Config file selection (default to testconfig for faster runs)
CONFIG_FILE="${1:-../config/testconfig.yaml}"
echo "Using config: $CONFIG_FILE"

snakemake --unlock

# Batch label for USID results grouping: YYYYMMDD_<coordinator_job_id>
COLD_BATCH_LABEL="$(date +%Y%m%d)-${SLURM_JOB_ID:-local}"

# SLURM executor mode: submits each job to SLURM queue
# --jobs 100: up to 100 parallel SLURM jobs
# --default-resources: set defaults for jobs that don't specify
# SLURM executor with PACE settings (adapted from profiles/pace)
# NOTE: Removed sdm=conda due to PACE conda version (24.1.2) not meeting
# Snakemake 8.x requirement (24.7.1+). Using already-activated USID env instead.
snakemake \
    --configfile "$CONFIG_FILE" \
    --config cold_batch_label="$COLD_BATCH_LABEL" \
    --executor slurm \
    --jobs 100 \
    --default-resources slurm_account=gts-crozell3-ece tasks_per_gpu=0 \
    --rerun-incomplete \
    --keep-going
