#!/bin/bash
#SBATCH -Jcold-snakemake-all                    # Job name
#SBATCH -N1 --ntasks-per-node=8                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH -t2-00:00:00                           # Time limit D-HH:MM:SS
#SBATCH -ologs/slurm-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load mamba                               # Load module dependencies
mamba activate cold
snakemake --unlock
snakemake --local-cores 8 --rerun-incomplete --verbose