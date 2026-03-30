#!/bin/bash

#SBATCH -J gw_test
#SBATCH -o ./log
#SBATCH -e ./log
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 192
#SBATCH -p cpu
#SBATCH --time=24:00:00

eval $(conda shell.bash hook)  # Initialize Conda in the script
conda activate nmma_x_fiesta

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun python run_injections_gw.py --source 2 --outdir .