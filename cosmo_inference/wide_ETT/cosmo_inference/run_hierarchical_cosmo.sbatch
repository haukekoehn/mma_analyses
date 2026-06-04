#!/bin/bash

#SBATCH -J wide_ETT_hierarchical_cosmo
#SBATCH -o ./log
#SBATCH -e ./log

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=08:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta
export JAX_TRACEBACK_FILTERING=off

srun python run_hierarchical_cosmo.py
