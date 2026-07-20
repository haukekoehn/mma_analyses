#!/bin/bash

#SBATCH -J wide_ETT_mm_pdet
#SBATCH -o ./log_pdet
#SBATCH -e ./log_pdet

#SBATCH --partition cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta
export JAX_TRACEBACK_FILTERING=off

python calculate_pdet.py
