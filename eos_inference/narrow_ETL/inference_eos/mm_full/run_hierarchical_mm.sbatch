#!/bin/bash

#SBATCH -J narrow_ETL_hierarchical_mm_full
#SBATCH -o ./log_mm
#SBATCH -e ./log_mm

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta
export JAX_TRACEBACK_FILTERING=off

srun python run_hierarchical_mm.py
