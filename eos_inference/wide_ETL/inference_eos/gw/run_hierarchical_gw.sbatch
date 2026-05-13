#!/bin/bash

#SBATCH -J wide_ETL_hierarchical_gw
#SBATCH -o ./log_gw
#SBATCH -e ./log_gw

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=04:30:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta
export JAX_TRACEBACK_FILTERING=off

srun python run_hierarchical_gw.py
