#!/bin/bash

#SBATCH -J narrow_ET_hierarchical
#SBATCH -o ./log_mm
#SBATCH -e ./log_mm

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=00:30:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

srun python run_hierarchical_mm.py
