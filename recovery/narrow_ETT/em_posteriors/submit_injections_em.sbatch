#!/bin/bash

#SBATCH -J narrow_ET_em
#SBATCH -o ./log
#SBATCH -e ./log

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

srun python run_injections_em.py