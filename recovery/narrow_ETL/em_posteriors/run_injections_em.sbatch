#!/bin/bash

#SBATCH -J narrow_ET_em
#SBATCH -o ./log_em
#SBATCH -e ./err_em

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=20:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

srun python run_injections_em.py