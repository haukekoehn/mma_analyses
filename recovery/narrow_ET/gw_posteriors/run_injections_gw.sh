#!/bin/bash
nsrc=$(wc -l < ../events.dat)
nsrc=$(( nsrc - 1 ))
nsrc=10

for (( src=0; src<nsrc; src++ ));
do


    sbatch <<EOF
#!/bin/bash
#SBATCH -J narrow_ET_${src}
#SBATCH -o ./source_${src}/log
#SBATCH -e ./source_${src}/err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 192
#SBATCH -p cpu
#SBATCH --time=48:00:00

eval "\$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p source_${src}

srun python run_injections_gw.py --source ${src} --outdir source_${src}
EOF

done
