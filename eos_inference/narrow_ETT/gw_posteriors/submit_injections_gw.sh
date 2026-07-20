#!/bin/bash

# ----------------------------
# Defaults
# ----------------------------
SEED=4789203
EVENTS_INPUT=""
ALL_EVENTS=1

# ----------------------------
# Parse arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --random-seed)
            SEED="$2"
            shift 2
            ;;
        *)
            # assume first positional argument = event list
            EVENTS_INPUT="$1"
            ALL_EVENTS=0
            shift
            ;;
    esac
done

# ----------------------------
# Build event list
# ----------------------------
nsrc=$(wc -l < ../events.dat)
nsrc=$(( nsrc - 1 ))

if [[ $ALL_EVENTS -eq 1 ]]; then
    EVENT_LIST=$(seq 0 $((nsrc - 1)))
else
    # convert "12,35,6" -> array
    IFS=',' read -r -a EVENT_ARRAY <<< "$EVENTS_INPUT"
    EVENT_LIST="${EVENT_ARRAY[@]}"
fi

# ----------------------------
# Submit jobs
# ----------------------------
for src in $EVENT_LIST; do

    # optional: basic bounds check
    if (( src < 0 || src >= nsrc )); then
        echo "Skipping invalid src index: $src"
        continue
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH -J narrow_ETT_gw_${src}
#SBATCH -o ./source_${src}/log
#SBATCH -e ./source_${src}/log
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 192
#SBATCH -p cpu
#SBATCH --time=08:00:00

eval "\$(conda shell.bash hook)"
conda activate nmma_x_fiesta

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

rm -r source_${src}/source_*
rm -r source_${src}/posterior*
rm -r source_${src}/.distance_marginalization_lookup.npz
mkdir -p source_${src}

srun python run_injections_gw.py \
    --source ${src} \
    --outdir source_${src} \
    --random-seed ${SEED}
EOF

done
