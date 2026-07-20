#!/bin/bash

# ----------------------------
# Defaults
# ----------------------------
SEED=58934836
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

    sbatch --qos short <<SBATCH_EOF
#!/bin/bash

#SBATCH -J nf_narrow_ETL_${src}
#SBATCH -o ./source_${src}/log_nf
#SBATCH -e ./source_${src}/log_nf

#SBATCH --partition cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=00:30:00

eval "\$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta


cat <<CONFIG_EOF > nf_config_${src}.yaml

# Configuration for training normalizing flow
# Updated to use new default hyperparameters (PR #55)
posterior_file: ./source_${src}/posterior.npz
output_dir: ./source_${src}/nf

# Parameter selection
parameter_names: ["log10_mej_wind"]

# Training parameters
num_epochs: 500
learning_rate: 0.0001
max_patience: 500
batch_size: 128
val_prop: 0.2
seed: 0

# Flow architecture
flow_type: block_neural_autoregressive_flow
flow_layers: 1
nn_depth: 4
nn_width: 50
nn_block_dim: 8
invert: true

# Transformer settings (NEW DEFAULTS)
transformer: rational_quadratic_spline
transformer_knots: 10
transformer_interval: 5.0

# Data preprocessing (NEW DEFAULTS)
max_samples: 30_000
standardize: true
standardization_method: zscore

# Plotting
plot_corner: true
plot_losses: true

# Conditional flow settings
cond_dim: 1
cond_parameter_names: ["log10_mej_dyn"]
CONFIG_EOF

train_jester_flow "./nf_config_${src}.yaml"
rm ./nf_config_${src}.yaml
SBATCH_EOF

done
