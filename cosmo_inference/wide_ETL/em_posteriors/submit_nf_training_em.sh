#!/bin/bash

nsrc=$(wc -l < ../../../eos_inference/wide_ETL/events.dat)-1

if [ $# -eq 0 ]; then
    sources=$(seq 0 $((nsrc-1)))
else
    sources=$(echo "$1" | tr ',' ' ')
fi

for src in $sources;
do

sbatch --qos short <<SBATCH_EOF
#!/bin/bash

#SBATCH -J nf_wide_ETL_${src}
#SBATCH -o ./source_${src}/log_nf
#SBATCH -e ./source_${src}/log_nf

#SBATCH --partition cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:00:00

eval "\$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

mkdir -p ./source_${src}

cat <<CONFIG_EOF > nf_config_${src}.yaml

# Configuration for training normalizing flow
# Updated to use new default hyperparameters (PR #55)
posterior_file: ../../../eos_inference/wide_ETL/em_posteriors/source_${src}/posterior.npz
output_dir: ./source_${src}/nf

# Parameter selection
parameter_names: ["log10_mej_dyn", "luminosity_distance", "cos_theta_jn"]

# Training parameters
num_epochs: 3000
learning_rate: 0.00005
max_patience: 500
batch_size: 128
val_prop: 0.2
seed: 564783

# Flow architecture
flow_type: masked_autoregressive_flow
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
max_samples: 10_000
standardize: true
standardization_method: zscore

# Plotting
plot_corner: true
plot_losses: true

# Conditional flow settings
cond_dim: null
CONFIG_EOF

train_jester_flow "./nf_config_${src}.yaml"
rm ./nf_config_${src}.yaml
SBATCH_EOF

done
