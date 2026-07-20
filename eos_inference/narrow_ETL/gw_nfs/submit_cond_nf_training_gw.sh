#!/bin/bash

nsrc=$(wc -l < ../events.dat)-1

if [ $# -eq 0 ]; then
    sources=$(seq 0 $((nsrc-1)))
else
    sources=$(echo "$1" | tr ',' ' ')
fi

for src in $sources;
do

    sbatch <<SBATCH_EOF
#!/bin/bash

#SBATCH -J narrow_ETL_cnf_gw_${src}
#SBATCH -o ./source_${src}/cnf_gw/log_cnf
#SBATCH -e ./source_${src}/cnf_gw/log_cnf

#SBATCH --partition cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:00:00

eval "\$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta


cat <<CONFIG_EOF > cnf_config_gw_${src}.yaml

# Configuration for training normalizing flow
# Updated to use new default hyperparameters (PR #55)
posterior_file: ../gw_posteriors/source_${src}/posterior.npz
output_dir: ./source_${src}/cnf_gw_nodist

# Parameter selection
parameter_names: ["lambda_1", "lambda_2"]

# Training parameters
num_epochs: 1000
learning_rate: 0.0001
max_patience: 500
batch_size: 128
val_prop: 0.15
seed: 465

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
cond_dim: 2
cond_parameter_names: ["mass_1_source", "mass_2_source"]
CONFIG_EOF

train_jester_flow "./cnf_config_gw_${src}.yaml"
rm ./cnf_config_gw_${src}.yaml
SBATCH_EOF

done
