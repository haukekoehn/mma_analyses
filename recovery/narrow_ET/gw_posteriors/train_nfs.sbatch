#!/bin/bash
nsrc=$(ls -lah . | grep "source" | wc -l)

for ((src=0; src<nsrc; src++));
do

    sbatch <<SBATCH_EOF
#!/bin/bash

#SBATCH -J nf_narrow_ET_${src}
#SBATCH -o ./source_${src}/log_nf
#SBATCH -e ./source_${src}/err_nf

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=10:00:00

eval "\$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta


cat <<CONFIG_EOF > nf_config_${src}.yaml

# Configuration for training normalizing flow
# Updated to use new default hyperparameters (PR #55)
posterior_file: ./source_${src}/posterior.npz
output_dir: ./source_${src}/nf

# Parameter selection
parameter_names: ["chirp_mass", "mass_ratio", "lambda_1", "lambda_2", "theta_jn", "luminosity_distance"]

# Training parameters
num_epochs: 3000
learning_rate: 0.0001
max_patience: 500
batch_size: 128
val_prop: 0.2
seed: 0

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
max_samples: 50000
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
