import os

import numpy as np
import matplotlib.pyplot as plt
import jax

from jesterTOV.inference.flows import Flow
from jesterTOV.inference.flows.train_flow import load_posterior, plot_corner

rng_key = jax.random.key(235)

parameter_names = ["chirp_mass", "mass_ratio", "lambda_1", "lambda_2", "theta_jn", "luminosity_distance"]

data, meta_data = load_posterior("./source_0/posterior.npz", parameter_names)
n_plot_samples = min(10_000, data.shape[0])

flow = Flow.from_directory("./source_0/nf")
flow_samples = np.array(flow.sample(rng_key, shape=(n_plot_samples,) ))

plot_corner(data, flow_samples, "./source_0/nf/figures/corner.png", parameter_names)

