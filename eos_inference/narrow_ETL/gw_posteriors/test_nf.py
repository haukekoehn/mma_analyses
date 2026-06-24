import os

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.special import logsumexp

from jesterTOV.inference.flows import Flow, ConditionalFlow
from jesterTOV.inference.flows.train_flow import load_posterior, plot_corner

rng_key = jax.random.key(235)

parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2", "luminosity_distance", "cos_theta_jn"]
data, meta_data = load_posterior("./source_0/posterior.npz", parameter_names)
data[:, 2] = 353
data[:, 3] = 548

cflow = ConditionalFlow.from_directory("./source_0/nfs_mm/cnf")
flow = Flow.from_directory("./source_0/nfs_mm/nf")
flow_cond = Flow.from_directory("./source_0/nfs_mm/nf_cond")

m1_test = 1.5
cnf_logprobs = cflow.log_prob(data[:, 2:4], y=data[:, [0,1,4,5]])
flow_logprobs = flow.log_prob(data)
flow_logprobs_cond = flow_cond.log_prob(data[:, [0,1,4,5]])
conditional_logprobs = flow_logprobs - flow_logprobs_cond
print(conditional_logprobs, cnf_logprobs)

plt.scatter(conditional_logprobs, cnf_logprobs)
plt.savefig("test.pdf")


print(np.mean(logsumexp(conditional_logprobs)))
print(np.mean(logsumexp(cnf_logprobs)))
