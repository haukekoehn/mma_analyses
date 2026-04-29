import os

import jax
import jax.numpy as jnp
import pandas as pd
import h5py

import bilby
from scipy.integrate import cumulative_simpson

from jesterTOV.inference.config.parser import load_config
from jesterTOV.inference.run_inference import setup_transform, create_sampler, run_sampling, setup_prior

from jesterTOV.inference.likelihoods import PopulationGWLikelihood, CombinedLikelihood, ConstraintEOSLikelihood, RadioTimingLikelihood
from jesterTOV.inference.population.populations import RecycledBinary

events = pd.read_csv("../../events.dat", sep=" ")
events["chirp_mass"] = (events["mass_1"]*events["mass_2"])**0.6 / (events["mass_1"] + events["mass_2"])**0.2

config = load_config("./config_gw.yaml")
outdir = config.sampler.output_dir
os.makedirs(outdir, exist_ok=True)

#########
# Prior #
#########

prior, _ = setup_prior(config)

#####################
# Prior subtraction #
#####################

# corresponds to 40 and 8000 Mpc
redshift_prior = bilby.gw.prior.UniformSourceFrame(name='redshift', minimum=0.008966150804313566, maximum=1.1421224848725744)
z_arr = redshift_prior.xx
integrand_arr = (1+z_arr)**2 * redshift_prior.yy
antiderivative_arr = cumulative_simpson(x=z_arr, y=integrand_arr, initial=0)

z_arr = jnp.array(z_arr)
antiderivative_arr = jnp.array(antiderivative_arr)

def integral_over_redshift_prior(zmin, zmax):
    """
    This function here implements the integral
    over \pi(z) (1+z)^2 from zmin to zmax
    for the UniformSourceFrame prior we used during the GW inference.
    """
    upper = jnp.interp(zmax, z_arr, antiderivative_arr)
    lower = jnp.interp(zmin, z_arr, antiderivative_arr)
    return upper - lower

def get_logprior(src: int):

    chirp_mass_detector = events.loc[src, "chirp_mass"]

    def logprior_fn(m1, m2):

        chirp_mass_source = (m1*m2)**0.6 / (m1+m2)**0.2
        q = m2/m1

        zmin = (chirp_mass_detector - 0.01) / chirp_mass_source -1
        zmax = (chirp_mass_detector + 0.01) / chirp_mass_source - 1
        zmin = jnp.maximum(z_arr[0], zmin)
        zmax = jnp.minimum(z_arr[-1], zmax)

        constraint = 0.4 <= q 
        constraint &= q <= 1.0
        constraint &= zmin < zmax # can happen if both zmin and zmax are over the limit

        integral = integral_over_redshift_prior(zmin, zmax)
        integral = jnp.maximum(1e-6, integral)


        logprior = jnp.log(integral)

        return jax.lax.select(constraint, logprior, 0.0)
    
    return logprior_fn


##############
# Likelihood #
##############

config.population.N_masses_evaluation = 10_000
transform = setup_transform(config, prior, fixed_params=dict(E_sat=-16.))

likelihoods = []
likelihoods.append(ConstraintEOSLikelihood())
likelihoods.append(RadioTimingLikelihood(psr_name="J1614", mean=1.937, std=0.014))

for src in range(0, events.shape[0]):
    logprior_m1m2 = get_logprior(src)
    likelihood = PopulationGWLikelihood(event_name = f"source_{src}",
                                        model_dir = f"../../gw_posteriors/source_{src}/nf",
                                        logprior_m1m2=logprior_m1m2,
                                        N_masses_batch_size=200)
    likelihoods.append(likelihood)
    logprior_m1m2=logprior_m1m2

likelihood = CombinedLikelihood(likelihoods)

########
# Test #
########

import pandas as pd
import matplotlib.pyplot as plt

samples = prior.sample(jax.random.key(42), 1000)

def check_samples_for_nan():
    likelihood_fn = jax.jit(likelihood.evaluate)
    for j in range(4000):
        params = {key: samples[key][j] for key in samples.keys()}
        params = transform.forward(params)
        logl = likelihood_fn(params)
        print(j, logl)

def check_specific_sample(j):
    
    params = {key: samples[key][j] for key in samples.keys()}
    params = transform.forward(params)
    for j in range(len(likelihoods)):
        logl = likelihoods[j].evaluate(params)
        print(j, logl)

def check_truth():

    with h5py.File("outdir_gw/results.h5") as f:
        logprob = f["posterior"]["log_prob"][:]

        j = logprob.argmax()
        params = {key: f["posterior"]["parameters"][key][j] for key in f["posterior"]["parameters"].keys()}
    
    params = transform.forward(params)
    for j in range(len(likelihoods)):
        logl = likelihoods[j].evaluate(params)
        print(j, logl)

check_truth()