import os

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import pandas as pd
import h5py


from jesterTOV.inference.config.parser import load_config
from jesterTOV.inference.run_inference import setup_transform, create_sampler, run_sampling, setup_prior

from jesterTOV.inference.likelihoods import MultiMessengerLikelihood, CombinedLikelihood, ConstraintEOSLikelihood, RadioTimingLikelihood
from jesterTOV.inference.population.populations import RecycledBinary

config = load_config("./config_mm.yaml")
outdir = config.sampler.output_dir
os.makedirs(outdir, exist_ok=True)

events = pd.read_csv("../../events.dat", sep=" ")
events["chirp_mass"] = (events["mass_1"]*events["mass_2"])**0.6 / (events["mass_1"] + events["mass_2"])**0.2


#########
# Prior #
#########

prior, _ = setup_prior(config)

#####################
# Prior subtraction #
#####################

def antiderivative(z, mu, sigma):
    """
    This function here implements the integral
    over \pi(z) (1+z)^2 from zmin to zmax
    for the normal redshift prior used for multi-messenger events.
    """

    term1 = jnp.sqrt(jnp.pi) / jnp.sqrt(2) * sigma  * ( sigma**2 + (mu+1)**2) * erf((z-mu)/(jnp.sqrt(2)*sigma))
    term2 = sigma**2 * (z + mu + 2) * jnp.exp(-0.5 * (z-mu)**2 / sigma**2)

    return term1 - term2


def get_logprior(src: int):

    chirp_mass_detector = events.loc[src, "chirp_mass"]
    redshift_mean = events.loc[src, "redshift"]
    redshift_sigma = 0.01 * redshift_mean

    def logprior_fn(m1, m2):

        chirp_mass_source = (m1*m2)**0.6 / (m1+m2)**0.2
        q = m2/m1

        zmin = (chirp_mass_detector - 0.01) / chirp_mass_source -1
        zmax = (chirp_mass_detector + 0.01) / chirp_mass_source - 1


        constraint = 0.4 <= q 
        constraint &= q <= 1.0

        upper = antiderivative(zmax, redshift_mean, redshift_sigma)
        lower = antiderivative(zmin, redshift_mean, redshift_sigma)
        integral = upper - lower
        integral = jnp.where(integral<1e-6, 1, integral)
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
    likelihood = MultiMessengerLikelihood(event_name = f"source_{src}",
                                          dir_gw = f"../../gw_posteriors/source_{src}/nf_mm",
                                          dir_em = f"../../em_posteriors/source_{src}/posterior.npz",
                                          logprior_m1m2=logprior_m1m2,
                                          N_masses_batch_size=200)
    likelihoods.append(likelihood)

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

    with h5py.File("outdir_mm/results.h5") as f:
        logprob = f["posterior"]["log_prob"][:]

        j = logprob.argmax()
        params = {key: f["posterior"]["parameters"][key][j] for key in f["posterior"]["parameters"].keys()}
    
    params = transform.forward(params)
    for j in range(len(likelihoods)):
        logl = likelihoods[j].evaluate(params)
        print(j, logl)

check_truth()