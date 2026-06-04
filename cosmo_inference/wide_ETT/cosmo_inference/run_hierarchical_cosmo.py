import os

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import pandas as pd

from bilby.gw.prior import UniformSourceFrame

from jesterTOV.inference.config.parser import load_config
from jesterTOV.inference.run_inference import setup_transform, create_sampler, run_sampling, setup_prior

from jesterTOV.inference.likelihoods import CosmoMultiMessengerLikelihood, CombinedLikelihood, ConstraintEOSLikelihood, RadioTimingLikelihood
from jesterTOV.inference.population.populations import massratiopowerlaw_logpdf
config = load_config("./config.yaml")
outdir = config.sampler.output_dir
os.makedirs(outdir, exist_ok=True)

events = pd.read_csv("../../../eos_inference/wide_ETT/events.dat", sep=" ")
events["chirp_mass"] = (events["mass_1"]*events["mass_2"])**0.6 / (events["mass_1"] + events["mass_2"])**0.2

key = jax.random.key(8902)

#########
# Prior #
#########

prior, _ = setup_prior(config)

#####################
# Prior subtraction #
#####################

bilby_dL_prior = UniformSourceFrame(name='luminosity_distance', minimum=40, maximum=8e3)

def get_logprior(src: int):

    def logprior_gw(sample):
        dL = sample[4]
        return jnp.log(jnp.interp(dL, bilby_dL_prior.xx, bilby_dL_prior.yy))
    
    def logprior_em(sample):
        dL = sample[1]
        return jnp.log(jnp.interp(dL, bilby_dL_prior.xx, bilby_dL_prior.yy))
    
    return logprior_gw, logprior_em


##############
# Likelihood #
##############

transform = setup_transform(config, prior, fixed_params=dict(E_sat=-16.))

likelihoods = []
likelihoods.append(ConstraintEOSLikelihood())
likelihoods.append(RadioTimingLikelihood(psr_name="J1614", mean=1.937, std=0.014))
for src in range(events.shape[0]):

    logprior_gw, logprior_em = get_logprior(src)
    key, subkey = jax.random.split(key)
    likelihood = CosmoMultiMessengerLikelihood(
        event_name = f"source_{src}",
        dir_gw = f"../gw_posteriors/source_{src}/nf",
        dir_gw_cond = f"../gw_posteriors/source_{src}/cnf",
        dir_em = f"../em_posteriors/source_{src}/nf",
        population_logpdf = massratiopowerlaw_logpdf,
        N_eval = 1000,
        redshift_mean = events.loc[src, "redshift_measured"],
        redshift_sigma = 0.01 * events.loc[src, "redshift"],
        logprior_gw = logprior_gw,
        logprior_em = logprior_em,
        N_masses_batch_size = 100,
        key = subkey
    )
    likelihoods.append(likelihood)

likelihood = CombinedLikelihood(likelihoods)


###########
# Sampler #
###########


sampler = create_sampler(config=config.sampler,
                         prior=prior,
                         likelihood=likelihood,
                         likelihood_transforms=[transform],
                         seed=config.seed)

result = run_sampling(sampler, config.seed, config, outdir)
result.add_eos_from_transform(
        transform=transform,  # Use the same transform from sampling
        n_eos_samples=config.sampler.n_eos_samples,
        batch_size=config.sampler.log_prob_batch_size,
    )

result.save(os.path.join(outdir, "results.h5"))
