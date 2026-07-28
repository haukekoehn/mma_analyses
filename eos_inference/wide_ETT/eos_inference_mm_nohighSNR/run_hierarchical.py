import os

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import pandas as pd

from bilby.gw.prior import UniformSourceFrame

from jesterTOV.inference.config.parser import load_config
from jesterTOV.inference.run_inference import setup_transform, create_sampler, run_sampling, setup_prior

from jesterTOV.inference.likelihoods import MultiMessengerLikelihood, CombinedLikelihood, ConstraintEOSLikelihood, RadioTimingLikelihood
from jesterTOV.inference.population.populations import massratiopowerlaw_logpdf


config = load_config("./config.yaml")
outdir = config.sampler.output_dir
os.makedirs(outdir, exist_ok=True)

events = pd.read_csv("../events.dat", sep=" ")
key = jax.random.key(98)

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
        dL = sample["luminosity_distance"]
        return jnp.log(jnp.interp(dL, bilby_dL_prior.xx, bilby_dL_prior.yy))
    
    def logprior_em(sample):
        dL = sample["luminosity_distance"]
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
    
    if src in [17,26,13]:
        continue
    logprior_gw, logprior_em = get_logprior(src)
    key, subkey = jax.random.split(key)
    likelihood = MultiMessengerLikelihood(
        event_name = f"source_{src}",
        posterior_gw = f"../gw_posteriors/source_{src}/posterior_mm.npz",
        conditional_flow_gw = f"../gw_nfs/source_{src}/cnf_mm",
        mass_model_logpdf=massratiopowerlaw_logpdf,
        flow_em = f"../../../cosmo_inference/wide_ETT/em_posteriors/source_{src}/nf",
        use_em=True,
        logprior_gw = logprior_gw,
        logprior_em = logprior_em,
        N_masses_evaluation=1000,
        N_masses_batch_size = 100,
        key = subkey,
    )
    likelihoods.append(likelihood)

likelihood = CombinedLikelihood(likelihoods)

"""
########
# TEST #
########


samples = prior.sample(jax.random.key(42), 1000)
import h5py
posterior = {}
with h5py.File("outdir/results.h5") as f:
    for key in f["posterior"]["parameters"].keys():
        posterior[key] = f["posterior"]["parameters"][key][:]
    posterior["log_prob"] = f["posterior"]["log_prob"][:]


def check_samples_for_nan():
    likelihood_fn = jax.jit(likelihood.evaluate)
    for j in range(1000):
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
    import numpy as np
    from copy import deepcopy
    m_eos, r_eos, l_eos = np.loadtxt("../../../eos/RMF3_MRL.dat", unpack=True)
    
    ind = posterior["log_prob"].argmax()
    best_posterior = {key: val[ind] for key, val in posterior.items()}
    best_posterior = transform.forward(best_posterior)

    truth = deepcopy(best_posterior)
    truth.update(dict(masses_EOS=m_eos, radii_EOS=r_eos, Lambdas_EOS=l_eos))
    #truth = transform.forward(truth)

    for j in range(0,events.shape[0]):
        
        total_mass = np.sum(events.loc[j, ["mass_1_source", "mass_2_source"]])
        pc_best = total_mass  >= best_posterior["masses_EOS"].max() * best_posterior["k_coll"]
        pc_truth = total_mass >= truth["masses_EOS"].max() * best_posterior["k_coll"] 


        logl_bestposterior= likelihoods[j+2].evaluate(best_posterior)
        logl_truth = likelihoods[j+2].evaluate(truth)

        print(j, events.loc[j, "redshift"], pc_best, pc_truth, logl_bestposterior, logl_truth, logl_truth - logl_bestposterior)

    breakpoint()


check_truth()
exit()
"""
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
