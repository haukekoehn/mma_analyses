import os

import jax
import pandas as pd

from jesterTOV.inference.config.parser import load_config
from jesterTOV.inference.run_inference import setup_transform, create_sampler, run_sampling, setup_prior

from jesterTOV.inference.likelihoods import MultiMessengerLikelihood, CombinedLikelihood, ConstraintEOSLikelihood, RadioTimingLikelihood
from jesterTOV.inference.population.populations import RecycledBinary

config = load_config("./config_mm.yaml")
outdir = config.sampler.output_dir
os.makedirs(outdir, exist_ok=True)

events = pd.read_csv("../../events.dat", sep=" ")
events["zeta"] = 10**(events["log10_mej_wind"] - events["log10_mdisk"])

key = jax.random.key(10662)

#########
# Prior #
#########

prior = setup_prior(config)


##############
# Likelihood #
##############

transform = setup_transform(config, prior, keep_names=['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'alpha', 'm_min', 'm_max', 'k_coll'], fixed_params=dict(E_sat=-16.))

likelihoods = []
likelihoods.append(ConstraintEOSLikelihood())
likelihoods.append(RadioTimingLikelihood(psr_name="J1614", mean=1.94, std=0.06))
for src in range(0, 20):
    likelihood = MultiMessengerLikelihood(event_name = f"source_{src}",
                                          dir_gw = f"../../gw_posteriors/source_{src}/nf_mm",
                                          dir_em = f"../../em_posteriors/source_{src}/nf",
                                          population=RecycledBinary,
                                          pop_random_key=key,
                                          zeta=events.loc[src, "zeta"],
                                          N_masses_evaluation=4000,
                                          N_masses_batch_size=500)
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