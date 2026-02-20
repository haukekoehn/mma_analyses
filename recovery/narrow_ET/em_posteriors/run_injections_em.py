import os

import numpy as np
import jax
import jax.numpy as jnp
import scipy.integrate as integrate

from fiesta.inference.prior import Uniform, Constraint
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import FluxModel
from fiesta.inference.injection import InjectionKN

def hdi_quantile_compute(samples, injected_value):
    hist, bins = np.histogram(samples, bins='auto', density=True)
    x = (bins[1:]+bins[:-1])/2
    density = np.interp(injected_value, x , hist)
    hist[hist<density] = 0.
    return integrate.simpson(x=x, y=hist)



#########
# MODEL #
#########
FILTERS = ["2massj", "besselli", "bessellg", "bessellux"]
model = FluxModel(name="Bu2026_MLP",
                      filters = FILTERS)


#########
# PRIOR #
#########

inclination_EM = Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM'])
log10_mej_dyn = Uniform(xmin=-3.0, xmax=-1.30, naming=["log10_mej_dyn"])
v_ej_dyn = Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"])
Ye_dyn = Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"])
log10_mej_wind = Uniform(xmin=-2.0, xmax=-0.89, naming=["log10_mej_wind"])
v_ej_wind = Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"])
Ye_wind = Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
sys_err = Uniform(xmin=0.3, xmax=1.0, naming=["sys_err"])  #used to be 0.5

prior_list = [inclination_EM, 
              log10_mej_dyn,
              v_ej_dyn,
              Ye_dyn,
              log10_mej_wind,
              v_ej_wind,
              Ye_wind,
              sys_err]

prior = ConstrainedPrior(prior_list)

################
# LIKELIHOOD & #
# SAMPLING     #
################

rng_key = jax.random.PRNGKey(3551)
trigger_time = 58849.

injection = InjectionKN(filters=FILTERS, 
                        N_datapoints=50, 
                        error_budget=0.1, 
                        tmin=0.5, 
                        tmax=20,
                        trigger_time=trigger_time,
                        detection_limit=24
                        )

injection_prior = ConstrainedPrior([Uniform(xmin=0., xmax=np.pi/2, naming=['inclination_EM']),
                                    Uniform(xmin=-3.0, xmax=-1.30, naming=["log10_mej_dyn"]),
                                    Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"]),
                                    Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
                                    Uniform(xmin=-2.0, xmax=-0.886, naming=["log10_mej_wind"]),
                                    Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
                                    Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
])
quantile_list = []
hdi_quantile_list = []
param_list = []

for j in range(0, 200):
    
    rng_key, subkey = jax.random.split(rng_key)
    param_dict = injection_prior.sample(subkey , n_samples=1)
    param_dict = {key: p.item() for key, p in param_dict.items()}
    param_dict["luminosity_distance"] = 40.0

    injection.create_injection(param_dict, file="/home/aya/work/hkoehn/fiesta/fiesta/surrogates/KN/training_data/Bu2025_raw_data.h5")
    param_dict = injection.injection_dict

    likelihood = EMLikelihood(model,
                              injection.data,
                              FILTERS,
                              tmin=0.5,
                              tmax = 15.0,
                              trigger_time=trigger_time,
                              detection_limit = None,
                              fixed_params={"luminosity_distance": 40.0, "redshift": 0.0},
                              )
    
    mass_matrix = jnp.eye(prior.n_dim)
    eps = 5e-3
    local_sampler_arg = {"step_size": mass_matrix * eps}
    
    # Save for postprocessing
    outdir = f"./outdir/"
    
    fiesta = Fiesta(likelihood,
                    prior,
                    n_chains = 1_000,
                    n_loop_training = 7,
                    n_loop_production = 3,
                    num_layers = 4,
                    hidden_size = [64, 64],
                    n_epochs = 20,
                    n_local_steps = 50,
                    n_global_steps = 200,
                    local_sampler_arg=local_sampler_arg,
                    outdir=outdir)
    
    fiesta.sample(jax.random.PRNGKey(42))

    state = fiesta.Sampler.get_sampler_state(training=False)
    chains = state["chains"]
    n_chains, n_steps, n_dim = jnp.shape(chains)
    samples = jnp.reshape(chains, (n_chains * n_steps, n_dim))

    quantiles = [jnp.sum(samples[:,j]<=param_dict[p])/(n_chains * n_steps)  for j, p in enumerate(prior.naming[:-1])]
    hdi_quantiles = [hdi_quantile_compute(samples[:,j], param_dict[p]) for j, p in enumerate(prior.naming[:-1])]

    hdi_quantile_list.append(hdi_quantiles)
    quantile_list.append(quantiles)
    param_list.append([param_dict[p] for p in prior.naming[:-1]])


np.savetxt("./outdir/hdi_quantiles.txt", np.array(hdi_quantile_list))
np.savetxt("./outdir/quantiles.txt", np.array(quantile_list))
np.savetxt("./outdir/params.txt", np.array(param_list)) 