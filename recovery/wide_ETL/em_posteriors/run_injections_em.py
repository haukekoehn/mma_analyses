import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn import model_selection
import jax
import jax.numpy as jnp
import scipy.integrate as integrate

from fiesta.inference.prior import Uniform, Constraint, Sine, UniformSourceFrame, Normal, ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import FluxModel, CombinedSurrogate
from fiesta.inference.injection import InjectionSurrogate


#########
# MODEL #
#########

FILTERS = ["radio-1.4GHz", "2massj", "besselli", "bessellv", "bessellux", "X-ray-1keV"]
FILTERS_KN = ["2massj", "besselli", "bessellv", "bessellux"]
detection_limit = {"radio-1.4GHz": 23.0, "2massj": 29.5, "besselli": 29.5, "bessellv": 29.5, "bessellux": 26, "X-ray-1keV": 39.0}

model_KN = FluxModel(name="Bu2026_MLP",
                      filters = FILTERS_KN)

model_afterglow = FluxModel(name="pbag_gaussian_CVAE",
                            filters=FILTERS)


#########
# PRIOR #
#########


def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

KN_prior = [
            Sine(xmin=0., xmax=np.pi/2, naming=["inclination_EM"]),
            Uniform(xmin=-4.0, xmax=-1.3, naming=["log10_mej_dyn"]),
            Uniform(xmin=0.12, xmax=0.35, naming=["v_ej_dyn"]),
            Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
            Uniform(xmin=-4, xmax=-0.55, naming=["log10_mej_wind"]),
            Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
            Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"]),
            UniformSourceFrame(dmin=40.0, dmax=8000.0, naming=["luminosity_distance"])
]

GRB_prior = [Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0']),
             Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore']),
             Uniform(xmin = 0.2, xmax = 3.5, naming= ["alphaWing"]),
             Constraint(xmin = 0, xmax = np.pi/2, naming = ["thetaWing"]),
             Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0']),
             Uniform(xmin=2.01, xmax=3.0, naming=['p']),
             Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e']),
             Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B']),
             Uniform(xmin=100, xmax=1000., naming=['Gamma0']),
             Uniform(xmin=0.3, xmax=1., naming=["em_syserr"]),
             Constraint(xmin = 0., xmax = 1., naming=["epsilon_tot"])
]

################
# SAMPLING     #
################

def enforce_surrogate_param_range(params, models):

    if not isinstance(models, Iterable):
        models = [models]

    for model in models:
        for p, (pmin, pmax, _) in model.parameter_distributions.items():
            params[p] = jnp.maximum(pmin, params[p])
            params[p] = jnp.minimum(pmax, params[p])

    return params

def afterglow_peak(param_dict):
    times, mags = model_afterglow.predict(param_dict)
    return times[mags["radio-1.4GHz"].argmax()]

def analyze_event(j, param_dict, rng_key):

    param_dict.update(dict(alphaWing=2., p=2.15, log10_epsilon_e=-1., log10_epsilon_B=-3., Gamma0=500))
    param_dict["log10_E0"] = param_dict.pop("log10_Ekin_iso")
    redshift = param_dict["redshift"]
    param_dict = enforce_surrogate_param_range(param_dict, [model_KN, model_afterglow])


    model = CombinedSurrogate(models=[model_KN, model_afterglow], 
                              sample_times= jnp.geomspace(0.9 * (1+redshift) * 0.2, 1.1 * (1+redshift) *  2000, 200))

    if param_dict["afterglow_detected"]:
        filters = FILTERS
        tpeak = afterglow_peak(param_dict)
        N_datapoints = 50
    else:
        filters = FILTERS_KN
        N_datapoints = 50


    injection = InjectionSurrogate(model=model,
                                    filters=filters,
                                    trigger_time=param_dict['trigger_time'],
                                    tmin=0.5,
                                    tmax=8.,
                                    N_datapoints=N_datapoints,
                                    error_budget=0.1,
                                    nondetections=False,
                                    detection_limit=detection_limit)
    injection.create_injection(param_dict)
            
    likelihood = EMLikelihood(model,
                              injection.data,
                              data_tmin=0.5,
                              data_tmax = 10.,
                              trigger_time=param_dict["trigger_time"],
                              detection_limit = None,
                            )
        
    # Save for postprocessing
    outdir = f"./source_{j}"

    prior = ConstrainedPrior([*KN_prior, Normal(mu=param_dict["redshift"], sigma=0.01*param_dict["redshift"], naming=["redshift"]), *GRB_prior], conversion_function)

    fiesta = Fiesta(likelihood,
                    prior,
                    n_chains = 500,
                    n_training_loops = 7,
                    n_production_loops = 3,
                    rq_spline_n_layers = 4,
                    n_epochs = 20,
                    n_local_steps = 50,
                    n_global_steps = 200,
                    outdir=outdir)
                
    fiesta.sample(rng_key)
    fiesta.save_results()
    fiesta.plot_lightcurves()
    fiesta.plot_corner(truths=injection.injection_dict)
    injection.write_to_file(f"./source_{j}/light_curve.dat")    



def main():

    events = pd.read_csv("../events.dat", sep=" ")
    rng_key = jax.random.PRNGKey(32903)
    for j in range(0, events.shape[0]):
        rng_key, sub_key = jax.random.split(rng_key)
        try:
            analyze_event(j, events.iloc[j].to_dict(), sub_key)
        except:
            print("\n \n")
            print(f"DID NOT WORK FOR SOURCE {j}.")
            print("\n \n")
            continue
        


if __name__ == "__main__":
    main()
