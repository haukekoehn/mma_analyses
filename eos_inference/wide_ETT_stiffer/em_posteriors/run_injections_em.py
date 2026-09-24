import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn import model_selection
import jax
import jax.numpy as jnp
import scipy.integrate as integrate

from fiesta.inference.prior import Uniform, Constraint, Sine, UniformSourceFrame, Normal, ConstrainedPrior, LogUniform
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import FluxModel, CombinedSurrogate
from fiesta.inference.injection import InjectionSurrogate

#####PARSER########

parser = argparse.ArgumentParser()

parser.add_argument("--source", help="integer, which source from events.dat to use", required=True, type=int)
parser.add_argument("--outdir", help="outdir for the result files", required=True)

parser.add_argument("--random-seed", help="random seed", default=4576892, type=int)

###################

#####PARSING########

args = parser.parse_args()

###################


def main():

    events = pd.read_csv("../events.dat", sep=" ")
    rng_key = jax.random.PRNGKey(args.random_seed)
    analyze_event(args.source, events.iloc[args.source].to_dict(), rng_key)


############
# UTILS    #
############

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"])
    converted_sample["inclination_EM"] = jnp.arccos(converted_sample["cos_inclination_EM"])
    return converted_sample

def enforce_surrogate_param_range(params, models):

    if not isinstance(models, Iterable):
        models = [models]

    for model in models:
        for p, (pmin, pmax, _) in model.parameter_distributions.items():
            params[p] = jnp.maximum(pmin, params[p])
            params[p] = jnp.minimum(pmax, params[p])

    return params

def add_cos_theta_jn(file):
    posterior = dict(np.load(file, allow_pickle=True))
    posterior["cos_theta_jn"] = posterior["cos_inclination_EM"] * np.random.choice([-1, 1], size=posterior["cos_inclination_EM"].shape[0])
    np.savez(file, **posterior)

def analyze_event(j, param_dict, rng_key):

    #########
    # MODEL #
    #########

    FILTERS = ["radio-1.4GHz", "2massj", "besselli", "bessellv", "bessellux", "X-ray-1keV"]
    FILTERS_KN = ["2massj", "besselli", "bessellv", "bessellux"]
    detection_limit = {"radio-1.4GHz": 23.0, "2massj": 29.5, "besselli": 29.5, "bessellv": 29.5, "bessellux": 26, "X-ray-1keV": 39.0}
    redshift = param_dict["redshift"]

    model_KN = FluxModel(name="Bu2026_MLP",
                          filters = FILTERS_KN)

    model_afterglow = FluxModel(name="pbag_gaussian_CVAE",
                                filters=FILTERS)

    model_combined = CombinedSurrogate(models=[model_KN, model_afterglow], 
                              sample_times= jnp.geomspace(0.9 * (1+redshift) * 0.2, 1.1 * (1+redshift) *  2000, 200))

    param_dict.update(dict(alphaWing=2., p=2.15, log10_epsilon_e=-1., log10_epsilon_B=-3., Gamma0=500))
    param_dict["log10_E0"] = param_dict.pop("log10_Ekin_iso")
    param_dict["cos_inclination_EM"] = np.cos(param_dict["inclination_EM"])
    param_dict = enforce_surrogate_param_range(param_dict, [model_KN, model_afterglow])


    if param_dict["has_grb"]:
        filters = FILTERS
        N_datapoints = 60
        tmax=10.
        model = model_combined
    else:
        filters = FILTERS_KN
        N_datapoints = 30
        tmax=10.
        model = model_KN


    injection = InjectionSurrogate(model=model,
                                    filters=filters,
                                    trigger_time=param_dict['trigger_time'],
                                    tmin=0.5,
                                    tmax=tmax,
                                    N_datapoints=N_datapoints,
                                    error_budget=0.1,
                                    nondetections=False,
                                    detection_limit=detection_limit)
    injection.create_injection(param_dict)
    

    #########
    # PRIOR #
    #########

    KN_prior = [
            Uniform(xmin=0.0, xmax=1.0, naming=["cos_inclination_EM"]),
            Uniform(xmin=-4.0, xmax=-1.3, naming=["log10_mej_dyn"]),
            Uniform(xmin=0.12, xmax=0.35, naming=["v_ej_dyn"]),
            Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
            Uniform(xmin=-4., xmax=-0.55, naming=["log10_mej_wind"]),
            Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
            Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"]),
            UniformSourceFrame(dmin=40.0, dmax=8000.0, naming=["luminosity_distance"])
    ]

    GRB_prior = [Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0']),
                 LogUniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore']),
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
            
    likelihood = EMLikelihood(model_combined,
                              injection.data,
                              data_tmin=0.5,
                              data_tmax = tmax,
                              trigger_time=param_dict["trigger_time"],
                              detection_limit = None,
                              conversion_function=conversion_function,
                            )
        
    # Save for postprocessing
    outdir = f"./source_{j}"

    prior = ConstrainedPrior([*KN_prior, Normal(mu=param_dict["redshift_measured"], sigma=0.01*param_dict["redshift"], naming=["redshift"]), *GRB_prior], conversion_function)

    fiesta = Fiesta(likelihood,
                    prior,
                    n_chains = 500,
                    n_training_loops = 7,
                    n_production_loops = 3,
                    rq_spline_n_layers = 4,
                    n_epochs = 20,
                    n_local_steps = 70,
                    n_global_steps = 200,
                    outdir=outdir)
                
    fiesta.sample(rng_key)
    fiesta.save_results()
    add_cos_theta_jn(f"./source_{j}/posterior.npz")
    fiesta.plot_lightcurves()
    fiesta.plot_corner(truths=injection.injection_dict)
    injection.write_to_file(f"./source_{j}/light_curve.dat")    
        


if __name__ == "__main__":
    main()
