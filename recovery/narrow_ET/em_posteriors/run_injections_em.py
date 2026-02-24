import os

import numpy as np
import pandas as pd
from sklearn import model_selection
import jax
import jax.numpy as jnp
import scipy.integrate as integrate

from fiesta.inference.prior import Uniform, Constraint, Sine, UniformSourceFrame
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import FluxModel, CombinedSurrogate
from fiesta.inference.injection import InjectionSurrogate


#########
# MODEL #
#########

FILTERS = ["radio-1.4GHz", "2massj", "besselli", "bessellv", "bessellux", "X-ray-1keV"]
FILTERS_KN = ["2massj", "besselli", "bessellv", "bessellux"]
detection_limit = {"radio-1.4GHz": 23.0, "2massj": 29.5, "besselli": 29.5, "bessellv": 29.5, "bessellux": 29.5, "X-ray-1keV": 39.0}

model_KN = FluxModel(name="Bu2026_MLP",
                      filters = FILTERS_KN)

model_afterglow = FluxModel(name="pbag_gaussian_CVAE",
                            filters=FILTERS)

model = CombinedSurrogate(models=[model_KN, model_afterglow], sample_times= jnp.geomspace(0.2, 2000, 200))


#########
# PRIOR #
#########


def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

KN_prior = [
            Uniform(xmin=0., xmax=np.pi/2, naming=["inclination_EM"]),
            Uniform(xmin=-4.0, xmax=-1.3, naming=["log10_mej_dyn"]),
            Uniform(xmin=0.12, xmax=0.35, naming=["v_ej_dyn"]),
            Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
            Uniform(xmin=-4., xmax=-0.55, naming=["log10_mej_wind"]),
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
             Constraint(xmin = 0., xmax = 1., naming=["epsilon_tot"])]


prior = ConstrainedPrior([*KN_prior, *GRB_prior], conversion_function)


################
# SAMPLING     #
################

def afterglow_peak(param_dict):
    times, mags = model_afterglow.predict(param_dict)
    return times[mags["radio-1.4GHz"].argmax()]

def main():

    events = pd.read_csv("../events.dat", sep=" ")
    rng_key = jax.random.PRNGKey(567893127)
    
    for j in range(0, events.shape[0]):
                
        param_dict = events.iloc[j].to_dict()
        param_dict.update(dict(alphaWing=2., p=2.15, log10_epsilon_e=-1., log10_epsilon_B=-3., Gamma0=500))
        param_dict["log10_E0"] = param_dict.pop("log10_Ekin_iso")

        if param_dict["afterglow_detected"]:
            filters = FILTERS
            tpeak = afterglow_peak(param_dict)
            tmax = max(14., 1.5 * tpeak)
            systematics_file = "./systematics_file_kn.yaml"
        else:
            filters = FILTERS_KN
            tmax = 14.
            systematics_file = "./systematics_file_kn.yaml"

        injection = InjectionSurrogate(model=model,
                                       filters=filters,
                                       trigger_time=param_dict['trigger_time'],
                                       tmin=0.5,
                                       tmax=tmax,
                                       N_datapoints=30,
                                       error_budget=0.1,
                                       nondetections=True,
                                       detection_limit=detection_limit)
        injection.create_injection(param_dict)
    
        likelihood = EMLikelihood(model,
                                  injection.data,
                                  tmin=0.5,
                                  tmax = 14.0,
                                  trigger_time=param_dict["trigger_time"],
                                  detection_limit = None,
                                  fixed_params={"redshift": param_dict["redshift"]}
                                  )
        
        # Save for postprocessing
        outdir = f"./source_{j}"

        fiesta = Fiesta(likelihood,
                        prior,
                        systematics_file=systematics_file,
                        n_chains = 200,
                        n_loop_training = 7,
                        n_loop_production = 3,
                        num_layers = 4,
                        hidden_size = [64, 64],
                        n_epochs = 20,
                        n_local_steps = 50,
                        n_global_steps = 200,
                        outdir=outdir)
        
        rng_key, subkey = jax.random.split(rng_key)
        fiesta.sample(subkey)
        fiesta.save_results()
        fiesta.plot_lightcurves()
        fiesta.plot_corner(truths=injection.injection_dict)


if __name__ == "__main__":
    main()
