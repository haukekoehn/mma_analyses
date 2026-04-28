import time
from multiprocessing import Pool
from functools import partial

import numpy as np
np.random.seed(1895)
import pandas as pd

from astropy.time import Time
from scipy.stats import stats

import bilby
from bilby.gw.conversion import luminosity_distance_to_redshift

from GWFish.modules.detection import Network
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.fishermatrix import sky_localization_percentile_factor
from GWFish.modules.utilities import get_snr

#from nmma.core.conversion import BNSEjectaFitting

#from fiesta.inference import FluxModel
#model_KN = FluxModel(name='Bu2026_MLP', filters=["lsstg", "lssti"])


m_val, r_val, l_val = np.loadtxt("../eos/RMF3_MRL.dat", unpack=True)
mtov = m_val.max()
r16 = np.interp(1.6, m_val, r_val)



def draw_BNSs(size: int):

    df = pd.DataFrame()
    df["mass_1_source"] = np.random.uniform(1, 2, size=size)
    df["mass_2_source"] = np.random.uniform(1, 2, size=size)

    df["mass_1_source"], df["mass_2_source"] = df[["mass_1_source", "mass_2_source"]].max(axis=1), df[["mass_1_source", "mass_2_source"]].min(axis=1)
    df["radius_1"] = np.interp(df["mass_1_source"], m_val, r_val)
    df["radius_2"] = np.interp(df["mass_2_source"], m_val, r_val)
    df["lambda_1"] = np.interp(df["mass_1_source"], m_val, l_val)
    df["lambda_2"] = np.interp(df["mass_2_source"], m_val, l_val)

    df["luminosity_distance"] = 100.0 # bilby.gw.prior.UniformSourceFrame(name="luminosity_distance", minimum=40, maximum=5000).sample(size)
    df["redshift"] = luminosity_distance_to_redshift(df["luminosity_distance"])

    df["mass_1"] = (1+df["redshift"]) * df["mass_1_source"]
    df["mass_2"] = (1+df["redshift"]) * df["mass_2_source"]

    # Observational parameters
    df = add_observational_BNS_parameters(df)

    return df

def add_observational_BNS_parameters(df):
    # observational parameters
    df["theta_jn"] = np.arccos(np.random.uniform(-1, 1, size=df.shape[0]))
    df["phase"] = np.random.uniform(0, 2*np.pi, size=df.shape[0])
    df["psi"] = np.random.uniform(0, np.pi, size=df.shape[0])
    df["ra"] = np.random.uniform(0, 2*np.pi, size=df.shape[0])
    df["dec"] = np.arcsin(np.random.uniform(-1, 1, size=df.shape[0]))
    df["geocent_time"] = np.random.uniform(Time("2050-01-01", scale="tcg").gps, Time("2050-12-31", scale="tcg").gps, size=df.shape[0])
    
    return df

def compute_chunk(df_chunk, network):
    cols = ["mass_1", "mass_2", "luminosity_distance", "theta_jn",
        "ra", "dec", "psi", "phase", "geocent_time"]
    
    events = df_chunk[cols]
    breakpoint()
    return get_snr(events, network, waveform_model="IMRPhenomXAS_NRTidalv3")["network"].to_numpy()

def parallel_snr(df, detectors: list[str], nprocs=1):
        
        chunks = np.array_split(df, nprocs)
        network = Network(detectors)
        #compute = partial(compute_chunk, network=network)

        compute_chunk(df, network=network)
        #with Pool(nprocs) as pool:
        #    results = pool.map(compute, chunks)
        
        result = np.concatenate(results)

        df["snr"] = result

        return df

def calculate_SNRs(df, detectors: list[str]):
    
    df = parallel_snr(df, detectors)

    return df


def calculate_sky_loc(df, detectors: list[str]):
    
    mask = df["snr"] >= 12
    df["DeltaOmega"] = np.inf * np.ones(df.shape[0])
    events = df.loc[mask, ["mass_1", "mass_2", "luminosity_distance", "theta_jn", "ra", "dec", "psi", "phase", "geocent_time"]]
    network = Network(detectors)

    detected, snr, errors, sky_localization = compute_network_errors(
        network,
        events,
        waveform_model="IMRPhenomXAS_NRTidalv3"
    )

    sky_localization *= sky_localization_percentile_factor()

    df.loc[mask, "DeltaOmega"] = sky_localization

def add_kn_parameters(df):
    compactness_1 = df["mass_1_source"] / df["radius_1"] * 1.477
    compactness_2 = df["mass_2_source"] / df["radius_2"] * 1.477

    pc = (df["mass_1_source"] + df["mass_2_source"]) > df["k_coll"] * mtov

    # dynamical mass 
    df["log10_mej_dyn"] = np.zeros(df.shape[0])
    df.loc[pc, "log10_mej_dyn"] = np.log10(BNSEjectaFitting().dynamic_mass_fitting_prompt_collapse(mass_1=df["mass_1_source"][pc],
                                                                                              mass_2=df["mass_2_source"][pc],
                                                                                              lambda_1=df["lambda_1"][pc],
                                                                                              lambda_2=df["lambda_2"][pc]))

    df.loc[~pc, "log10_mej_dyn"] = np.log10(BNSEjectaFitting().dynamic_mass_fitting_KrFo(mass_1=df["mass_1_source"][~pc],
                                                                                              mass_2=df["mass_2_source"][~pc],
                                                                                              compactness_1=compactness_1[~pc],
                                                                                              compactness_2=compactness_2[~pc]))
    
    # dynamical velocity
    df["v_ej_dyn"] = np.zeros(df.shape[0])
    df.loc[pc, "v_ej_dyn"] = BNSEjectaFitting().dynamic_vel_fitting_prompt_collapse(mass_1=df["mass_1_source"][pc],
                                                                                  mass_2=df["mass_2_source"][pc],
                                                                                  compactness_1=compactness_1[pc],
                                                                                  compactness_2=compactness_2[pc])
    df.loc[~pc, "v_ej_dyn"] = BNSEjectaFitting().dynamic_vel_fitting_Radice2018(mass_1=df["mass_1_source"][~pc],
                                                                                   mass_2=df["mass_2_source"][~pc],
                                                                                   compactness_1=compactness_1[~pc],
                                                                                   compactness_2=compactness_2[~pc])
    

    # disk mass
    df["log10_mdisk"] = np.zeros(df.shape[0])
    df.loc[pc, "log10_mdisk"] = BNSEjectaFitting().log10_disk_mass_fitting_prompt_collapse(mass_1=df["mass_1_source"][pc],
                                                                                              mass_2=df["mass_2_source"][pc],
                                                                                              lambda_1=df["lambda_1"][pc],
                                                                                              lambda_2=df["lambda_2"][pc])
    df.loc[~pc, "log10_mdisk"] = BNSEjectaFitting().log10_disk_mass_fitting(total_mass=df["mass_1_source"][~pc]+df["mass_2_source"][~pc],
                                                                               mass_ratio=df["mass_2_source"][~pc]/df["mass_1_source"][~pc],
                                                                               MTOV=mtov,
                                                                               R16=r16)
    
    # wind ejecta
    df["log10_mej_wind"] = np.zeros(df.shape[0])
    df.loc[pc, "log10_mej_wind"] = np.log10(stats.truncnorm(loc=0.2, scale=0.1, a=-2, b=2).rvs(size=np.sum(pc))) + df["log10_mdisk"][pc]
    df.loc[~pc, "log10_mej_wind"] = np.log10(stats.truncnorm(loc=0.5, scale=0.2, a=-2.5, b=1.5).rvs(size=np.sum(~pc))) + df["log10_mdisk"][~pc]


    # misc 
    df["v_ej_wind"] = np.random.uniform(0.05, 0.15, size=df.shape[0])
    df["Ye_dyn"] = np.random.uniform(0.15, 0.35, size=df.shape[0])
    df["Ye_wind"] = np.random.uniform(0.2, 0.4, size=df.shape[0])
    df["inclination_EM"] = np.arccos(np.abs(np.cos(df["theta_jn"])))
    
    # for the trigger time in mjd
    df["trigger_time"] = Time(df["geocent_time"], format="gps").mjd

    return df

def calculate_lightcurves(df):

    inclination_EM = np.arccos(np.abs(np.cos(df["theta_jn"])))

    df = add_kn_parameters(df)

    # misc
    v_ej_wind = np.random.uniform(0.05, 0.15, size=df.shape[0])
    Ye_dyn = np.random.uniform(0.15, 0.35, size=df.shape[0])
    Ye_wind = np.random.uniform(0.2, 0.4, size=df.shape[0])

    params = dict(inclination_EM=inclination_EM,
                  log10_mej_dyn=df["log10_mej_dyn"],
                  v_ej_dyn=df["v_ej_dyn"],
                  Ye_dyn=Ye_dyn,
                  log10_mej_wind=df["log10_mej_wind"],
                  v_ej_wind=v_ej_wind,
                  Ye_wind=Ye_wind,
                  redshift=df["redshift"],
                  luminosity_distance=df["luminosity_distance"])

    return model_KN.vpredict(params)



def check_lightcurves(df, network: str):

    DeltaOmega_thr = 50 if network=="ETL" else 100
    mask = df["DeltaOmega"] <= DeltaOmega_thr

    mags, times = calculate_lightcurves(df.loc[mask])



    



def determine_mm_detection(df, network: str):

    detection = df["snr"] > 12

    DeltaOmega_thr = 50 if network=="ETL" else 100

    detection &= df["DeltaOmega"] < DeltaOmega_thr



def generate_training_data(detector:str, size: int=10_000):

    df = draw_BNSs(size)
    df = calculate_SNRs(df, detector)

    df["gw_detected"] = df["snr"] >= 12

    return df


def main():


    start = time.time()
    df_ETL = generate_training_data(["ETL1", "ETL2"], size=5_000)
    df_ETL.to_csv("./training_data/train_ETL.dat", sep=" ")
    end = time.time()

    print(f"ETL done, took {end-start} seconds.")

    start = time.time()
    df_ETT = generate_training_data(["ETT"], size=5_000)
    df_ETT.to_csv("./training_data/train_ETT.dat", sep=" ")
    end = time.time()

    print(f"ETT done, took {end-start} seconds.")


if __name__=="__main__":
    main()

