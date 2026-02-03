import sys

import numpy as np
import pandas as pd
import tqdm
from astropy.time import Time

from fiesta.inference.lightcurve_model import FluxModel, CombinedSurrogate
from telescopes import ztf, vr, pstarrs, ultrasat, ultrasat_filter

Mpc_to_cm = 3.8057e24
np.random.seed(671938)

model_KN = FluxModel(name="Bu2026_MLP", filters=["ztfg", "ztfi", "lsstg", "lssti", "ps1::g", "ps1::i"])
model_afterglow = FluxModel(name="pbag_gaussian_CVAE", filters=["ztfg", "ztfi", "lsstg", "lssti", "ps1::g", "ps1::i", "radio-2.4GHz"])
model = CombinedSurrogate(models = [model_KN, model_afterglow], sample_times=np.geomspace(0.2, 2000, 200))
model.add_filter(ultrasat_filter)


def main():

    df_gw = pd.read_csv(sys.argv[1], sep=" ")
    df_gw["trigger_time"] = Time(df_gw["geocent_time"], format="gps").mjd
    df_FIM = pd.read_csv(sys.argv[2], sep=" ")
    df_kn = pd.read_csv(sys.argv[3], sep=" ")
    df_grb = pd.read_csv(sys.argv[4], sep=" ")

    NBNS = df_gw.shape[0]

    if "narrow" in sys.argv[1]:
        mass_dist = "narrow"
    else: 
        mass_dist = "wide"
    
    if "CE" in sys.argv[1]:
        detectors = "ET_CE"
    else:
        detectors = "ET"

    outfile = f"./detection_output/{mass_dist}_{detectors}.dat"
    with open(outfile, "w") as out:
        out.write(f"redshift    gw_detected    grb_detected    DeltaOmega    ")
        for key in ["telt_vr", "lsstg", "lssti", "telt_ztf", "ztfg", "ztfi", "telt_pstarrs", "ps1::g", "ps1::i", "telt_ultrasat", "ultrasat_custom"]:
            out.write(f"{key}    ")
        out.write("afterglow_detected \n")

    for j in tqdm.tqdm(range(NBNS)):
        observation_campaign(outfile, df_gw.iloc[j], df_FIM.iloc[j], df_kn.iloc[j], df_grb.iloc[j])



def observation_campaign(outfile, gw_event, fim_event, kn_event, grb_event) -> tuple[bool, bool, float, bool, bool]:
    
    ################
    # GW Detection #
    ################
    
    gw_detected = (fim_event["snr"] > 12)
    
    #################
    # GRB Detection #
    #################

    grb_detected, DeltaOmegaGRB = grb_detection(grb_event)
    DeltaOmega = min(fim_event["sky_localization"], DeltaOmegaGRB)

    if not gw_detected or DeltaOmega > 500:
        if not gw_detected:
            DeltaOmega = np.inf
        with open(outfile, "a") as out:
            out.write(f"{gw_event["redshift"]:.3f}    {int(gw_detected)}    {int(grb_detected)}    {DeltaOmega:.2f}    ")
            for key in ["telt_vr", "lsstg", "lssti", "telt_ztf", "ztfg", "ztfi", "telt_pstarrs", "ps1::g", "ps1::i", "telt_ultrasat", "ultrasat_custom"]:
                out.write(f"{0}    ")
            out.write(f"{0} \n")
        
        return

    ###################
    # UVOIR Detection #
    ###################
    kn_results = kn_detection(gw_event, kn_event, grb_event, DeltaOmega)

    #######################
    # Afterglow Detection #
    #######################
    afterglow_detected = afterglow_detection(gw_event, kn_event, grb_event, DeltaOmega)

    

    with open(outfile, "a") as out:
        out.write(f"{gw_event["redshift"]:.3f}    1    {int(grb_detected)}    {DeltaOmega:.2f}    ")
        for key in ["telt_vr", "lsstg", "lssti", "telt_ztf", "ztfg", "ztfi", "telt_pstarrs", "ps1::g", "ps1::i", "telt_ultrasat", "ultrasat_custom"]:
            out.write(f"{kn_results[key]}    ")
        out.write(f"{int(afterglow_detected)} \n")



def log10_fluence(df, key):
    fluence = df[key] + np.log10( (1+df['redshift']) / (4*np.pi*df["luminosity_distance"]**2 * Mpc_to_cm**2))
    return fluence


def grb_detection(event):
    
    log10_fermi_fluence = log10_fluence(event, "log10_Egamma_fermi_gbm")
    log10_swift_fluence = log10_fluence(event, "log10_Egamma_swift_bat")
    log10_gecam_fluence = log10_fluence(event, "log10_Egamma_gecam")
    
    mask_fermi = (log10_fermi_fluence > np.log10(2e-7)) & (np.random.uniform() < 0.6)
    mask_swift = (log10_swift_fluence > np.log10(2e-8)) & (np.random.uniform() < 0.1)
    mask_gecam = (log10_gecam_fluence > np.log10(2e-8)) & (np.random.uniform() < 0.8)

    DeltaOmega = np.inf
    if mask_fermi: 
        DeltaOmega = 100
    if mask_gecam: 
        DeltaOmega = 10
    if mask_swift:
        DeltaOmega = 0.1
    
    return (mask_fermi or (mask_swift or mask_gecam)), DeltaOmega

def kn_detection(gw_event, kn_event, grb_event, DeltaOmega):

    trigger_time = gw_event["trigger_time"]
    dec = gw_event["dec"]
    ra = gw_event["ra"]
    
    params = dict(kn_event)
    params.update(kn_event)
    params.update(grb_event)
    params.update(dict(alphaWing=2., p=2.15, log10_epsilon_e=-1., log10_epsilon_B=-3., Gamma0=500))
    params["log10_E0"] = params.pop("log10_Ekin_iso")
    times_transient, mags_transient = model.predict(params)
    times_transient += trigger_time

    telt_ztf, detections_ztf = ztf.follow_up_campaign(trigger_time, 
                                                                dec, 
                                                                ra, 
                                                                DeltaOmega, 
                                                                times_transient, 
                                                                mags_transient)
    
    telt_vr, detections_vr = vr.follow_up_campaign(trigger_time, 
                                                   dec, 
                                                   ra, 
                                                   DeltaOmega, 
                                                   times_transient, 
                                                   mags_transient)

    telt_pstarrs, detections_pstarrs = pstarrs.follow_up_campaign(trigger_time, 
                                                                  dec, 
                                                                  ra, 
                                                                  DeltaOmega, 
                                                                  times_transient, 
                                                                  mags_transient)

    telt_ultrasat, detections_ultrasat = ultrasat.follow_up_campaign(trigger_time, 
                                                                     dec, 
                                                                     ra, 
                                                                     DeltaOmega, 
                                                                     times_transient, 
                                                                     mags_transient)
    
    kn_results= dict(telt_ztf=telt_ztf,
                     telt_vr=telt_vr,
                     telt_pstarrs=telt_pstarrs,
                     telt_ultrasat=telt_ultrasat)
    kn_results.update(detections_ztf)
    kn_results.update(detections_vr)
    kn_results.update(detections_pstarrs)
    kn_results.update(detections_ultrasat)
    
    return kn_results


def afterglow_detection(gw_event, kn_event, grb_event, DeltaOmega):
    return False

if __name__=="__main__":
    main()


