import sys

import numpy as np
import pandas as pd
import tqdm
from astropy.time import Time
from scipy.interpolate import interp1d

from fiesta.filters import Filter
from fiesta.conversions import apply_redshift
from fiesta.inference.lightcurve_model import FluxModel, CombinedSurrogate
from fiesta.extinction import extinctionFactorP92SMC

from telescopes import ztf, vr, pstarrs, ultrasat, ultrasat_filter, ska, dsa, einsteinprobe
from detection_utils import (which_telescopes_will_observe, 
                             check_afterglow_thresholds, 
                             which_telescopes_will_observe_afterglow, 
                             check_surrogate_param_range, 
                             apply_extinction_mag, 
                             determine_kn_visibility,
                             predict_kilonova_afterglow,
                             determine_afterglow_visibility)

Mpc_to_cm = 3.8057e24
np.random.seed(671938)

model_KN = FluxModel(name="Bu2026_MLP", filters=["ztfg", "ztfi", "lsstg", "lssti", "ps1::g", "ps1::i"])
model_afterglow = FluxModel(name="pbag_gaussian_CVAE", filters=["ztfg", "ztfi", "lsstg", "lssti", "ps1::g", "ps1::i", "radio-1.4GHz", "X-ray-1keV"])
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
    
    if "CE" in sys.argv[2]:
        detectors = "ET_CE"
        has_CE = True
    else:
        detectors = "ET"
        has_CE = False
    
    outfile = f"./detection_output/{mass_dist}_{detectors}.dat"
    prepare_outfile(outfile)

    for j in tqdm.tqdm(range(NBNS)):
        observation_campaign(j, mass_dist, outfile, df_gw.iloc[j], df_FIM.iloc[j], df_kn.iloc[j], df_grb.iloc[j], has_CE)

def prepare_outfile(outfile):
    with open(outfile, "w") as out:
        out.write(f"redshift    gw_detected    grb_detected    DeltaOmega    ")
        for key in ["telt_vr", "lsstg", "lssti", "telt_ztf", "ztfg", "ztfi", "telt_pstarrs", "ps1::g", "ps1::i", "telt_ultrasat", "ultrasat_custom"]:
            out.write(f"{key}    ")
        for key in ["radio_afterglow", "opt_afterglow", "xray_afterglow", "telt_vr_afterglow"]:
            out.write(f"{key}    ")
        for key in ["kn_visible", "grb_afg_visible", "kn_afg_visible"]:
            out.write(f"{key}    ")
        out.write(f" \n")

def observation_campaign(j, mass_dist, outfile, gw_event, fim_event, kn_event, grb_event, has_CE) -> tuple[bool, bool, float, bool, bool]:
    
    ################
    # GW Detection #
    ################
    
    gw_detected = (fim_event["snr"] > 12)
    
    #################
    # GRB Detection #
    #################

    grb_detected, DeltaOmegaGRB = grb_detection(grb_event)
    DeltaOmega = min(fim_event["sky_localization"], DeltaOmegaGRB)

    no_follow_up = (not gw_detected) or (DeltaOmega > 400)

    if no_follow_up:
        if not gw_detected:
            DeltaOmega = np.inf

        with open(outfile, "a") as out:
            out.write(f"{gw_event['redshift']:.3f}    {int(gw_detected)}    {int(grb_detected)}    {DeltaOmega:.2f}    ")
            for key in ["telt_vr", "lsstg", "lssti", "telt_ztf", "ztfg", "ztfi", "telt_pstarrs", "ps1::g", "ps1::i", "telt_ultrasat", "ultrasat_custom"]:
                out.write(f"{0}    ")
            for key in ["radio_afterglow", "opt_afterglow", "xray_afterglow", "telt_vr_afterglow"]:
                out.write(f"{0}    ")
            for key in ["kn_visible", "grb_afg_visible", "kn_afg_visible"]:
                out.write(f"{0}    ")
            out.write(f" \n")
        
        return

    ###################
    # UVOIR Detection #
    ###################
    kn_results = kn_detection(gw_event, kn_event, grb_event, DeltaOmega, has_CE)

    #######################
    # Afterglow Detection #
    #######################
    afg_results = afterglow_detection(gw_event, kn_event, grb_event, DeltaOmega, kn_results, has_CE, j, mass_dist)
    
    afg_results["grb_afg_visible"] += kn_results["grb_afg_visible"]
    
    with open(outfile, "a") as out:
        out.write(f"{gw_event['redshift']:.3f}    1    {int(grb_detected)}    {DeltaOmega:.2f}    ")
        for key in ["telt_vr", "lsstg", "lssti", "telt_ztf", "ztfg", "ztfi", "telt_pstarrs", "ps1::g", "ps1::i", "telt_ultrasat", "ultrasat_custom"]:
            out.write(f"{kn_results[key]}    ")
        
        for key in ["radio_afterglow", "opt_afterglow", "xray_afterglow", "telt_vr"]:
            out.write(f"{afg_results[key]}    ")
        
        for key in ["kn_visible"]:
            out.write(f"{int(kn_results[key])}    ")

        for key in ["grb_afg_visible", "kn_afg_visible"]:
            out.write(f"{int(afg_results[key])}    ")

        out.write("\n")

def grb_detection(event):

    if not event["has_grb"]:
        return 0, np.inf
    
    DeltaOmega = np.inf
    if event["fermi_detected"]: 
        DeltaOmega = 100
    if event["gecam_detected"]: 
        DeltaOmega = 10
    if event["swift_detected"]:
        DeltaOmega = 0.1
    
    return DeltaOmega < np.inf, DeltaOmega

def kn_detection(gw_event, kn_event, grb_event, DeltaOmega, has_CE):

    trigger_time = gw_event["trigger_time"]
    dec = gw_event["dec"]
    ra = gw_event["ra"]
    
    params = dict(kn_event)
    params.update(kn_event)
    params.update(grb_event)
    params.update(dict(alphaWing=2., p=2.15, log10_epsilon_e=-1., log10_epsilon_B=-3., Gamma0=500))
    params["log10_E0"] = params.pop("log10_Ekin_iso")
    params = check_surrogate_param_range(params, [model_KN, model_afterglow])

    if grb_event["has_grb"]:
        times_transient, mags_transient = model.predict(params)
    else:
        times_transient, mags_transient = model_KN.predict(params)

    mags_transient = apply_extinction_mag(mags_transient, params["redshift"], kn_event["Ebv"])
    times_transient += trigger_time
    
    start = which_telescopes_will_observe(DeltaOmega, params["redshift"], has_CE)
    
    telt_ztf, detections_ztf = ztf.kilonova_campaign(start["ztf"],
                                                      trigger_time, 
                                                      dec, 
                                                      ra, 
                                                      DeltaOmega, 
                                                      times_transient, 
                                                      mags_transient)

    telt_vr, detections_vr = vr.kilonova_campaign(start["vr"],
                                                   trigger_time, 
                                                   dec, 
                                                   ra, 
                                                   DeltaOmega, 
                                                   times_transient, 
                                                   mags_transient)

    telt_pstarrs, detections_pstarrs = pstarrs.kilonova_campaign(start["pstarrs"],
                                                                  trigger_time, 
                                                                  dec, 
                                                                  ra, 
                                                                  DeltaOmega, 
                                                                  times_transient, 
                                                                  mags_transient)

    telt_ultrasat, detections_ultrasat = ultrasat.kilonova_campaign(start["ultrasat"],
                                                                     trigger_time, 
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

    uvoir_detected = np.sum([kn_results[key] for key in ["ztfg", "ztfi", "lsstg", "lssti", "ps1::g", "ps1::i", "ultrasat_custom"]] ) >=2
    kn_results["kn_visible"] = uvoir_detected
    kn_results["grb_afg_visible"] = 0
    
    if uvoir_detected and grb_event["has_grb"]:
        kn_results["kn_visible"], kn_results["grb_afg_visible"] = determine_kn_visibility(params, model_KN, model_afterglow)
    
    return kn_results


def afterglow_detection(gw_event, kn_event, grb_event, DeltaOmega, kn_result, has_CE, j, mass_dist):
    
    trigger_time = kn_event["trigger_time"]
    dec = gw_event["dec"]
    ra = gw_event["ra"]
    redshift = gw_event["redshift"]

    times, nus, log10_flux_kn_afg = predict_kilonova_afterglow(j, mass_dist, gw_event)
    log10_flux_kn_afg += np.log10(extinctionFactorP92SMC(nus, kn_event["Ebv"], redshift)[:, None])

    if grb_event["has_grb"]:

        # get grb afterglow flux
        params = dict(grb_event)
        params["log10_E0"] = params.pop("log10_Ekin_iso")
        params.update(dict(alphaWing=2., p=2.15, log10_epsilon_e=-1., log10_epsilon_B=-3., Gamma0=500))
        params = check_surrogate_param_range(params, [model_afterglow])

        times_grb_afg, nus_afg, log10_flux_grb_afg = model_afterglow.predict_log_flux(params)
        log10_flux_grb_afg = interp1d(np.log10(nus_afg), log10_flux_grb_afg, axis=0, fill_value="extrapolate")(np.log10(nus))
        log10_flux_grb_afg = interp1d(np.log10(times_grb_afg), log10_flux_grb_afg, axis=1, fill_value="extrapolate")(np.log10(times))
        log10_flux_grb_afg += np.log10(extinctionFactorP92SMC(nus, kn_event["Ebv"], redshift)[:, None])

        log10_flux = np.log10(10**log10_flux_grb_afg + 10**log10_flux_kn_afg)

    else:
        log10_flux = log10_flux_kn_afg
    
    mags = {Filt.name: Filt.get_mag(10**log10_flux, nus) for Filt in [Filter("radio-1.4GHz"), Filter("lsstg"), Filter("lssti"), Filter("X-ray-0.5-4keV")]}
    times += trigger_time

    # check whether a KN has been detected
    uvoir_detected = np.sum([kn_result[key] for key in ["lssti", "lsstg", "ultrasat_custom", "ztfg", "ztfi", "ps1::g", "ps1::i"]]) >=2

    if uvoir_detected:
        afterglow_results = check_afterglow_thresholds(log10_flux, times, nus)

    else:

        start = which_telescopes_will_observe_afterglow(DeltaOmega, redshift, has_CE)

        _, detections_ska = ska.afterglow_campaign(start["ska"],
                                                   trigger_time,
                                                   dec,
                                                   ra,
                                                   DeltaOmega,
                                                   times,
                                                   mags)
        
        _, detections_dsa = dsa.afterglow_campaign(start["dsa"],
                                                   trigger_time,
                                                   dec,
                                                   ra,
                                                   DeltaOmega,
                                                   times,
                                                   mags)

        telt_vr, detections_vr = vr.afterglow_campaign(start["vr"],
                                                       trigger_time,
                                                       dec,
                                                       ra,
                                                       DeltaOmega,
                                                       times,
                                                       mags)

        _, detections_ep = einsteinprobe.afterglow_campaign(start["ep"],
                                                            trigger_time,
                                                            dec,
                                                            ra,
                                                            DeltaOmega,
                                                            times,
                                                            nus,
                                                            log10_flux)

        afterglow_results = dict(telt_vr=telt_vr/2) # divide by two because we only use one filter, little hacky
        afterglow_results["radio_afterglow"] = sum(detections_ska.values()) + sum(detections_dsa.values())
        afterglow_results["opt_afterglow"] = detections_vr["lsstg"]
        afterglow_results["xray_afterglow"] = detections_ep["X-ray-0.5-4keV"]

    
    afterglow_results = determine_afterglow_visibility(log10_flux, log10_flux_kn_afg, times-trigger_time, nus, afterglow_results, ~uvoir_detected)

    return afterglow_results
         
if __name__=="__main__":
    main()


