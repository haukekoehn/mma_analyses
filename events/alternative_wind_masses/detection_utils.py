from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from astropy import coordinates
from astropy.utils import iers
iers.conf.auto_max_age = None

from fiesta.filters import Filter
from fiesta.conversions import mag_app_from_mag_abs, apply_redshift
from fiesta.extinction import extinctionFactorP92SMC

from telescopes import ultrasat_filter

Mpc_to_cm = 3.8057e24

def log10_fluence(df, key):
    fluence = df[key] + np.log10( (1+df['redshift']) / (4*np.pi*df["luminosity_distance"]**2 * Mpc_to_cm**2))
    return fluence


def which_telescopes_will_observe(DeltaOmega, redshift, grb_detected: bool, gw_detectors):
    if grb_detected:
        start = dict(
            ultrasat = 0.2 >= redshift, # 1000 Mpc
            ztf = 0.2 >= redshift, # 1000 Mpc
            pstarrs = 0.5 >= redshift, # 3000 Mpc
            vr = 1 >= redshift, # 7000 Mpc
            roman = 0.5>=redshift # 3000 Mpc
        )    
    else:
        start = dict(
            ultrasat = 0.105 >= redshift, # 500 Mpc
            ztf = 0.085 >= redshift, # 400 Mpc
            pstarrs = 0.2 >= redshift, # 1000 Mpc
            vr = 0.65 >= redshift, # 4000 Mpc
            roman = 0.36>=redshift # 2000 Mpc
        )    

    if "CE" in gw_detectors:
        # ZTF
        if DeltaOmega > 100:
            start['ztf'] &= False
        # VR
        if DeltaOmega > 10. or (DeltaOmega > 7. and redshift > 0.2):
            start['vr'] &= False
        # PSTARRS
        if DeltaOmega > 30:
            start['pstarrs'] &= False
        # ROMAN
        if DeltaOmega > 1:
            start['roman'] &= False

    elif gw_detectors=="ETL":
        # ZTF
        if DeltaOmega > 200:
            start["ztf"] &= False
        # VR
        if DeltaOmega > 50:
            start["vr"] &= False
        # PSTARRS
        if DeltaOmega > 50:
            start['pstarrs'] &= False
        # ROMAN
        if DeltaOmega > 10:
            start['roman'] &= False

    elif gw_detectors=="ETT":
        # ZTF
        if DeltaOmega > 200.:
            start["ztf"] &= False
        # VR
        if DeltaOmega > 100.:
            start["vr"] &= False
        # PSTARRS
        if DeltaOmega > 50:
            start['pstarrs'] &= False
        # ROMAN
        if DeltaOmega > 10:
            start['roman'] &= False

    else:
        raise ValueError(f"Invalid GW detector {gw_detectors}.")

    return start

def which_telescopes_will_observe_afterglow(DeltaOmega, redshift, detectors):
    start = dict(ska=True, dsa=True, vr=True, ep=True)

    if "CE" in detectors:
        if DeltaOmega>10. or redshift > 1.5:
            start = dict(ska=False, dsa=False, vr=False, ep=False)
        
    elif detectors=="ETL":
        if DeltaOmega>50. or redshift > 1.5:
            start = dict(ska=False, dsa=False, vr=False, ep=False)
    
    elif detectors=="ETT":
        if DeltaOmega>100. or redshift > 1.5:
            start = dict(ska=False, dsa=False, vr=False, ep=False)

    return start

def check_afterglow_thresholds(log10_flux, times, nus, start_time):

    mask = times >= start_time
    times = times[mask]
    log10_flux = log10_flux[:, mask].reshape(len(nus), len(times))
    
    # radio 
    radio_filt = Filter("radio-1.4GHz")
    log10flux_radio = interpolate.interp1d(nus, log10_flux, axis=0)(radio_filt.nu)
    radio_duration = total_time_visible(times, log10flux_radio >= np.log10(5e-4))
    radio_peak = times[log10flux_radio.argmax()]
    radio_detectable = radio_duration >= 7 or radio_peak <=21

    
    # gband
    gband = Filter("lsstg")
    mag_g = gband.get_mag(10**log10_flux, nus)
    gband_duration = total_time_visible(times, mag_g <= 29)
    gband_peak = times[mag_g.argmin()]
    gband_detectable = (gband_duration >= 7 or gband_peak<=21) and gband_peak >= 10

    
    # xray
    xray_filt = Filter("X-ray-0.5-2keV")
    log10flux_xray = interpolate.interp1d(nus, log10_flux, axis=0)(xray_filt.nus)
    xray_fluence = integrate.simpson(y=10**(log10flux_xray-26), x=xray_filt.nus, axis=0)
    xray_duration = total_time_visible(times, xray_fluence >= 1e-19)
    xray_peak = times[xray_fluence.argmax()]
    xray_detectable = xray_duration >= 7 or xray_peak <=21

    return dict(radio_afterglow=int(radio_detectable), opt_afterglow=int(gband_detectable), xray_afterglow=int(xray_detectable), telt_vr_afterglow=0.)


def total_time_visible(times, visible_mask):
    y = np.zeros_like(times)
    y[visible_mask] = 1
    return np.trapezoid(y=y, x=times)

def check_surrogate_param_range(params, models):

    if not isinstance(models, Iterable):
        models = [models]

    for model in models:
        for p, (pmin, pmax, _) in model.parameter_distributions.items():
            params[p] = np.maximum(pmin, params[p])
            params[p] = np.minimum(pmax, params[p])

    return params

def apply_extinction_mag(mags, redshift, Ebv):

    nus = np.array([filter_freq(key) for key in mags])
    ext = extinctionFactorP92SMC(nus, Ebv, redshift)
    for j, key in enumerate(mags):
        mags[key] += -2.5*np.log10(ext[j])
    
    return mags

def filter_freq(filt):

    if filt != "ultrasat_custom":
        return Filter(filt).nu
    else:
        return ultrasat_filter.nu


def determine_kn_visibility(params, model_kn, model_afg):
    """
    This function tries to determine whether the kn is outshined by the afterglow (e.g. at low inclination).
    """
    times_kn, mags_kn = model_kn.predict(params)
    times_afg, mags_afg = model_afg.predict(params)

    times = np.geomspace(1, 10, 100)    
    mag_difference = np.interp(times, times_kn, mags_kn["lsstg"]) - np.interp(times, times_afg, mags_afg["lsstg"])
    mags_kn = np.interp(times, times_kn, mags_kn["lsstg"])
    mags_afg_xray = np.interp(times, times_afg, mags_afg["X-ray-1keV"])
    mags_afg_radio = np.interp(times, times_afg, mags_afg["radio-1.4GHz"])

    kn_visible = np.any( (mag_difference <= 1) & (mags_kn<30.) )

    grb_visible = np.any( (mag_difference >= 1) & (mags_kn<30) ) or np.any(mags_afg_xray<42) or np.any(mags_afg_radio<24.5)

    return int(kn_visible), int(grb_visible)



def predict_kilonova_afterglow(j, mass_dist, params):
    with h5py.File(f"../kilonova_afterglow/kilonova_afterglow_files/kn_afterglow_results_{mass_dist}.mat") as f:
        flux = f["flux_mJy"][:, :, j]
        times_kn_afg = f["t_days"][:].flatten()
        nus_kn_afg = f["nu_Hz"][:].flatten()
    
    times_kn_afg, nus_kn_afg, flux = apply_redshift(flux, times_kn_afg, nus_kn_afg, params["redshift"])

    times = np.geomspace(1, 365*10, 250)
    nus = np.geomspace(1e9, 2e18, 200)

    log10_flux = interpolate.interp1d(np.log10(nus_kn_afg), np.log10(flux), 
                                      fill_value="extrapolate", axis=0)(np.log10(nus))
    log10_flux = interpolate.interp1d(np.log10(times_kn_afg), log10_flux, 
                                      fill_value="extrapolate", axis=1)(np.log10(times))

    log10_flux += 2*np.log10(1e2/params["luminosity_distance"])

    return times, nus, log10_flux


def determine_afterglow_visibility(log10_flux, log10_flux_kn_afg, times, nus, afterglow_result, transient_search):
    
    if not (afterglow_result["radio_afterglow"] or afterglow_result["xray_afterglow"] or afterglow_result["opt_afterglow"]):
        afterglow_result["kn_afg_visible"] = 0
        afterglow_result["grb_afg_visible"] = 0
        return afterglow_result

    log10_flux_reduced = interpolate.interp1d(np.log10(nus), log10_flux, axis=0)(np.log10([1.4e9, 6.25e14, 2.417989e+17]))

    if transient_search:
        t_radio = times[log10_flux_reduced[0]>np.log10(5e-3)]
        t_opt = times[log10_flux_reduced[1]>np.log10(9e-5)]
        t_xray = t_opt # xray will never be found in transient searched  

    else:
        t_radio = times[log10_flux_reduced[0]>np.log10(5e-4)]
        t_opt = times[log10_flux_reduced[1]>np.log10(1e-5)]
        t_xray = times[log10_flux_reduced[2]>np.log10(5e-11)]
    
    non_empty = [a for a in [t_radio, t_opt, t_xray] if a.size > 0]
    if not non_empty:
        # if none of the thresholds is met 
        # (because these are in flux density and the detection is with fluxes or mags)
        # then simply use the radio peak
        first_epoch = times[log10_flux_reduced[0].argmax()]
    else:
        first_epoch = np.min(np.concatenate(non_empty))
    t_obs = np.geomspace(first_epoch, 10*365, 50)
    
    # interpolate the fluxes to the observed frequencies and times
    flux_diff = log10_flux_kn_afg - log10_flux
    flux_diff = interpolate.interp1d(np.log10(nus), flux_diff, axis=0)(np.log10([1.4e9, 6.25e14, 2.417989e+17]))
    flux_diff = interpolate.interp1d(times, flux_diff, axis=1)(t_obs)
    flux_diff = 10**(flux_diff)
    
    log10_flux = interpolate.interp1d(np.log10(nus), log10_flux, axis=0)(np.log10([1.4e9, 6.25e14, 2.417989e+17]))
    log10_flux = interpolate.interp1d(times, log10_flux, axis=1)(t_obs)

    kn_afg_visible = 0
    grb_afg_visible = 0

    # in radio
    kn_afg_visible += int( np.any( (flux_diff[0, :]>=0.3) & (log10_flux[0]>np.log10(5e-4)) ) )
    grb_afg_visible += int(  np.any( (flux_diff[0, :]<=0.7) & (log10_flux[0]>np.log10(5e-4)) ) )

    # in optical
    kn_afg_visible += int( np.any( (flux_diff[1, :]>=0.3) & (log10_flux[1]>np.log10(1e-5)) ) )
    grb_afg_visible += int( np.any( (flux_diff[1, :]<=0.7) & (log10_flux[1]>np.log10(1e-5)) ) )

    # in xray
    kn_afg_visible += int( np.any((flux_diff[2, :]>=0.3) & (log10_flux[2]>np.log10(5e-11)) ) )
    grb_afg_visible += int( np.any((flux_diff[2, :]<=0.7) & (log10_flux[2]>np.log10(5e-11)) ) )

    afterglow_result["kn_afg_visible"] = kn_afg_visible
    afterglow_result["grb_afg_visible"] = grb_afg_visible
        
    return afterglow_result


