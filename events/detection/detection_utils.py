from typing import Iterable

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from astropy import coordinates


from fiesta.filters import Filter
from fiesta.conversions import mag_app_from_mag_abs
from fiesta.extinction import extinctionFactorP92SMC

from telescopes import ultrasat_filter

Mpc_to_cm = 3.8057e24

def log10_fluence(df, key):
    fluence = df[key] + np.log10( (1+df['redshift']) / (4*np.pi*df["luminosity_distance"]**2 * Mpc_to_cm**2))
    return fluence


def which_telescopes_will_observe(DeltaOmega, redshift, has_CE):
    start = dict(ztf=True, vr=True, pstarrs=True, ultrasat=True)

    if (DeltaOmega>200) or (DeltaOmega>100 and has_CE) or redshift>0.2:
        start["ztf"] = False
    
    if (DeltaOmega>50.) or (has_CE and DeltaOmega>=10.) or (has_CE and (DeltaOmega>8. and redshift>=0.2) ) or redshift>1.:
        start["vr"] = False
    
    if (DeltaOmega>50.) or (DeltaOmega>30 and has_CE) or redshift>0.2:
        start["pstarrs"] = False

    if redshift>0.5:
        start["ultrasat"] = False

    return start

def which_telescopes_will_observe_afterglow(DeltaOmega, redshift, has_CE):
    start = dict(ska=True, dsa=True, vr=True, ep=True)

    if (DeltaOmega>50.) or (DeltaOmega>10. and has_CE) or redshift>1.:
        start["ska"] = False
        start["dsa"] = False
        start["vr"] = False
        start["ep"] = False

    return start

def check_afterglow_thresholds(log10_flux, times, nus):
    
    # radio 
    radio_filt = Filter("radio-1.4GHz")
    log10flux_radio = interpolate.interp1d(nus, log10_flux, axis=0)(radio_filt.nu)
    radio_duration = total_time_visible(times, log10flux_radio >= np.log10(5e-4))
    radio_peak = times[log10flux_radio.argmax()]
    radio_detectable = radio_duration >= 7 or radio_peak <=21

    
    # iband
    gband = Filter("lsstg")
    mag_g = gband.get_mag(10**log10_flux, nus)
    gband_duration = total_time_visible(times, mag_g <= 29)
    gband_peak = times[mag_g.argmin()]
    gband_detectable = gband_duration >= 7 or gband_peak<=21

    
    # xray
    xray_filt = Filter("X-ray-0.5-2keV")
    log10flux_xray = interpolate.interp1d(nus, log10_flux, axis=0)(xray_filt.nus)
    xray_fluence = integrate.simpson(y=10**(log10flux_xray-26), x=xray_filt.nus, axis=0)
    xray_duration = total_time_visible(times, xray_fluence >= 1e-19)
    xray_peak = times[xray_fluence.argmax()]
    xray_detectable = xray_duration >= 7 or xray_peak <=21

    return dict(radio_afterglow=int(radio_detectable), opt_afterglow=int(gband_detectable), xray_afterglow=int(xray_detectable), telt_vr=0.)


def total_time_visible(times, visible_mask):
    y = np.zeros_like(times)
    y[visible_mask] = 1
    return np.trapz(y=y, x=times)

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


