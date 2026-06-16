import numpy as np

import scipy.stats as stats
import csv
import pandas as pd

from astropy.cosmology import Planck18
from astropy.time import Time

from gwtoolbox.tools_earth import set_cosmology, Tools


# load EOS
m_val, r_val, l_val = np.loadtxt("../../eos/RMF3_MRL.dat", unpack=True)

np.random.seed(9167896)


def narrow_mass_model(n_samples: int):
    
    alpha = 0.68
    mu1 = 1.34
    sigma1 = 0.02
    mu2 = 1.47
    sigma2 = 0.15

    m_l = 1.16
    m_u = 1.42
    
    mrecycled_peak1 = np.random.normal(loc=mu1, scale=sigma1, size=n_samples)
    mrecycled_peak2 = np.random.normal(loc=mu2, scale=sigma2, size=n_samples)

    mask = np.random.choice([False, True], replace=True, size=n_samples, p=[alpha, 1-alpha])
    mrecycled = mrecycled_peak1
    mrecycled[mask] = mrecycled_peak2[mask]
    chi_r = np.random.gamma(shape=2, scale=0.012, size=n_samples)
    

    mslow = np.random.uniform(m_l, m_u, size=n_samples)
    
    m1 = np.zeros(n_samples)
    m2 = np.zeros(n_samples)
    chi1_z = np.zeros(n_samples)
    chi2_z = np.zeros(n_samples)

    mask_m1 = mrecycled > mslow
    m1[mask_m1] = mrecycled[mask_m1]
    m1[~mask_m1] = mslow[~mask_m1]
    m2[mask_m1] = mslow[mask_m1]
    m2[~mask_m1] = mrecycled[~mask_m1]

    chi1_z[mask_m1] = chi_r[mask_m1]
    chi2_z[~mask_m1] = chi_r[~mask_m1]

    return m1, m2, chi1_z, chi2_z

def wide_mass_model(n_samples: int):

    m_min = 1.1
    m_max = 2.0

    m1 = np.empty(n_samples)
    m2 = np.empty(n_samples)
    
    ind_start = 0
    ind_end = 0


    while True:
        m1_tmp = np.random.uniform(low=m_min, high=m_max, size=1000)
        m2_tmp = np.random.uniform(low=m_min, high=m_max, size=1000)
        m1_tmp, m2_tmp = np.maximum(m1_tmp, m2_tmp), np.minimum(m1_tmp, m2_tmp)
        
        p = (m2_tmp/m1_tmp)**2
        alpha = np.random.uniform(0, 1, size=p.shape[0])

        mask = (alpha <= p)
        ind_end += np.sum(mask)

        if ind_end >= n_samples:
            
            n_rest = n_samples-ind_start
            m1[ind_start:None] = m1_tmp[mask][:n_rest]
            m2[ind_start:None] = m2_tmp[mask][:n_rest]
            break
        
        else:
            m1[ind_start:ind_end] = m1_tmp[mask]
            m2[ind_start:ind_end] = m2_tmp[mask]

        ind_start = ind_end
      
    # spins
    chi1_z = np.random.normal(loc=0, scale=0.05, size=n_samples)
    chi2_z = np.random.normal(loc=0, scale=0.05, size=n_samples)

    return m1, m2, chi1_z, chi2_z




def sample_merger_rate(R0: float = 50.,
                    tau: float = 3.,
                    ):
    
    cosmo = set_cosmology(H0=Planck18.H0.value, Om0=Planck18.Om0, Tcmb=Planck18.Tcmb0.value)
    m_mean = 1.35 # Msol
    m_scale = 0.3# Msol
    m_low = 1.1 # Msol
    m_high = 2.1 # Msol
    chi_sigma = 0.04

    Toolset = Tools(detector_type='et',
                event_type='nsns',
                population='I',
                cosmos=cosmo,
                det_setup=None,
                new_theta=[R0, tau, m_mean, m_scale, m_low, m_high, chi_sigma]) # population parameters
    

    df = Toolset.list_params_df(time_obs=60*24*365*1, rho_cri=0)
    df["m1"], df["m2"] = df[["m1", "m2"]].max(axis=1), df[["m1", "m2"]].min(axis=1)
    df.rename(columns={"m1": "mass_1_source", "m2": "mass_2_source", "z": "redshift", "D": "luminosity_distance", "χ": "chi_eff"}, inplace=True)
    return df 

def add_mass_and_spin_BNS_parameters(df, mass_distribution: str):
    match mass_distribution:
        case "narrow":
            mass_model = narrow_mass_model
        case "wide":
            mass_model = wide_mass_model

    df["mass_1_source"], df["mass_2_source"], df["chi1_z"], df["chi2_z"] = mass_model(df.shape[0])
    df["mass_1"] = (1+df["redshift"]) * df["mass_1_source"]
    df["mass_2"] = (1+df["redshift"]) * df["mass_2_source"]
    
    # EOS
    df["radius_1"] = np.interp(df["mass_1_source"], m_val, r_val) # EOS
    df["radius_2"] = np.interp(df["mass_2_source"], m_val, r_val)
    df["lambda_1"] = np.interp(df["mass_1_source"], m_val, l_val)
    df["lambda_2"] = np.interp(df["mass_2_source"], m_val, l_val)

    df["chi_eff"] = (df["mass_1"] * df["chi1_z"] + df["mass_2"] * df["chi2_z"]) / (df["mass_1"] + df["mass_2"])

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


def get_BNS_catalog(mass_distribution: str,
                    R0: float = 50.,
                    tau: float = 3.,
                    chi_sigma: float =0.04,
                    ):
    
    df = sample_merger_rate(R0, tau)
    df = add_mass_and_spin_BNS_parameters(df, mass_distribution)
    df = add_observational_BNS_parameters(df)

    return df


def main():

    df_narrow = get_BNS_catalog("narrow")
    df_narrow.to_csv("gw_params_narrow.dat", sep=" ", index=False)

    df_wide = add_mass_and_spin_BNS_parameters(df_narrow.copy(), "wide")
    df_wide.to_csv("gw_params_wide.dat", sep=" ", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

if __name__="__main__":
    main()
