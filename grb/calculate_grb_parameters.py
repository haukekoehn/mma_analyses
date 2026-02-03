import numpy as np
np.random.seed(167834359)
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma
import scipy.integrate as integrate
import tqdm

from joblib import Parallel, delayed
import sys
n_jobs = sys.argv[1]

msol_to_erg = 1.787624e54

# load EOS
m_val, r_val, l_val = np.loadtxt("../eos/RMF3_MRL.dat", unpack=True)
mtov = m_val.max()
r16 = np.interp(1.6, m_val, r_val)


def eta(chi_BH):
    OmegaH = chi_BH / (2*(1+np.sqrt(1-chi_BH**2)))
    return 0.011 * OmegaH**2 * (1+1.38*OmegaH**2-9.2*OmegaH**4)

def draw_thetaCore(n):
    theta_c = np.zeros(n)
    mask = np.random.choice([True, False], replace=True, size=n)
    theta_c[mask] = stats.truncnorm(loc=6.1, scale=3.2, a=-6.09/3.2, b=0).rvs(np.sum(mask))
    theta_c[~mask] = stats.truncnorm(loc=6.1, scale=9.3, a=0, b=(90-6.1)/9.3).rvs(np.sum(~mask))
    
    return np.deg2rad(theta_c)


def spectral_shape(nu, a=0.24, nu_p = 3):
    normalization = ((1+a)/nu_p)**(a+1) / gamma(1+a)
    return normalization * nu**a*np.exp(-(1+a)*nu/nu_p)

def gamma_gaussian(theta, thetaCore, gamma_c=500):
    return 1 + (gamma_c - 1)*np.exp(-0.5*(theta/thetaCore)**2)

def energy_gaussian(theta, thetaCore, energy):
    u_arr = np.linspace(0, 2, 100)
    integral_normfactor = integrate.simpson(y= np.sin(thetaCore*u_arr) *np.exp(-0.5*u_arr**2), x=u_arr)
    epsilon_c = energy / (2*np.pi*thetaCore) * integral_normfactor**(-1)

    return epsilon_c * np.exp(-0.5*(theta/thetaCore)**2)

def integrate_flux_density(Fnu, nu, numin, numax):

    mask = (nu >= numin) & (nu<=numax)
    return integrate.simpson(y=nu[mask]*Fnu[mask], x=np.log(nu[mask]))

def energy_gamma_iso(gamma_energy, thetaCore, iota, redshift):

    phi_arr = np.linspace(0, 2*np.pi-0.001, 60)
    theta_arr = np.linspace(0, thetaCore*2, 100)
    Phi, Theta = np.meshgrid(phi_arr, theta_arr)
    
    Gamma_arr = gamma_gaussian(Theta, thetaCore)
    E_arr = energy_gaussian(Theta, thetaCore, gamma_energy)

    cos_alpha = np.cos(Phi) * np.sin(Theta) *np.sin(iota) + np.cos(Theta) * np.cos(iota)
    delta = 1 / (Gamma_arr - np.sqrt(Gamma_arr**2 -1)* cos_alpha)

    nu_obs = np.geomspace(6, 40_000, 300) # keV
    nu_prime = (1+redshift) * nu_obs[None, None, :] / delta[:, :, None]

    S = spectral_shape(nu_prime)
    energy_doppler = (delta**2 / Gamma_arr) * E_arr
    integrand = S * energy_doppler[:, :, None] * np.sin(Theta)[:, :, None]
    
    integrand = integrate.simpson(y=integrand, x=phi_arr, axis=1)
    F_nu = integrate.simpson(y=integrand, x=theta_arr, axis=0)

    fermi_gbm = integrate_flux_density(F_nu, nu_obs, 8, 40_000)
    swift_bat = integrate_flux_density(F_nu, nu_obs, 15, 150)
    gecam = integrate_flux_density(F_nu, nu_obs, 6, 5000)
    
    return np.log10(fermi_gbm), np.log10(swift_bat), np.log10(gecam)

def calculate_Egamma_iso(gamma_energy, df):

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(j, gamma_energy, df) for j in tqdm.tqdm(range(df.shape[0]))
    )

    results = np.array(results)
    return results[:, 0], results[:, 1], results[:, 2]

def _worker(j, gamma_energy, df):
    np.seterr(divide = 'ignore') 

    if df.loc[j, "has_grb"]:
        return energy_gamma_iso(gamma_energy[j], df.loc[j, "thetaCore"], df.loc[j, "inclination_EM"], df.loc[j, "redshift"])
    return 0.0, 0.0, 0.0


def add_grb_parameters(df):

    has_BH = df["prompt_collapse"] < 4

    jet_energy = eta(df["chi_BH"])*(10**df["log10_mdisk"] - 10**df["log10_mej_wind"])* msol_to_erg / 2 # divide by 2 for counter jet
    breakout_energy = 0.05 * 0.78**2 *(10**df["log10_mej_dyn"]/2 *df["v_ej_dyn"]**2 + 10**df["log10_mej_wind"]/2 * df["v_ej_wind"]**2) * msol_to_erg / 2
    jet_energy -= breakout_energy

    df["has_grb"] = has_BH & (jet_energy > 0) # jet_energy > breakout_energy is always fulfilled in our sample
    df["thetaCore"] = draw_thetaCore(df.shape[0])
    df["eta_gamma"] = np.random.uniform(low=0.01, high=0.15, size=df.shape[0])

    gamma_energy = df["eta_gamma"]* jet_energy
    fermi_gbm, swift_bat, gecam = calculate_Egamma_iso(gamma_energy, df)
    df["log10_Egamma_fermi_gbm"] = fermi_gbm
    df["log10_Egamma_swift_bat"] = swift_bat
    df["log10_Egamma_gecam"] = gecam


    kinetic_energy = jet_energy - gamma_energy
    df["log10_Ekin_iso"] = np.log10(np.vectorize(energy_gaussian)(0, df["thetaCore"], kinetic_energy))
    df["log10_n0"] = np.random.uniform(low=-5, high=0, size=df.shape[0])

    return df

def main():
    df_narrow_kn_parameters = pd.read_csv("../kilonova/full_narrow_kn_parameters.dat", sep=" ")
    df_wide_kn_parameters = pd.read_csv("../kilonova/full_wide_kn_parameters.dat", sep=" ")

    df_narrow_kn_parameters = add_grb_parameters(df_narrow_kn_parameters)
    df_wide_kn_parameters = add_grb_parameters(df_wide_kn_parameters)

    df_narrow_kn_parameters[["inclination_EM", "has_grb", "thetaCore", "eta_gamma", "log10_Egamma_fermi_gbm", "log10_Egamma_swift_bat", "log10_Egamma_gecam", "log10_Ekin_iso", "log10_n0", "redshift", "luminosity_distance", "prompt_collapse"]].to_csv("full_narrow_grb_parameters.dat", sep=" ", index=False)
    df_wide_kn_parameters[["inclination_EM", "has_grb", "thetaCore", "eta_gamma", "log10_Egamma_fermi_gbm", "log10_Egamma_swift_bat", "log10_Egamma_gecam", "log10_Ekin_iso", "log10_n0", "redshift", "luminosity_distance", "prompt_collapse"]].to_csv("full_wide_grb_parameters.dat", sep=" ", index=False)
    


if __name__=="__main__":
    main()
