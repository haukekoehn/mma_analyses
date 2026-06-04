import pickle
import sys
sys.path.insert(0, "../../../det_probability")
import tqdm


import h5py
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from astropy.cosmology import Planck18
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import astropy.units as u

from gwtoolbox.tools_earth import set_cosmology, Tools
from bilby.gw.prior import UniformSourceFrame, Interped

from fiesta.inference.lightcurve_model import FluxModel

from jesterTOV.inference.population.populations import RecycledBinary, MassRatioPowerLaw
from jesterTOV.utils import redshift_to_luminosity_distance, solar_mass_in_meter, lambda1_lambda2_to_lambda_tilde
from jesterTOV.inference.transforms.multimessenger_transform import dynamic_mass_fitting_prompt_collapse, dynamic_mass_fitting, log10_disk_mass_fitting_prompt_collapse, log10_disk_mass_fitting

from flax.training.train_state import TrainState
import optax

from train_nn import MLP, Config

ETL_location = EarthLocation(lat=40.5*u.deg, lon=9.35*u.deg, height=100*u.m)
ETT_location = EarthLocation(lat=50.795483*u.deg, lon=5.848956*u.deg, height=100*u.m)

### CONFIG ###

n_shape = 2000
events = pd.read_csv("../../../eos_inference/narrow_ETT/events.dat", sep=" ")
n_events = events.shape[0]


location = ETT_location
mass_model = RecycledBinary

#############

with open("../../../det_probability/networks/ETT_snr_nn.pkl", "rb") as f:
    network_dict = pickle.load(f)
    params = network_dict["params"]
    config = network_dict["config"]
    net = MLP(config.layer_sizes)
    
    # Create train state without optimizer
    state = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(config.learning_rate))

with open("../../../det_probability/networks/ETT_snr_nn_scalers.pkl", "rb") as f:
    scaler_dict = pickle.load(f)
    Xscaler = scaler_dict["Xscaler"]
    yscaler = scaler_dict["yscaler"]

def sample_merger_rate(
        R0: float = 50.,
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

    dummy_prior = UniformSourceFrame(minimum=0.01, maximum=1.2, name="redshift")
    dummy_prior.yy *= np.interp(dummy_prior.xx, Toolset.population_class.zs, Toolset.population_class.Rm)
    prior = Interped(xx=dummy_prior.xx, yy=dummy_prior.yy, minimum=0.01, maximum=1.15, name="redshift")
    redshifts = prior.sample(n_shape)
    return redshifts

def convert_to_altaz(ra, dec, geocent_time, location):
    time = Time(geocent_time, format="gps")
    target = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
    altaz_frame = AltAz(obstime=time, location=location)
    
    sky_coords = target.transform_to(altaz_frame)
    return sky_coords.alt.rad, sky_coords.az.rad

theta_jn = np.arccos(np.random.uniform(-1, 1, size=n_shape))
psi = np.random.uniform(0, np.pi, size=n_shape)

ra = np.random.uniform(0, 2*np.pi, size=n_shape)
dec = np.arcsin(np.random.uniform(-1, 1, size=n_shape))
geocent_time = np.random.uniform(Time("2050-01-01", scale="tcg").gps, Time("2050-12-31", scale="tcg").gps, size=n_shape)
alt, az = convert_to_altaz(ra, dec, geocent_time, ETL_location)

redshifts = sample_merger_rate()

kn_model = FluxModel("Bu2026_MLP", filters=["lssti"])
inclination_EM = np.arccos(np.abs(np.cos(theta_jn)))
v_ej_dyn = np.random.uniform(0.12, 0.35, size=n_shape)
v_ej_wind = np.random.uniform(0.05, 0.15, size=n_shape)
Ye_dyn = np.random.uniform(0.15, 0.35, size=n_shape)
Ye_wind = np.random.uniform(0.2, 0.4, size=n_shape)


#####
# functions 
#####


def softmax(logp):

    logp -= logp.max()
    p = np.exp(logp)
    p /= np.sum(p)

    return p

@jax.jit
def call_network(X):

    X = Xscaler.transform(X)
    y = state.apply_fn({"params": state.params}, X, train=False)
    return 10**yscaler.inverse_transform(y)

def truncated_random_normal(lower, upper, mu, sigma, size: int):

    lower = (lower-mu)/sigma
    upper = (upper-mu)/sigma
    
    samples = jax.random.truncated_normal(jax.random.key(19), lower, upper, shape=size)

    return samples * sigma + mu



def multi_messenger_conversion(
            mass_1,
            mass_2,
            params: dict[str, float]
        ) -> dict[str, np.ndarray]:
        
        masses_EOS = params["masses_EOS"]
        radii_EOS = params["radii_EOS"]
        Lambdas_EOS = params["Lambdas_EOS"]
        mtov = jnp.max(masses_EOS)
        r16 = jnp.interp(1.6, masses_EOS, radii_EOS)
        m_coll = params.get("k_coll", 1.3) * mtov
        m_coll = jnp.maximum(m_coll, 2.9)

        prompt_collapse = m_coll < (mass_1 + mass_2)

        lambda_1 = jnp.interp(mass_1, masses_EOS, Lambdas_EOS)
        lambda_2 = jnp.interp(mass_2, masses_EOS, Lambdas_EOS)
        radii_1 = jnp.interp(mass_1, masses_EOS, radii_EOS)
        radii_2 = jnp.interp(mass_2, masses_EOS, radii_EOS)
        compactness_1 = mass_1 / radii_1 * solar_mass_in_meter * 1e-3
        compactness_2 = mass_2 / radii_2 * solar_mass_in_meter * 1e-3

        mej_dyn = jnp.where(
            prompt_collapse, 
            dynamic_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2), 
            dynamic_mass_fitting(mass_1, mass_2, compactness_1, compactness_2)
        )
        log10_mej_dyn = jnp.log10(mej_dyn)
       

        log10_mdisk = jnp.where(
            prompt_collapse,
            log10_disk_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2),
            log10_disk_mass_fitting(mass_1+mass_2, mass_1/mass_2, mtov, r16)
        )

        zeta = jnp.where(
            prompt_collapse,
            truncated_random_normal(0, 0.4, 0.2, 0.1, n_shape),
            truncated_random_normal(0, 0.8, 0.5, 0.2, n_shape)
        )

        log10_mej_wind = jnp.log10(zeta) + log10_mdisk
       
        return log10_mej_dyn, log10_mej_wind

@jax.jit
def gw_detection_probability(m1: np.ndarray, m2: np.ndarray, luminosity_distance: np.ndarray):

    X = jnp.stack((m1, m2, theta_jn, alt, az, psi)).T
    snr = jax.vmap(call_network)(X)

    snr = snr.flatten() * 100 / luminosity_distance
    return jnp.mean(snr>=12), snr >=12

@jax.jit
def em_detection_probability(m1: np.ndarray, m2: np.ndarray, luminosity_distance: np.ndarray, sample: dict, gw_detected: np.ndarray):

    log10_mej_dyn, log10_mej_wind = multi_messenger_conversion(m1, m2, sample)

    params = dict(inclination_EM=inclination_EM,
                  log10_mej_dyn=log10_mej_dyn, 
                  log10_mej_wind=log10_mej_wind,
                  v_ej_dyn=v_ej_dyn,
                  v_ej_wind=v_ej_wind,
                  Ye_dyn=Ye_dyn,
                  Ye_wind=Ye_wind,
                  luminosity_distance=luminosity_distance,
                  redshift=redshifts)
    
    times, mags = kn_model.vpredict(params)
    kn_detectable = jnp.any(mags["lssti"] < 25.6, axis=1)

    kn_detectable = jnp.where(
        gw_detected,
        kn_detectable,
        0
    )

    return jnp.sum(kn_detectable) / jnp.sum(gw_detected)
    

def detection_prob_per_sample(posterior: dict, index: int):

    sample = {key: val[index] for key, val in posterior.items()}
    m1, m2 = mass_model(jax.random.key(183902), sample, size=n_shape)

    H0 = sample.get("H0", Planck18.H0.value)
    Omega0 = sample.get("Omega0", Planck18.Om0)
    luminosity_distance = redshift_to_luminosity_distance(redshifts, H0, Omega0)

    gw_prob, gw_detected = gw_detection_probability(m1, m2, luminosity_distance)

    em_prob = em_detection_probability(
        m1, 
        m2, 
        luminosity_distance,
        sample,
        gw_detected,
    )

    return gw_prob * em_prob

def calculate_pdet(file: str):

    posterior = {}
    with h5py.File(file, "r") as f:

        for key in ["mu_1", "mu_2", "alpha", "sigma_1", "sigma_2", "m_max", "m_min", "k_coll", "H0", "Omega0"]:
            if key in f["posterior"]["parameters"].keys():
                posterior[key] = f["posterior"]["parameters"][key][:]
        
        posterior["log_prob"] = f["posterior"]["log_prob"][:]
        posterior["masses_EOS"] = f["posterior"]["derived_eos"]["masses_EOS"][:]
        posterior["radii_EOS"] = f["posterior"]["derived_eos"]["radii_EOS"][:]
        posterior["Lambdas_EOS"] = f["posterior"]["derived_eos"]["Lambdas_EOS"][:]
    
    nsamp = posterior["log_prob"].shape[0]
    pdet = np.zeros(nsamp)

    for j in tqdm.tqdm(range(nsamp)):
        pdet[j] = detection_prob_per_sample(posterior, j)

    inverse_pdet = softmax(-n_events * np.log(pdet))

    outfile = file.split("/")[:-1]
    outfile = "/".join(outfile)
    outfile += "/inverse_pdet.dat"
    np.savetxt(outfile, inverse_pdet)





def main():

    calculate_pdet("./outdir/results.h5")


if __name__=="__main__":
    main()
