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
from gwtoolbox.tools_earth import set_cosmology, Tools
from bilby.gw.prior import UniformSourceFrame, Interped

from jesterTOV.inference.population.populations import RecycledBinary
from flax.training.train_state import TrainState
import optax

from train_nn import MLP, Config


n_shape = 2000
events = pd.read_csv("../events.dat", sep=" ")
n_events = events.shape[0]

with open("../../../det_probability/networks/ETL_snr_nn.pkl", "rb") as f:
    network_dict = pickle.load(f)
    params = network_dict["params"]
    config = network_dict["config"]
    net = MLP(config.layer_sizes)
    
    # Create train state without optimizer
    state = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(config.learning_rate))

with open("../../../det_probability/networks/ETL_snr_nn_scalers.pkl", "rb") as f:
    scaler_dict = pickle.load(f)
    Xscaler = scaler_dict["Xscaler"]
    yscaler = scaler_dict["yscaler"]

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

    dummy_prior = UniformSourceFrame(minimum=0.01, maximum=1.15, name="redshift")
    dummy_prior.yy *= np.interp(dummy_prior.xx, Toolset.population_class.zs, Toolset.population_class.Rm)
    prior = Interped(xx=dummy_prior.xx, yy=dummy_prior.yy, minimum=0.01, maximum=1.15, name="redshift")
    redshift = prior.sample(n_shape)
    luminosity_distance = Planck18.luminosity_distance(redshift).to_value()

    return luminosity_distance

theta_jn = np.arccos(np.random.uniform(-1, 1, size=n_shape))
ra = np.random.uniform(0, 2*np.pi, size=n_shape)
dec = np.arcsin(np.random.uniform(-1, 1, size=n_shape))
geocent_time = np.random.uniform(Time("2050-01-01", scale="tcg").gps, Time("2050-12-31", scale="tcg").gps, size=n_shape)
luminosity_distance = sample_merger_rate()


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
    return yscaler.inverse_transform(y)


def gw_detection_probability(m1: np.ndarray, m2: np.ndarray):

    X = np.stack((m1, m2, theta_jn, ra, dec, geocent_time)).T
    snr = jax.vmap(call_network)(X)

    snr = snr.flatten() * 100 / luminosity_distance
    return np.mean(snr>12)
    

def detection_prob_per_sample(posterior: dict, index: int):

    sample = {key: val[index] for key, val in posterior.items()}
    m1, m2 = RecycledBinary(jax.random.key(183902), sample, size=n_shape)
    total_mass = m1 + m2

    gw_det = gw_detection_probability(m1, m2)

    mthr = 3.0 #sample.get("k_coll", 1.3) * sample["masses_EOS"].max()
    em_det = np.mean(total_mass<mthr)

    return gw_det * em_det

def calculate_pdet(file: str):

    posterior = {}
    with h5py.File(file, "r") as f:

        for key in ["mu_1", "mu_2", "alpha", "sigma_1", "sigma_2", "m_max", "m_min", "k_coll"]:
            if key in f["posterior"]["parameters"].keys():
                posterior[key] = f["posterior"]["parameters"][key][:]
        
        posterior["log_prob"] = f["posterior"]["log_prob"][:]
        posterior["masses_EOS"] = f["posterior"]["derived_eos"]["masses_EOS"][:]
    
    
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

    calculate_pdet("./gw/outdir_gw/results.h5")
    calculate_pdet("./mm/outdir_mm/results.h5")
    calculate_pdet("./mm_full/outdir_mm/results.h5")


if __name__=="__main__":
    main()
