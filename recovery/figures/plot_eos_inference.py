import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    })
from matplotlib.colors import Normalize
import corner
import jax

from jesterTOV.utils import geometric_to_fm_inv3, geometric_to_MeV_fm_inv3

# load EOS
m_eos, r_eos, l_eos = np.loadtxt("../../eos/RMF3_MRL.dat", unpack=True)
mtov = m_eos.max()
n_eos, p_eos = np.loadtxt("../../eos/RMF3_cold_beta_lamb.d", unpack=True, usecols=[1,3])

latex_labels={"mu_1": "$\\mu_1$", "mu_2": "$\\mu_2$", "sigma_1": "$\\sigma_1$", "sigma_2": "$\\sigma_2$", "alpha": "$\\alpha$", "m_min": "$m_{\\rm{min}}$", "m_max": "$m_{\\rm{max}}$" }
fontsize=14

def load_posterior(file):

    posterior = {}

    with h5py.File(file) as f:

        for key in ["mu_1", "mu_2", "alpha", "sigma_1", "sigma_2", "m_max", "m_min", "k_coll"]:
            if key in f["posterior"]["parameters"].keys():
                posterior[key] = f["posterior"]["parameters"][key][:]
        
        posterior["masses_EOS"] = f["posterior"]["derived_eos"]["masses_EOS"][:]
        posterior["radii_EOS"] = f["posterior"]["derived_eos"]["radii_EOS"][:]
        posterior["lambdas_EOS"] = f["posterior"]["derived_eos"]["Lambdas_EOS"][:]
        posterior["pressures_EOS"] = f["posterior"]["derived_eos"]["p"][:] * geometric_to_MeV_fm_inv3
        posterior["densities_EOS"] = f["posterior"]["derived_eos"]["n"][:] * geometric_to_fm_inv3

        posterior["log_prob"] = f["posterior"]["log_prob"][:]

        if os.path.exists(file.parent / "inverse_pdet.dat"):
            inverse_pdet = np.loadtxt(file.parent / "inverse_pdet.dat")
            posterior["weights"] = inverse_pdet
        else:
            posterior["weights"] = np.full(posterior["log_prob"].shape[0], 1/posterior["log_prob"].shape[0])

    return posterior

def double_gaussian_pdf(x, params):
    first_peak = np.exp(-0.5 * (x-params["mu_1"])**2 / params["sigma_1"]**2) * 1/np.sqrt(2*np.pi*params["sigma_1"]**2)
    second_peak = np.exp(-0.5 * (x-params["mu_2"])**2 / params["sigma_2"]**2) * 1/np.sqrt(2*np.pi*params["sigma_2"]**2)
    return (1-params["alpha"]) * first_peak + params["alpha"] * second_peak

def double_gaussian_cdf(x, params):
    first_peak = stats.norm.cdf(x, loc=params["mu_1"], scale=params["sigma_1"])
    second_peak = stats.norm.cdf(x, loc=params["mu_2"], scale=params["sigma_2"])
    return (1-params["alpha"]) * first_peak + params["alpha"] * second_peak

def uniform_pdf(x, params):
    return np.where((x<params["m_max"]) & (x>params["m_min"]), 1/(params["m_max"]-params["m_min"]), 0)

def uniform_cdf(x, params):
    cdf = (x-params["m_min"]) / (params["m_max"] - params["m_min"])
    cdf = np.maximum(cdf, 0)
    cdf = np.minimum(cdf, 1)
    return cdf

def RecycledBinary(m_arr, params):

    pdf_m1 = double_gaussian_pdf(m_arr, params) * uniform_cdf(m_arr, params) + uniform_pdf(m_arr, params) * double_gaussian_cdf(m_arr, params)
    pdf_m2 = uniform_pdf(m_arr, params) * (1-double_gaussian_cdf(m_arr, params)) + double_gaussian_pdf(m_arr, params) * (1-uniform_cdf(m_arr, params))

    return pdf_m1, pdf_m2


def MassRatioPowerLaw(m_arr, params):

    return pdf_m1, pdf_m2


def get_quantiles(xs, x_val, y_val, weights, alphas=[0.025, 0.16, 0.5, 0.84, 0.975]):

    y_interp = np.vstack([
                          np.interp(xs, x_val[j, :], y_val[j,:], left=np.nan, right=np.nan) 
                          for j in range(y_val.shape[0])
    ])
    
    quantiles = np.vstack([
                           np.nanquantile(y_interp[:, j], q=alphas, weights=weights, method="inverted_cdf")
                           for j in range(xs.shape[0])
    ])

    return xs, quantiles, alphas


def plot_quantiles(ax, x, quantiles, color, fillx: bool= False):

    if fillx:
        ax.fill_betweenx(x, quantiles[:, 0], quantiles[:, 4], color=color, alpha=0.15)
        #ax.plot(quantiles[:, 1], x, color=color, linestyle="dashed")
        #ax.plot(quantiles[:, 3], x, color=color, linestyle="dashed")
        ax.plot(quantiles[:, 2], x, color=color)
    else:
        ax.fill_between(x, quantiles[:, 0], quantiles[:, 4], color=color, alpha=0.15)
        #ax.plot(x, quantiles[:, 1], color=color, linestyle="dashed")
        #ax.plot(x, quantiles[:, 3], color=color, linestyle="dashed")
        ax.plot(x, quantiles[:, 2], color=color)
    

def plot_mr(ax, posterior, color, ):

    best_ind = posterior["log_prob"].argmax()
    x = np.linspace(1, 2.5, 100) # masses to plot for

    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["radii_EOS"], posterior["weights"])
    plot_quantiles(ax, x, quantiles, color, fillx=True)

    # plot truth
    ax.plot(r_eos, m_eos, color="red", zorder=3)
    #ax.plot(eos_posterior["radii_EOS"][best_ind], eos_posterior["masses_EOS"][best_ind], color=color, linestyle="solid")
    
    ax.set_ylabel("$M$ [$M_\\odot$]", fontsize=fontsize)
    ax.set_xlabel("$R$ [km]", fontsize=fontsize)
    ax.set(ylim=(1, 2.25), xlim=(10.5, 13))

def plot_ml(ax, posterior, color):

    best_ind = posterior["log_prob"].argmax()
    x = np.linspace(1, 2.5, 100) # masses to plot for

    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["lambdas_EOS"], posterior["weights"])
    plot_quantiles(ax, x, quantiles, color, fillx=True)

    # plot truth
    ax.plot(l_eos, m_eos, color="red", zorder=3)
    
    ax.set_ylabel("$M$ [$M_\\odot$]", fontsize=fontsize)
    ax.set_xlabel("$\\Lambda$", fontsize=fontsize)
    ax.set(ylim=(1, 2), xscale="log", xlim=(20, 2e3))

def plot_pressure(ax, posterior, color):

    best_ind = posterior["log_prob"].argmax()
    x = np.linspace(0.08, 1.6, 100) # densities to plot for

    x, quantiles, _ = get_quantiles(x, posterior["densities_EOS"], posterior["pressures_EOS"], posterior["weights"])
    plot_quantiles(ax, x/0.16, quantiles, color)

    # plot truth
    ax.plot(n_eos/0.16, p_eos, color="red", zorder=3)

    ax.set_ylabel("$p$ [MeV fm$^{-3}$]", fontsize=fontsize)
    ax.set_xlabel("$n$ [$n_{\\mathrm{sat}}$]", fontsize=fontsize)
    ax.set(xlim=(0.5, 8), yscale="log", ylim=(0.1, 2e3))


def plot_pop(ax, posterior, color):

    best_ind = posterior["log_prob"].argmax()
    x = np.linspace(1, 2, 100) # masses to plot for

    if "mu_1" in posterior:
        pop_model = RecycledBinary
        truths = dict(mu_1=1.34, mu_2=1.43, sigma_1=0.02, sigma_2=0.15, alpha=0.68, m_min=1.16, m_max=1.42, k_coll=1.3)
    else:
        pop_model = MassRatioPowerLaw
        truths = dict(m_min=1.1, m_max=2.1, alpha=2.0, k_coll=1.3)
    
    X = np.tile(x, (posterior["log_prob"].shape[0], 1))
    Y_pdf_m1 = np.zeros_like(X)
    Y_pdf_m2 = np.zeros_like(X)

    for j in range(Y_pdf_m1.shape[0]):
        sample_point = {key: val[j] for key, val in posterior.items()}
        Y_pdf_m1[j], Y_pdf_m2[j] = pop_model(X[j], sample_point)

    pdf_m1_truth, pdf_m2_truth = pop_model(x, truths)

    # plot m1
    x, quantiles_m1, _ = get_quantiles(x, X, Y_pdf_m1, posterior["weights"])
    plot_quantiles(ax[0], x, quantiles_m1, color)
    ax[0].plot(x, pdf_m1_truth, color="red", zorder=3)
    ax[0].set_xlabel("$m_1$ [$M_\\odot$]", fontsize=fontsize)
    ax[0].set_ylabel("pop. density", fontsize=fontsize)
    ax[0].set(xlim=(1, 2), yscale="log", ylim=(0.1, 20))

    # plot m2
    x, quantiles_m2, _ = get_quantiles(x, X, Y_pdf_m2, posterior["weights"])
    plot_quantiles(ax[1], x, quantiles_m2, color)
    ax[1].plot(x, pdf_m2_truth, color="red", zorder=3)
    ax[1].set_xlabel("$m_2$ [$M_\\odot$]", fontsize=fontsize)
    ax[1].set_ylabel("pop. density", fontsize=fontsize)
    ax[1].set(xlim=(1, 2), yscale="log", ylim=(0.1, 20))


def main(directory: str):

    directory = Path(directory)
    name = directory.name

    events = pd.read_csv(directory / "events.dat", sep=" ")

    posterior_gw = load_posterior(directory / "inference_eos" / "gw" / "outdir_gw" / "results.h5")
    posterior_mm = load_posterior(directory / "inference_eos" / "mm" / "outdir_mm" / "results.h5")

    fig, ax = plt.subplots(5, 1, figsize=(5, 22))
    fig.subplots_adjust(hspace=0.2)


    plot_ml(ax[0], posterior_gw, color="purple")
    plot_ml(ax[0], posterior_mm, color="orange")

    plot_mr(ax[1], posterior_gw, color="purple")
    plot_mr(ax[1], posterior_mm, color="orange")

    plot_pressure(ax[2], posterior_gw, color="purple")
    plot_pressure(ax[2], posterior_mm, color="orange")
    
    plot_pop(ax[3:], posterior_gw, color="purple")
    plot_pop(ax[3:], posterior_mm, color="orange")

    fig.savefig(f"{name}.pdf", dpi=250, bbox_inches="tight")



if __name__=="__main__":
    main(sys.argv[1])



