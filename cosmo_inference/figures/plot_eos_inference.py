import os
from pathlib import Path
import sys
from copy import deepcopy

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

latex_labels={"mu_1": "$\\mu_1$", "mu_2": "$\\mu_2$", "sigma_1": "$\\sigma_1$", "sigma_2": "$\\sigma_2$", "alpha": "$\\alpha$", "m_min": "$m_{\\rm{min}}$", "m_max": "$m_{\\rm{max}}$", "H0": "$H_0$", "Omega0": "$\\Omega_0$", "beta_1": "$\\beta_1$", "beta_2": "$\\beta_2$"}
fontsize=14

def load_posterior(file):

    posterior = {}

    with h5py.File(file) as f:

        for key in ["mu_1", "mu_2", "alpha", "sigma_1", "sigma_2", "m_max", "m_min", "k_coll", "H0", "Omega0", "beta_1", "beta_2"]:
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
    return params["alpha"] * first_peak + (1 - params["alpha"]) * second_peak

def double_gaussian_cdf(x, params):
    first_peak = stats.norm.cdf(x, loc=params["mu_1"], scale=params["sigma_1"])
    second_peak = stats.norm.cdf(x, loc=params["mu_2"], scale=params["sigma_2"])
    return params["alpha"] * first_peak + (1 - params["alpha"]) * second_peak

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

    m_min = params["m_min"]
    m_max = params["m_max"]
    alpha = params["alpha"]

    if alpha != 1:
        normalization_constant = (1+alpha) * (
            m_max**2/2 
            + (1+alpha)/(2*(1-alpha))*m_min**2 
            - (m_min**(alpha+1) * m_max**(1-alpha)) / (1-alpha) 
        )**(-1)

        pdf_m2 = np.where(
            (m_arr >= m_min) & (m_arr <= m_max),
            normalization_constant * m_arr**alpha * (m_max**(1-alpha) - m_arr**(1-alpha)) / (1-alpha),
            0
        )

    else: 
        normalization_constant = (alpha+1) * (
            0.5 *(m_max**2 - m_min**2)
            - m_min**(alpha+1) * np.log(m_max/m_min)
        )**(-1)
        pdf_m2 = np.where(
            (m_arr >= m_min) & (m_arr <= m_max),
            normalization_constant * m_arr**alpha * np.log(m_max/m_arr),
            0
        )
    
    pdf_m1 = np.where(
            (m_arr >= m_min) & (m_arr <= m_max),
            normalization_constant * (m_arr**(1+alpha)-m_min**(1+alpha)) / (alpha+1) * m_arr**(-alpha),
            0
    )

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


def plot_quantiles(ax, x, quantiles, color, fillx: bool= False, plot_median=True, alpha=0.15):

    if fillx:
        ax.fill_betweenx(x, quantiles[:, 0], quantiles[:, 4], color=color, alpha=alpha)
        #ax.plot(quantiles[:, 1], x, color=color, linestyle="dashed")
        #ax.plot(quantiles[:, 3], x, color=color, linestyle="dashed")
        ax.plot(quantiles[:, 2], x, color=color) if plot_median else None
    else:
        ax.fill_between(x, quantiles[:, 0], quantiles[:, 4], color=color, alpha=alpha)
        #ax.plot(x, quantiles[:, 1], color=color, linestyle="dashed")
        #ax.plot(x, quantiles[:, 3], color=color, linestyle="dashed")
        ax.plot(x, quantiles[:, 2], color=color) if plot_median else None
    

def plot_mr(ax, posterior, color, plot_bestfit=True, alpha=0.15):

    best_ind = (posterior["log_prob"] +  np.log(posterior["weights"])).argmax()
    x = np.linspace(1, 2.5, 100) # masses to plot for

    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["radii_EOS"], posterior["weights"])
    plot_quantiles(ax, x, quantiles, color, fillx=True, plot_median=not plot_bestfit, alpha=alpha)

    # plot best fit 
    if plot_bestfit:
        ax.plot(posterior["radii_EOS"][best_ind], posterior["masses_EOS"][best_ind], color=color, linestyle="solid")

    # plot truth
    ax.plot(r_eos, m_eos, color="red", zorder=3)
    
    ax.set_ylabel("$M$ [$M_\\odot$]", fontsize=fontsize, labelpad=2)
    ax.set_xlabel("$R$ [km]", fontsize=fontsize)
    ax.set(ylim=(1, 2.25), xlim=(10.5, 13))

def plot_mr_nofill(ax, posterior, color, plot_bestfit=True, alpha=0.15):

    best_ind = (posterior["log_prob"] +  np.log(posterior["weights"])).argmax()
    x = np.linspace(1, 2.5, 100) # masses to plot for

    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["radii_EOS"], posterior["weights"])
    ax.plot(quantiles[:,0], x, color=color, linestyle="dashed")
    ax.plot(quantiles[:,4], x, color=color, linestyle="dashed")

    # plot best fit 
    if plot_bestfit:
        ax.plot(posterior["radii_EOS"][best_ind], posterior["masses_EOS"][best_ind], color=color, linestyle="solid")

    # plot truth
    ax.plot(r_eos, m_eos, color="red", zorder=3)
    
    ax.set_ylabel("$M$ [$M_\\odot$]", fontsize=fontsize, labelpad=2)
    ax.set_xlabel("$R$ [km]", fontsize=fontsize)
    ax.set(ylim=(1, 2.25), xlim=(10.5, 13))

def plot_ml(ax, posterior, color, plot_bestfit=True):

    best_ind = (posterior["log_prob"] +  np.log(posterior["weights"])).argmax()
    x = np.linspace(1, 2.5, 100) # masses to plot for

    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["lambdas_EOS"], posterior["weights"])
    plot_quantiles(ax, x, quantiles, color, fillx=True, plot_median=not plot_bestfit)

    # plot best fit 
    if plot_bestfit:
        ax.plot(posterior["lambdas_EOS"][best_ind], posterior["masses_EOS"][best_ind], color=color, linestyle="solid")
    # plot truth
    ax.plot(l_eos, m_eos, color="red", zorder=3)
    
    ax.set_ylabel("$M$ [$M_\\odot$]", fontsize=fontsize, labelpad=2)
    ax.set_xlabel("$\\Lambda$", fontsize=fontsize)
    ax.set(ylim=(1, 2), xscale="log", xlim=(20, 2e3))

def plot_pressure(ax, posterior, color, plot_bestfit=True):

    best_ind = (posterior["log_prob"] +  np.log(posterior["weights"])).argmax()
    x = np.linspace(0.08, 1.6, 100) # densities to plot for

    x, quantiles, _ = get_quantiles(x, posterior["densities_EOS"], posterior["pressures_EOS"], posterior["weights"])
    plot_quantiles(ax, x/0.16, quantiles, color, plot_median=not plot_bestfit)

    # plot best fit 
    if plot_bestfit:
        ax.plot(posterior["densities_EOS"][best_ind] / 0.16, posterior["pressures_EOS"][best_ind], color=color, linestyle="solid")

    # plot truth
    ax.plot(n_eos/0.16, p_eos, color="red", zorder=3)

    ax.set_ylabel("$p$ [MeV fm$^{-3}$]", fontsize=fontsize, labelpad=-2)
    ax.set_xlabel("$n$ [$n_{\\mathrm{sat}}$]", fontsize=fontsize)
    ax.set(xlim=(0.5, 8), yscale="log", ylim=(0.1, 2e3))


def plot_pop(ax, posterior, color, plot_bestfit=False):

    best_ind = (posterior["log_prob"] + np.log(posterior["weights"])).argmax()
    x = np.linspace(1, 2.1, 100) # masses to plot for

    if "mu_1" in posterior:
        pop_model = RecycledBinary
        truths = dict(mu_1=1.34, mu_2=1.43, sigma_1=0.02, sigma_2=0.15, alpha=0.68, m_min=1.16, m_max=1.42, k_coll=1.3)
    else:
        pop_model = MassRatioPowerLaw
        truths = dict(m_min=1.1, m_max=2.0, alpha=2.0, k_coll=1.3)
    
    X = np.tile(x, (posterior["log_prob"].shape[0], 1))
    Y_pdf_m1 = np.zeros_like(X)
    Y_pdf_m2 = np.zeros_like(X)

    for j in range(Y_pdf_m1.shape[0]):
        sample_point = {key: val[j] for key, val in posterior.items()}
        Y_pdf_m1[j], Y_pdf_m2[j] = pop_model(X[j], sample_point)

    pdf_m1_truth, pdf_m2_truth = pop_model(x, truths)
    pdf_m1_bestfit, pdf_m2_bestfit = pop_model(x, {key: val[best_ind] for key, val in posterior.items()})

    # plot m1
    x, quantiles_m1, _ = get_quantiles(x, X, Y_pdf_m1, posterior["weights"])
    plot_quantiles(ax[0], x, quantiles_m1, color, plot_median=not plot_bestfit)
    ax[0].plot(x, pdf_m1_bestfit) if plot_bestfit else None
    ax[0].plot(x, pdf_m1_truth, color="red", zorder=3)
    ax[0].set_xlabel("$m_1$ [$M_\\odot$]", fontsize=fontsize)
    ax[0].set_ylabel("pop. density", fontsize=fontsize, labelpad=-4)
    ax[0].set(xlim=(1, 2.1), yscale="log", ylim=(0.1, quantiles_m1.max() * 2))

    # plot m2
    x, quantiles_m2, _ = get_quantiles(x, X, Y_pdf_m2, posterior["weights"])
    plot_quantiles(ax[1], x, quantiles_m2, color, plot_median=not plot_bestfit)
    ax[1].plot(x, pdf_m2_bestfit) if plot_bestfit else None
    ax[1].plot(x, pdf_m2_truth, color="red", zorder=3)
    ax[1].set_xlabel("$m_2$ [$M_\\odot$]", fontsize=fontsize)
    ax[1].set_ylabel("pop. density", fontsize=fontsize, labelpad=-4)
    ax[1].set(xlim=(1, 2.1), yscale="log", ylim=(0.1, quantiles_m2.max() * 2))

def plot_pop_nofill(ax, posterior, color, plot_bestfit=False):

    best_ind = (posterior["log_prob"] + np.log(posterior["weights"])).argmax()
    x = np.linspace(1, 2.1, 100) # masses to plot for

    if "mu_1" in posterior:
        pop_model = RecycledBinary
        truths = dict(mu_1=1.34, mu_2=1.43, sigma_1=0.02, sigma_2=0.15, alpha=0.68, m_min=1.16, m_max=1.42, k_coll=1.3)
    else:
        pop_model = MassRatioPowerLaw
        truths = dict(m_min=1.1, m_max=2.0, alpha=2.0, k_coll=1.3)
    
    X = np.tile(x, (posterior["log_prob"].shape[0], 1))
    Y_pdf_m1 = np.zeros_like(X)
    Y_pdf_m2 = np.zeros_like(X)

    for j in range(Y_pdf_m1.shape[0]):
        sample_point = {key: val[j] for key, val in posterior.items()}
        Y_pdf_m1[j], Y_pdf_m2[j] = pop_model(X[j], sample_point)

    pdf_m1_truth, pdf_m2_truth = pop_model(x, truths)
    pdf_m1_bestfit, pdf_m2_bestfit = pop_model(x, {key: val[best_ind] for key, val in posterior.items()})

    # plot m1
    x, quantiles_m1, _ = get_quantiles(x, X, Y_pdf_m1, posterior["weights"])
    ax[0].plot(x, quantiles_m1[:, 0], color=color, linestyle="dashed")
    ax[0].plot(x, quantiles_m1[:, 4], color=color, linestyle="dashed")
    ax[0].plot(x, pdf_m1_bestfit) if plot_bestfit else None
    ax[0].plot(x, pdf_m1_truth, color="red", zorder=3)
    ax[0].set_xlabel("$m_1$ [$M_\\odot$]", fontsize=fontsize)
    ax[0].set_ylabel("pop. density", fontsize=fontsize, labelpad=-4)
    ax[0].set(xlim=(1, 2.1), yscale="log", ylim=(0.1, quantiles_m1.max() * 2))

    # plot m2
    x, quantiles_m2, _ = get_quantiles(x, X, Y_pdf_m2, posterior["weights"])
    ax[1].plot(x, quantiles_m2[:, 0], color=color, linestyle="dashed")
    ax[1].plot(x, quantiles_m2[:, 4], color=color, linestyle="dashed")
    ax[1].plot(x, pdf_m2_bestfit) if plot_bestfit else None
    ax[1].plot(x, pdf_m2_truth, color="red", zorder=3)
    ax[1].set_xlabel("$m_2$ [$M_\\odot$]", fontsize=fontsize)
    ax[1].set_ylabel("pop. density", fontsize=fontsize, labelpad=-4)
    ax[1].set(xlim=(1, 2.1), yscale="log", ylim=(0.1, quantiles_m2.max() * 2))

def corner_plot(posterior, parameter_names, fig=None, color="purple"):

    if fig is None:
        n_params = len(parameter_names)
        fig, _ = plt.subplots(n_params, n_params, figsize=(10/7*n_params, 10/7*n_params))

    if "mu_1" in posterior:
        pop_model = RecycledBinary
        truths = dict(mu_1=1.34, mu_2=1.43, sigma_1=0.02, sigma_2=0.15, alpha=0.68, m_min=1.16, m_max=1.42, k_coll=1.3)
    else:
        pop_model = MassRatioPowerLaw
        truths = dict(m_min=1.1, m_max=2.0, alpha=2.0, k_coll=1.3)
    
    truths.update(dict(H0=67.66, Omega0=0.30966, beta_1=500, beta_2=3000))

    labels = [latex_labels.get(p,p) for p in parameter_names]

    data = {p: posterior[p] for p in parameter_names}

    corner.corner(data,
                  weights=posterior['weights'],
                  smooth=True, 
                  levels=[0.68, 0.95],
                  fig=fig, 
                  plot_density=False,
                  fill_contours=True,
                  plot_datapoints=False,
                  truths=truths,
                  color=color,
                  truth_color="red",
                  labels=labels,
                  hist_kwargs=dict(density=True))
    
    return fig

def plot_cosmo(ax, posterior, color="orange", levels=[0.68, 0.95], fill_contours=True, labelpad=6, linestyle="solid"):

    truths = dict(H0=67.66, Omega0=0.30966)
    labels = [latex_labels[p] for p in ["H0", "Omega0"]]

    corner.hist2d(
        posterior["H0"], posterior["Omega0"], 
        weights=posterior["weights"], 
        ax=ax,
        smooth=True, 
        levels=levels,
        plot_density=False,
        no_fill_contours=not fill_contours,
        fill_contours=fill_contours,
        plot_datapoints=False,
        truths=truths,
        color=color,
        truth_color="red",
        labels=labels,
        hist_kwargs=dict(density=True),
        contour_kwargs=dict(linestyles=linestyle, zorder=2)
    )

    ax.vlines([67.66], *ax.get_ylim(), color="red")
    ax.hlines([0.30966], *ax.get_xlim(), color="red")

    ax.set_xlabel("$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", labelpad=labelpad, zorder=0)
    ax.set_ylabel("$\\Omega_0$", labelpad=labelpad, zorder=0)

    

def r14_confidence_interval(posterior):
    x = np.linspace(1, 2.5, 100) # masses to plot for
    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["radii_EOS"], posterior["weights"])

    lower = np.interp(1.4, x, quantiles[:, 0])
    median = np.interp(1.4, x, quantiles[:, 2])
    upper = np.interp(1.4, x, quantiles[:, 4])
    plus = upper-median
    minus = median-lower

    return f"{{{median:.2f}}}^{{+{plus:.2f}}}_{{-{minus:.2f}}}"

def l14_confidence_interval(posterior):
    x = np.linspace(1, 2.5, 100) # masses to plot for
    x, quantiles, _ = get_quantiles(x, posterior["masses_EOS"], posterior["lambdas_EOS"], posterior["weights"])

    lower = np.interp(1.4, x, quantiles[:, 0])
    median = np.interp(1.4, x, quantiles[:, 2])
    upper = np.interp(1.4, x, quantiles[:, 4])
    plus = upper-median
    minus = median-lower

    return f"{{{median:.0f}}}^{{+{plus:.0f}}}_{{-{minus:.0f}}}"

def mtov_confidence_interval(posterior):

    mtovs = np.max(posterior["masses_EOS"], axis=1)
    lower, median, upper = np.quantile(mtovs, [0.025, 0.5, 0.975], weights=posterior["weights"], method="inverted_cdf")
    plus = upper-median
    minus = median-lower
    return f"{{{median:.2f}}}^{{+{plus:.2f}}}_{{-{minus:.2f}}}"

def plot_all(fig, ax, posterior, color):


    plot_ml(ax[0], posterior, color)
    plot_mr(ax[1], posterior, color)
    plot_pressure(ax[2], posterior, color)
    plot_pop(ax[3:], posterior, color)

    return fig, ax

def main(directory: str):

    directory = Path(directory)
    name = directory.name

    events = pd.read_csv(directory / "events.dat", sep=" ")

    posterior_gw = load_posterior(directory / "inference_eos" / "gw" / "outdir_gw" / "results.h5")
    posterior_gw_unweighed = deepcopy(posterior_gw)
    posterior_gw_unweighed['weights'] = np.full((7000,), 1/7000) 

    posterior_mm = load_posterior(directory / "inference_eos" / "mm" / "outdir_mm" / "results.h5")
    posterior_mm_unweighed = deepcopy(posterior_mm)
    posterior_mm_unweighed['weights'] = np.full((7000,), 1/7000) 

    #posterior_mm_full = load_posterior(directory / "inference_eos" / "mm_full" / "outdir_mm" / "results.h5")
    #posterior_mm_full_unweighed = deepcopy(posterior_mm_full)
    #posterior_mm_full_unweighed['weights'] = np.full((7000,), 1/7000)

    #posterior_mm_cheating = load_posterior(directory / "inference_eos" / "mm_cheating" / "outdir_mm" / "results.h5") 
    #posterior_mm_cheating_unweighed = deepcopy(posterior_mm_cheating)
    #posterior_mm_cheating_unweighed['weights'] = np.full((7000,), 1/7000)


    fig, ax = plt.subplots(5, 1, figsize=(5, 22))
    fig.subplots_adjust(hspace=0.2)

    fig, ax = plot_all(fig, ax, posterior_gw, color="purple")
    fig, ax = plot_all(fig, ax, posterior_gw_unweighed, color="blue")

    fig.savefig(f"{name}.pdf", dpi=250, bbox_inches="tight")



if __name__=="__main__":
    main(sys.argv[1])



