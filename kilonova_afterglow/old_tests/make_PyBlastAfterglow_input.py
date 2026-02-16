import numpy as np
import os
import sys
import shutil
from argparse import ArgumentParser
import shutil
import numpy.ma as ma
import h5py
import units
import load_output_r as outr

def integral_over_sphere(integrand, surface_element, dtheta, dphi):
    res = integrand*surface_element
    res = np.trapz(res, dx=dphi)
    res = np.trapz(res, dx=dtheta)
    return res

def integral_over_time(integrand, dt):
    return np.trapz(integrand, dx=dt, axis=0)


def integral_over_phi(integrand, surface_element, dtheta, dphi):
    res = integrand*surface_element
    res = np.trapz(res, dx=dphi, axis=-1)
    return res*dtheta

directory = "/home/abzu/work/hengieg/RMF3_MHD/"
sims = [ 'RMF3_MHD_p00_p00_256_premrg','RMF3_MHD_p01_p01_256_premrg']

for sim in sims:

    t = outr.read_time(directory+sim+"/output_r/hydroa_integrand_dDudt.1.l2")
    dt = t[2] - t[1]
    t, indexes_t = np.unique(t, return_index=True)

    r, dtheta, dphi, ntheta, nphi, theta, phi = outr.read_ang_grid(directory+sim+"/output_r/hydroa_integrand_dDudt.1.l2")
    phi, theta = np.meshgrid(phi, theta)
    surface_element = r**2*np.sin(theta)

    ## load data 
    fD  = outr.load_spherical_data(directory+sim+"/output_r/hydroa_integrand_dDudt.1.l2", ntheta, nphi)
    fD = fD[list(indexes_t),:,:]

    ut  = outr.load_spherical_data(directory+sim+"/output_r/hydroa_ut.1.l2", ntheta, nphi)
    ut = ut[list(indexes_t),:,:]

    fD[np.where(fD<0)]=0
    fD[np.where((ut)>-1)] = 0

    gamma = np.copy(-ut)
    gamma[np.where(gamma<1)] = 1
    beta = np.sqrt(1-1/gamma**2)
    gamma_beta = gamma*beta
    v = np.sqrt(1-1/gamma**2)

    # define sampeling arrays
    gamma_beta_samp = np.linspace(0.01, np.max(gamma_beta), 31)
    v_inf = gamma_beta_samp/np.sqrt(1+gamma_beta_samp**2)
    Theta = np.zeros(2*ntheta+1)
    Theta[1:ntheta+1] = theta[:,0]
    Theta[ntheta+1:]  = theta[:,0]+0.5*np.pi
    Mej   = np.zeros((len(Theta)-1,len(v_inf)-1))

    print("computing histogram of v:")

    for j in range(len(v_inf)-1):
        v_min = v_inf[j]
        v_max = v_inf[j+1]

        fD_filt = np.copy(fD)
        fD_filt[np.where((v<v_min) | (v>=v_max))] = 0

        dDdt = integral_over_phi(fD_filt, surface_element, dtheta, dphi)
        
        Mej[:ntheta,j]  = integral_over_time(dDdt,  dt)
        Mej[ntheta:,j]  = np.flip(Mej[:ntheta,j])

    print("     ejecta mass ", np.sum(Mej))
    
    ## wrte in h5 ##
    print("Writing h5 file:")

    out = h5py.File(sim[:17]+"corr_vinf_theta.h5", "w")

    out.create_dataset("vel_inf", (len(v_inf),), dtype=float)
    out['vel_inf'][:] = v_inf

    out.create_dataset("theta", (len(Theta),), dtype=float)
    out['theta'][:] = Theta

    out.create_dataset("mass", (np.shape(Mej)), dtype=float)
    out['mass'][:] = Mej
