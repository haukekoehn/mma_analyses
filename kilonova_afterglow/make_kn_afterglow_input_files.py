import sys
import tqdm
import numpy as np
import h5py

import scipy.optimize as optimize
import pandas as pd


def find_vmax_wind(v_ej_wind, v_min = 0.02):

    res = optimize.brentq(lambda x: (x-v_min) / np.log(x/v_min) - v_ej_wind, 1e-3, 0.9)

    return res

def find_vmax_dyn(v_ej_dyn, vmin=0.1, vmid=0.4):

    if v_ej_dyn <= np.log(vmid/vmin) / (1/vmin-1/vmid):
        def eq(x):
            out = np.log(x/vmin)
            out /= vmin**(-1) - x**(-1)
            return out - v_ej_dyn
        res = optimize.brentq(eq, 0.11, 0.4)
    
    else:
        def eq(x):
            out = np.log(vmid/vmin) + 0.4**4 * (vmid**(-4)-x**(-4)) / 4
            out /= (vmin**(-1)- vmid**(-1) ) + 0.4**4 * ( vmid**(-5) - x**(-5) ) / 5
            return out - v_ej_dyn
        
        try:
            res = optimize.brentq(eq, 0.4, 0.9)
        except ValueError:
            res = 0.9
        
    return res

def find_rho0_wind(mej, vmax):
    rho0 = mej
    rho0 /= (4*np.pi* np.log(vmax/0.02))
    return rho0

def find_rho0_dyn(mej, vmax):
    rho0 = mej

    if vmax <= 0.4:
        rho0 /= (8*np.pi)/3 *(1/0.1 - 1/vmax)
    else:
        rho0 /= (8*np.pi)/3 * ( (1/0.1 - 1/0.4) + 0.4**4 * (0.4**(-5) - vmax**(-5)) / 5 )

    return rho0

class POSSISDensity:
    def __init__(self, rho0_dyn, vmax_dyn, rho0_wind, vmax_wind):

        self.rho0_dyn = rho0_dyn
        self.rho0_wind = rho0_wind
        self.vmax_dyn = vmax_dyn
        self.vmax_wind = vmax_wind

    def mass_segment(self, v0, v1, theta0, theta1):

        delta_mej_wind = self.wind_mass_segment(v0, v1, theta0, theta1)
        delta_mej_dyn = self.dyn_mass_segment(v0, v1, theta0, theta1)

        return delta_mej_wind + delta_mej_dyn
    
    def wind_mass_segment(self, v0, v1, theta0, theta1):

        v0 = min( max(0.02, v0), self.vmax_wind)
        v1 = min( max(0.02, v1), self.vmax_wind)
        return 2*np.pi*self.rho0_wind*np.log(v1/v0) * (np.cos(theta0) - np.cos(theta1))
    
    def dyn_mass_segment(self, v0, v1, theta0, theta1):
        v0 = min( max(0.1, v0), self.vmax_dyn)
        v1 = min( max(0.1, v1), self.vmax_dyn)

        if v1 <= 0.4:
            radial_factor = self.rho0_dyn * (1/v0 - 1/v1)
        elif v0 < 0.4 and v1 > 0.4:
            radial_factor = self.rho0_dyn * ( (1/v0 - 1/0.4) + 0.4**4 * (0.4**(-5) - v1**(-5) ) / 5 )
        elif v0 >= 0.4:
            radial_factor = self.rho0_dyn * 0.4**4 * (v0**(-5) - v1**(-5)) / 5

        angular_factor = 2*np.pi * (np.cos(theta1)**3 / 3 - np.cos(theta1) - np.cos(theta0)**3 / 3 + np.cos(theta0))

        return radial_factor * angular_factor



def make_histogram(density: POSSISDensity, theta_arr, vel_arr):
    
    ntheta = theta_arr.shape[0]
    nvel = vel_arr.shape[0]
    mej_2d = np.zeros((ntheta-1, nvel-1))

    for j in range(ntheta-1):
        for k in range(nvel-1):
            mej_2d[j, k] = density.mass_segment(vel_arr[k], vel_arr[k+1], theta_arr[j], theta_arr[j+1])
    
    return mej_2d



def get_mass_histogram(m_ej_dyn, v_ej_dyn, m_ej_wind, v_ej_wind):

    vmax_dyn = find_vmax_dyn(v_ej_dyn)
    vmax_wind = find_vmax_wind(v_ej_wind)
    rho0_dyn = find_rho0_dyn(m_ej_dyn, vmax_dyn)
    rho0_wind = find_rho0_wind(m_ej_wind, vmax_wind)

    density = POSSISDensity(rho0_dyn, vmax_dyn, rho0_wind, vmax_wind)


    theta_arr = np.linspace(0, np.pi, 101)
    vel_arr = np.linspace(0, max(vmax_dyn, vmax_wind), 31)

    mej_2d = make_histogram(density, theta_arr, vel_arr)
    return mej_2d, vel_arr, theta_arr



def write_to_file(filename, 
                  vel_inf,
                  theta,
                  m_ej):

    with h5py.File(filename, "w") as out:
        out.create_dataset("vel_inf", data=vel_inf, dtype=float)
        out.create_dataset("theta", data=theta, dtype=float)
        out.create_dataset("mass", data=m_ej, dtype=float)


def main():
    
    file = sys.argv[1]
    df = pd.read_csv(file, sep=" ")

    if "narrow" in file:
        outdir = "narrow_kn_afterglow_input"
    elif "wide" in file:
        outdir = "wide_kn_afterglow_input"

    
    for j in tqdm.tqdm(range(df.shape[0])):
        mej_2d, vel_arr, theta_arr = get_mass_histogram(10**df["log10_mej_dyn"][j], df["v_ej_dyn"][j], 10**df["log10_mej_wind"][j], df["v_ej_wind"][j])
        filename = f"{outdir}/possis_geometry_{j}.h5"
        write_to_file(filename, vel_arr, theta_arr, mej_2d)

        

if __name__=="__main__":
    main()