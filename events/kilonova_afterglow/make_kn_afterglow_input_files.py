import sys
import tqdm
import numpy as np
import h5py
from pathlib import Path

import scipy.optimize as optimize
import pandas as pd


def powerlaw_integral(alpha, x0, x1):

    if alpha==-1.: 
        return np.log(x1/x0)
    else:
        return (x1**(alpha+1) - x0**(alpha+1)) / (alpha+1)
    
def double_powerlaw_integral(alpha1, alpha2, xmid, x0, x1):
    if x1 <= xmid:
        return powerlaw_integral(alpha1, x0, x1)
    elif x0 <= xmid and xmid < x1:
        return powerlaw_integral(alpha1, x0, xmid) + xmid**(alpha1 - alpha2) * powerlaw_integral(alpha2, xmid, x1)
    elif xmid < x0:
        return powerlaw_integral(alpha2, x0, x1)

def velocity_grid_wind(v_ej_wind, v_min = 0.02, alpha=3):

    def eq(x):
        a = powerlaw_integral(3-alpha, v_min, x)
        b = powerlaw_integral(2-alpha, v_min, x)
        return a/b - v_ej_wind

    res = optimize.brentq(eq, 0.0201, 0.9)

    return v_min, res, alpha

def velocity_grid_dyn(v_ej_dyn, v_min=0.1, v_mid=0.4, alpha1=4, alpha2=8):

    max_vavg = double_powerlaw_integral(3-alpha1, 3-alpha2, v_mid, v_min, 0.9) / double_powerlaw_integral(2-alpha1, 2-alpha2, v_mid, v_min, 0.9)

    if v_ej_dyn < max_vavg:

        def eq(x):
            a = double_powerlaw_integral(3-alpha1, 3-alpha2, v_mid, v_min, x)
            b = double_powerlaw_integral(2-alpha1, 2-alpha2, v_mid, v_min, x)
            return a / b - v_ej_dyn
        
        res = optimize.brentq(eq, 0.101, 0.9)
        v_max = res
    
    else:

        v_max = 0.9

        def eq(x):
            alpha1_tmp = 6 - 2* x/0.1
            a = double_powerlaw_integral(3-alpha1_tmp, 3-alpha2, v_mid, x, v_max)
            b = double_powerlaw_integral(2-alpha1_tmp, 2-alpha2, v_mid, x, v_max)
            return a / b - v_ej_dyn
        
        res = optimize.brentq(eq, 0.1, 0.2)
        v_min = res
        alpha1 = 6 - 2*v_min/0.1

    return v_min, v_mid, v_max, alpha1, alpha2



class POSSISDensity:
    def __init__(self, m_ej_dyn, v_ej_dyn, m_ej_wind, v_ej_wind):

        self.m_ej_dyn = m_ej_dyn
        self.v_ej_dyn = v_ej_dyn
        self.m_ej_wind = m_ej_wind
        self.v_ej_wind = v_ej_wind

        self.v_min_dyn, self.v_mid_dyn, self.v_max_dyn, self.alpha1, self.alpha2 = velocity_grid_dyn(v_ej_dyn)
        self.v_min_wind, self.v_max_wind, self.alpha_wind = velocity_grid_wind(v_ej_wind)

        self.rho0_dyn = self.find_rho0_dyn(self.m_ej_dyn)
        self.rho0_wind = self.find_rho0_wind(self.m_ej_wind)
    
    def find_rho0_dyn(self, m_ej_dyn):
        radial_integral = double_powerlaw_integral(2-self.alpha1, 2-self.alpha2, self.v_mid_dyn, self.v_min_dyn, self.v_max_dyn)
        angular_integral = 8*np.pi/3
        return m_ej_dyn / (angular_integral * radial_integral)
    
    def find_rho0_wind(self, m_ej_wind):
        radial_integral = powerlaw_integral(2-self.alpha_wind, self.v_min_wind, self.v_max_wind)
        angular_integral = 4*np.pi
        return m_ej_wind / (angular_integral * radial_integral)

    def mass_segment(self, v0, v1, theta0, theta1):

        delta_mej_wind = self.wind_mass_segment(v0, v1, theta0, theta1)
        delta_mej_dyn = self.dyn_mass_segment(v0, v1, theta0, theta1)

        return delta_mej_dyn + delta_mej_wind
    
    def wind_mass_segment(self, v0, v1, theta0, theta1):

        v0 = min( max(self.v_min_wind, v0), self.v_max_wind)
        v1 = min( max(self.v_min_wind, v1), self.v_max_wind)
        
        velocity_integral = powerlaw_integral(2-self.alpha_wind, v0, v1)

        return 2*np.pi*self.rho0_wind * velocity_integral * (np.cos(theta0) - np.cos(theta1))
    
    def dyn_mass_segment(self, v0, v1, theta0, theta1):
        v0 = min( max(self.v_min_dyn, v0), self.v_max_dyn)
        v1 = min( max(self.v_min_dyn, v1), self.v_max_dyn)

        if v1 <= self.v_mid_dyn:
            radial_factor = powerlaw_integral(2-self.alpha1, v0, v1)
        elif v0 < self.v_mid_dyn and v1 > self.v_mid_dyn:
            radial_factor = double_powerlaw_integral(2-self.alpha1, 2-self.alpha2, self.v_mid_dyn, v0, v1)
        elif v0 >= self.v_mid_dyn:
            radial_factor =  self.v_mid_dyn**(self.alpha2-self.alpha1) * powerlaw_integral(2-self.alpha2, v0, v1)

        angular_factor = 2*np.pi * ( np.cos(theta1)**3 / 3 - np.cos(theta1) - np.cos(theta0)**3 / 3 + np.cos(theta0) )

        return self.rho0_dyn * radial_factor * angular_factor
    

def make_histogram(density: POSSISDensity, theta_arr, vel_arr):
    
    ntheta = theta_arr.shape[0]
    nvel = vel_arr.shape[0]
    mej_2d = np.zeros((ntheta-1, nvel-1))
    flags = np.zeros((ntheta-1, nvel-1))

    for j in range(ntheta-1):
        for k in range(nvel-1):
            theta = 0.5*(theta_arr[j] + theta_arr[j+1])
            v = 0.5*(vel_arr[k] + vel_arr[k+1])
            
            if (density.v_min_wind < v < density.v_max_wind) and (v < density.v_min_dyn or theta < np.pi/4 or theta > 3/4*np.pi):
                mej_2d[j, k] = density.wind_mass_segment(vel_arr[k], vel_arr[k+1], theta_arr[j], theta_arr[j+1])
            elif density.v_min_dyn < v < density.v_max_dyn:
                mej_2d[j, k] = density.dyn_mass_segment(vel_arr[k], vel_arr[k+1], theta_arr[j], theta_arr[j+1])
                flags[j,k] = 1
            else:
                mej_2d[j, k] = 0.
                flags[j,k] = 3
    
    mej_2d[flags==0] *= density.m_ej_wind / np.sum(mej_2d[flags==0])
    mej_2d[flags==1] *= density.m_ej_dyn / np.sum(mej_2d[flags==1])
    
    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    #ax.pcolormesh(0.5*(theta_arr[1:]+theta_arr[:-1]), 0.5*(vel_arr[1:] + vel_arr[:-1]), np.log10(mej_2d.T))
    #ax.set_xlim((0, np.pi))
    #breakpoint()
        
    return mej_2d



def get_mass_histogram(m_ej_dyn, v_ej_dyn, m_ej_wind, v_ej_wind):

    density = POSSISDensity(m_ej_dyn, v_ej_dyn, m_ej_wind, v_ej_wind)


    theta_arr = np.linspace(0, np.pi, 101)
    vel_arr = np.linspace(min(density.v_min_dyn, density.v_min_wind), max(density.v_max_dyn, density.v_max_wind), 31)

    mej_2d = make_histogram(density, theta_arr, vel_arr)
    return mej_2d, vel_arr, theta_arr



def write_to_file(filename, 
                  vel_inf,
                  theta,
                  m_ej,
                  inclination_EM,
                  log10_nism):

    with h5py.File(filename, "w") as out:
        out.create_dataset("vel_inf", data=vel_inf, dtype=float)
        out.create_dataset("theta", data=theta, dtype=float)
        out.create_dataset("mass", data=m_ej, dtype=float)
        out.create_dataset("inclination_EM", data=inclination_EM, dtype=float)
        out.create_dataset("log10_nism", data=log10_nism)


def main():
    
    file = Path(sys.argv[1])
    df = pd.read_csv(file, sep=" ")
    df_grb = pd.read_csv(file.parent.parent / "grb" / file.name.replace("kn", "grb"), sep=" ")

    if "narrow" in str(file):
        outdir = "narrow_kn_afterglow_input"
    elif "wide" in str(file):
        outdir = "wide_kn_afterglow_input"

    for j in tqdm.tqdm(range(df.shape[0])):
        mej_2d, vel_arr, theta_arr = get_mass_histogram(10**df["log10_mej_dyn"][j], df["v_ej_dyn"][j], 10**df["log10_mej_wind"][j], df["v_ej_wind"][j])
        filename = f"{outdir}/possis_geometry_{j}.h5"
        write_to_file(filename, vel_arr, theta_arr, mej_2d, df["inclination_EM"][j], df_grb["log10_n0"][j])

        

if __name__=="__main__":
    main()