import joblib

import numpy as np
np.random.seed(167834359)

import scipy.stats as stats
import scipy.interpolate as interpolate
import pandas as pd

from astropy.time import Time
import astropy.coordinates as coordinates

import dustmaps.sfd
from nmma.core.conversion import BNSEjectaFitting

# load EOS
m_val, r_val, l_val = np.loadtxt("../../eos/RMF3_MRL.dat", unpack=True)
mtov = m_val.max()
r16 = np.interp(1.6, m_val, r_val)


def add_kn_parameters(df):
    compactness_1 = df["mass_1_source"] / df["radius_1"] * 1.477
    compactness_2 = df["mass_2_source"] / df["radius_2"] * 1.477
    pc = df["prompt_collapse"] <=1

    # dynamical mass 
    df["log10_mej_dyn"] = np.zeros(df.shape[0])
    df.loc[pc, "log10_mej_dyn"] = np.log10(BNSEjectaFitting().dynamic_mass_fitting_prompt_collapse(mass_1=df["mass_1_source"][pc],
                                                                                              mass_2=df["mass_2_source"][pc],
                                                                                              lambda_1=df["lambda_1"][pc],
                                                                                              lambda_2=df["lambda_2"][pc]))

    df.loc[~pc, "log10_mej_dyn"] = np.log10(BNSEjectaFitting().dynamic_mass_fitting_KrFo(mass_1=df["mass_1_source"][~pc],
                                                                                              mass_2=df["mass_2_source"][~pc],
                                                                                              compactness_1=compactness_1[~pc],
                                                                                              compactness_2=compactness_2[~pc]))
    
    # dynamical velocity
    df["v_ej_dyn"] = np.zeros(df.shape[0])
    df.loc[pc, "v_ej_dyn"] = BNSEjectaFitting().dynamic_vel_fitting_prompt_collapse(mass_1=df["mass_1_source"][pc],
                                                                                  mass_2=df["mass_2_source"][pc],
                                                                                  compactness_1=compactness_1[pc],
                                                                                  compactness_2=compactness_2[pc])
    df.loc[~pc, "v_ej_dyn"] = BNSEjectaFitting().dynamic_vel_fitting_Radice2018(mass_1=df["mass_1_source"][~pc],
                                                                                   mass_2=df["mass_2_source"][~pc],
                                                                                   compactness_1=compactness_1[~pc],
                                                                                   compactness_2=compactness_2[~pc])
    

    # disk mass
    df["log10_mdisk"] = np.zeros(df.shape[0])
    df.loc[pc, "log10_mdisk"] = BNSEjectaFitting().log10_disk_mass_fitting_prompt_collapse(mass_1=df["mass_1_source"][pc],
                                                                                              mass_2=df["mass_2_source"][pc],
                                                                                              lambda_1=df["lambda_1"][pc],
                                                                                              lambda_2=df["lambda_2"][pc])
    df.loc[~pc, "log10_mdisk"] = BNSEjectaFitting().log10_disk_mass_fitting(total_mass=df["mass_1_source"][~pc]+df["mass_2_source"][~pc],
                                                                               mass_ratio=df["mass_2_source"][~pc]/df["mass_1_source"][~pc],
                                                                               MTOV=mtov,
                                                                               R16=r16 / 1.477)
    
    # wind ejecta
    df["log10_mej_wind"] = np.zeros(df.shape[0])
    df.loc[pc, "log10_mej_wind"] = np.log10(stats.truncnorm(loc=0.2, scale=0.1, a=-2, b=2).rvs(size=np.sum(pc))) + df["log10_mdisk"][pc]
    df.loc[~pc, "log10_mej_wind"] = np.log10(stats.truncnorm(loc=0.5, scale=0.2, a=-2.5, b=1.5).rvs(size=np.sum(~pc))) + df["log10_mdisk"][~pc]


    # misc 
    df["v_ej_wind"] = np.random.uniform(0.05, 0.15, size=df.shape[0])
    df["Ye_dyn"] = np.random.uniform(0.15, 0.35, size=df.shape[0])
    df["Ye_wind"] = np.random.uniform(0.2, 0.4, size=df.shape[0])
    df["inclination_EM"] = np.arccos(np.abs(np.cos(df["theta_jn"])))
    
    # for the GRB afterglow
    df["chi_BH"] = BNSEjectaFitting().chiBH_fitting(mass_1=df["mass_1_source"], mass_2=df["mass_2_source"], lambda_1=df["lambda_1"], lambda_2=df["lambda_2"])

    # for the trigger time in mjd
    df["trigger_time"] = Time(df["geocent_time"], format="gps").mjd

    # for the extinction correction
    df["Ebv"] = get_Ebv_from_dustmap(df["dec"], df["ra"])

    return df

def get_Ebv_from_dustmap(decs, ras):
    coord = coordinates.SkyCoord(ras, decs, unit='rad')
    Ebv = dustmaps.sfd.SFDQuery()(coord)
    return Ebv


def main():

    df_wide = pd.read_csv("../gw/gw_params_wide.dat", sep=" ")
    df_narrow = pd.read_csv("../gw/gw_params_narrow.dat", sep=" ")

    df_wide = add_kn_parameters(df_wide)
    df_narrow = add_kn_parameters(df_narrow)
    

    params = ["inclination_EM",
              "log10_mej_dyn", 
              "v_ej_dyn", 
              "Ye_dyn", 
              "log10_mdisk", 
              "log10_mej_wind", 
              "v_ej_wind", 
              "Ye_wind", 
              "chi_BH",
              "Ebv",
              "trigger_time",
              "redshift", 
              "luminosity_distance", 
              "prompt_collapse"]
    
    df_narrow[params].to_csv("kn_params_narrow_correct_disk_mass.dat", sep=" ", index=False)
    df_wide[params].to_csv("kn_params_wide_correct_disk_mass.dat", sep=" ", index=False)


if __name__ =="__main__":
    main()