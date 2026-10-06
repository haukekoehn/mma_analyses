import numpy as np
import pandas as pd

from nmma.core.conversion import BNSEjectaFitting

# load EOS
m_val, r_val, l_val = np.loadtxt("../../eos/MM_CSE_stiff/MM_CSE_MRL.dat", unpack=True)
mtov = m_val.max()
r16 = np.interp(1.6, m_val, r_val)


events = pd.read_csv("events.dat", sep=" ")


# GW parameters
events["radius_1"] = np.interp(events["mass_1_source"], m_val, r_val)
events["radius_2"] = np.interp(events["mass_2_source"], m_val, r_val)

events["lambda_1"] = np.interp(events["mass_1_source"], m_val, l_val)
events["lambda_2"] = np.interp(events["mass_2_source"], m_val, l_val)

# chi_BH
events["chi_BH"] = BNSEjectaFitting().chiBH_fitting(mass_1=events["mass_1_source"], mass_2=events["mass_2_source"], lambda_1=events["lambda_1"], lambda_2=events["lambda_2"])

# EM parameters
compactness_1 = events["mass_1_source"] / events["radius_1"] * 1.477
compactness_2 = events["mass_2_source"] / events["radius_2"] * 1.477
pc = events["prompt_collapse"] <=1

events.loc[pc, "log10_mej_dyn"] = np.log10(
    BNSEjectaFitting().dynamic_mass_fitting_prompt_collapse(
        mass_1=events["mass_1_source"][pc],
        mass_2=events["mass_2_source"][pc],
        lambda_1=events["lambda_1"][pc],
        lambda_2=events["lambda_2"][pc]
    )
)

events.loc[~pc, "log10_mej_dyn"] = np.log10(
    BNSEjectaFitting().dynamic_mass_fitting_KrFo(
        mass_1=events["mass_1_source"][~pc],
        mass_2=events["mass_2_source"][~pc],
        compactness_1=compactness_1[~pc],
        compactness_2=compactness_2[~pc]
    )
)

# dynamical velocity
events.loc[pc, "v_ej_dyn"] = BNSEjectaFitting().dynamic_vel_fitting_prompt_collapse(
    mass_1=events["mass_1_source"][pc],
    mass_2=events["mass_2_source"][pc],
    compactness_1=compactness_1[pc],
    compactness_2=compactness_2[pc]
)

events.loc[~pc, "v_ej_dyn"] = BNSEjectaFitting().dynamic_vel_fitting_Radice2018(
    mass_1=events["mass_1_source"][~pc],
    mass_2=events["mass_2_source"][~pc],
    compactness_1=compactness_1[~pc],
    compactness_2=compactness_2[~pc]
)

# disk mass
log10_zeta = np.copy(events["log10_mej_wind"] - events["log10_mdisk"])
events.loc[pc, "log10_mdisk"] = BNSEjectaFitting().log10_disk_mass_fitting_prompt_collapse(
    mass_1=events["mass_1_source"][pc],
    mass_2=events["mass_2_source"][pc],
    lambda_1=events["lambda_1"][pc],
    lambda_2=events["lambda_2"][pc]
)

events.loc[~pc, "log10_mdisk"] = BNSEjectaFitting().log10_disk_mass_fitting(
    total_mass=events["mass_1_source"][~pc]+events["mass_2_source"][~pc],
    mass_ratio=events["mass_2_source"][~pc]/events["mass_1_source"][~pc],
    MTOV=mtov,
    R16=r16 / 1.477
)


# wind mass
events["log10_mej_wind"] = log10_zeta + events["log10_mdisk"]


# save
events.to_csv("events.dat", sep=" ")