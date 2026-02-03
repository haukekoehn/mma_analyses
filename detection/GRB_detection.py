import sys

import numpy as np
import pandas as pd

np.random.seed(671938)
Mpc_to_cm = 3.8057e24


def log10_fluence(df, key):
    fluence = df[key] + np.log10( (1+df['redshift']) / (4*np.pi*df["luminosity_distance"]**2 * Mpc_to_cm**2))
    return fluence


def grb_detection(df):
    
    log10_fermi_fluence = log10_fluence(df, "log10_Egamma_fermi_gbm")
    log10_swift_fluence = log10_fluence(df, "log10_Egamma_swift_bat")
    log10_gecam_fluence = log10_fluence(df, "log10_Egamma_gecam")
    
    mask_fermi = (log10_fermi_fluence> np.log10(2e-7)) & (np.random.uniform(size=df.shape[0]) < 0.6)
    mask_swift = (log10_swift_fluence > np.log10(2e-8)) & (np.random.uniform(size=df.shape[0]) < 0.1)
    mask_gecam = (log10_gecam_fluence > np.log10(2e-8)) & (np.random.uniform(size=df.shape[0]) < 0.8)

    DeltaOmega = np.full(df.shape[0], 100)
    DeltaOmega[mask_gecam] = 10
    DeltaOmega[mask_swift] = 0.1

    df["GRB_detected"] = (mask_fermi & (mask_swift & mask_gecam))

    return df 


def main():
    
    narrow_grb = pd.read_csv("/home/aya/work/hkoehn/mma_analysis/grb/full_narrow_grb_parameters.dat", sep=" ")
    #narrow_FIM_ET = pd.read_csv("/home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_narrow_ETL1_ETL2_gwfish.dat", sep=" ")
    #narrow_FIM_ET_CE = pd.read_csv("/home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_narrow_ETL1_ETL2_CE_gwfish.dat", sep=" ")

    wide_grb = pd.read_csv("/home/aya/work/hkoehn/mma_analysis/grb/full_wide_grb_parameters.dat", sep=" ")
    #wide_FIM_ET = pd.read_csv("/home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_wide_ETL1_ETL2_gwfish.dat", sep=" ")
    #wide_FIM_ET_CE = pd.read_csv("/home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_wide_ETL1_ETL2_CE_gwfish.dat", sep=" ")

    grb_detection(narrow_grb)
    grb_detection(wide_grb)



if __name__ == "__main__":
    main()