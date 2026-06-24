import sys

import numpy as np
import pandas as pd


file_lists = {
         "narrow_ETT": ["../gw/gw_params_narrow.dat", "../kilonova/kn_params_narrow.dat", "../grb/grb_params_narrow.dat", "./detection_output/narrow_ETT.dat"],
         "narrow_ETL": ["../gw/gw_params_narrow.dat", "../kilonova/kn_params_narrow.dat", "../grb/grb_params_narrow.dat", "./detection_output/narrow_ETL.dat"],
         "narrow_ETT_CE": ["../gw/gw_params_narrow.dat", "../kilonova/kn_params_narrow.dat", "../grb/grb_params_narrow.dat", "./detection_output/narrow_ETT_CE.dat"],
         "narrow_ETL_CE": ["../gw/gw_params_narrow.dat", "../kilonova/kn_params_narrow.dat", "../grb/grb_params_narrow.dat", "./detection_output/narrow_ETL_CE.dat"],
         "wide_ETT": ["../gw/gw_params_wide.dat", "../kilonova/kn_params_wide.dat", "../grb/grb_params_wide.dat", "./detection_output/wide_ETT.dat"],         
         "wide_ETL": ["../gw/gw_params_wide.dat", "../kilonova/kn_params_wide.dat", "../grb/grb_params_wide.dat", "./detection_output/wide_ETL.dat"],
         "wide_ETT_CE": ["../gw/gw_params_wide.dat", "../kilonova/kn_params_wide.dat", "../grb/grb_params_wide.dat", "./detection_output/wide_ETT_CE.dat"],
         "wide_ETL_CE": ["../gw/gw_params_wide.dat", "../kilonova/kn_params_wide.dat", "../grb/grb_params_wide.dat", "./detection_output/wide_ETL_CE.dat"],
        }

def main():

    
    catalog = sys.argv[1]

    files = file_lists[catalog]

    df_gw = pd.read_csv(files[0], sep=" ")
    df_gw.rename(columns={"chi1_z": "chi_1", "chi2_z": "chi_2"}, inplace=True)
    df_kn = pd.read_csv(files[1], sep=" ")
    df_grb = pd.read_csv(files[2], sep=" ")
    df_detection = pd.read_csv(files[3], sep=" ")
    
   
    selected = df_detection["kn_visible"].astype(bool)

    print(f"Total of {np.sum(selected)} selected events.")

    selected_df = pd.concat([df_gw[selected], df_kn[selected], df_grb[selected][["has_grb", "thetaCore", "log10_Ekin_iso", "log10_n0"]]], axis=1)
    selected_df = selected_df.loc[:, ~selected_df.columns.duplicated()]
    selected_df["redshift_measured"] = np.random.normal(loc=selected_df["redshift"], scale=0.01*selected_df["redshift"])
    selected_df["afterglow_detected"] = np.sum(df_detection[["radio_afterglow", "opt_afterglow", "xray_afterglow"]], axis=1) >= 1
    selected_df["snr"] = df_detection.loc[selected, "snr"]
    
    selected_df.to_csv(f"../../eos_inference/{catalog}/events.dat", sep=" ", index=False)

if __name__=="__main__":
    main()