import sys

import joblib
import numpy as np
import pandas as pd
import csv

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde

file = sys.argv[1]

m_val, r_val, l_val = np.loadtxt("../eos/RMF3_MRL.dat", unpack=True)

PC_classifier = joblib.load("./postmerger_classifier/classifierC_model.pkl")
scaler = joblib.load("./postmerger_classifier/scaler_classifierC.pkl")

events = pd.read_csv(file, sep=" ")
input = events[["chi_eff"]].copy()
input["Mratio_fixed"] = events["mass_1_source"] / events["mass_2_source"]
input["LambdaTilde"] = lambda_1_lambda_2_to_lambda_tilde(events["lambda_1"], events["lambda_2"], events["mass_1_source"], events["mass_2_source"])
input["Mtot"] = events["mass_1_source"] + events["mass_2_source"]
input.rename(columns={"chi_eff": "ChiEff"}, inplace=True)


input_scaled = scaler.transform(input[["Mtot", "LambdaTilde", "ChiEff", "Mratio_fixed"]])
pc = PC_classifier.predict(input_scaled)

events["prompt_collapse"] = pc
events.to_csv(file, sep=" ", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


