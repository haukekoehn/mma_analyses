import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GWFish.modules.detection import Network
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.fishermatrix import sky_localization_percentile_factor




file = sys.argv[1]

df = pd.read_csv(file, sep=" ")

events = df[["mass_1", "mass_2", "luminosity_distance", "theta_jn", "ra", "dec", "psi", "phase", "geocent_time"]]
network = Network(["ETL1", "ETL2", "CE1"])

detected, snr, errors, sky_localization = compute_network_errors(network,
                                                                 events,
                                                                 waveform_model="IMRPhenomXAS_NRTidalv3")

sky_localization *= sky_localization_percentile_factor()

outfile = f"./outdir_FIM/{file.split('.')[0]}_ETL1_ETL2_CE_gwfish.dat"
out = np.column_stack((snr, sky_localization, errors))
np.savetxt(outfile, out, header = "snr sky_localization mass_1 mass_2 luminosity_distance theta_jn ra dec psi phase geocent_time", comments="")
