import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(18950)

from GWFish.modules.detection import Network
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.fishermatrix import sky_localization_percentile_factor
from GWFish.modules.utilities import get_snr


events = pd.read_csv("train_ETL_test.dat", sep=" ")
network = Network(["ETL1", "ETL2"])
cols = ["mass_1", "mass_2", "luminosity_distance", "theta_jn",
        "ra", "dec", "psi", "phase", "geocent_time"]

new_snr = get_snr(events.loc[:1000, cols], network, waveform_model="IMRPhenomXAS_NRTidalv3")["network"].to_numpy()

plt.scatter(events.loc[:1000, "snr"], new_snr)
plt.plot(np.linspace(100, 1000, 2), np.linspace(100, 1000, 2), color="black", linestyle="dashed")
plt.show()

breakpoint()
