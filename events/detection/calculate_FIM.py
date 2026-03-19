import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GWFish.modules.detection import Network
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.fishermatrix import sky_localization_percentile_factor


def calculate_fim(file, network):
    df = pd.read_csv(file, sep=" ")
    events = df[["mass_1", "mass_2", "luminosity_distance", "theta_jn", "ra", "dec", "psi", "phase", "geocent_time"]]

    detected, snr, errors, sky_localization = compute_network_errors(network,
                                                                     events,
                                                                     waveform_model="IMRPhenomXAS_NRTidalv3")

    sky_localization *= sky_localization_percentile_factor()

    out = np.column_stack((snr, sky_localization, errors))

    return out


def main(file):

    out = calculate_fim(file, network = Network(["ETT", "CE"]))
    if "narrow" in file:
         outfile = f"./outdir_FIM/gwfish_narrow_ETT_CE.dat"
    else:
        outfile = f"./outdir_FIM/gwfish_wide_ETT_CE.dat"
    np.savetxt(outfile, out, header = "snr sky_localization mass_1 mass_2 luminosity_distance theta_jn ra dec psi phase geocent_time", comments="")

    out = calculate_fim(file, network = Network(["ETL1", "ETL2", "CE"]))
    if "narrow" in file:
         outfile = f"./outdir_FIM/gwfish_narrow_ETL_CE.dat"
    else:
        outfile = f"./outdir_FIM/gwfish_wide_ETL_CE.dat"
    np.savetxt(outfile, out, header = "snr sky_localization mass_1 mass_2 luminosity_distance theta_jn ra dec psi phase geocent_time", comments="")

if __name__ == "__main__":
    file = sys.argv[1]
    main(file)
