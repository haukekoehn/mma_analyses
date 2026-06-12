import numpy as np
import sys

src = sys.argv[1]


posterior = dict(np.load(f"source_{src}/posterior.npz", allow_pickle=True))
posterior["cos_theta_jn"] = posterior["cos_inclination_EM"] * np.random.choice([-1, 1], size=posterior["cos_inclination_EM"].shape[0])
np.savez(f"source_{src}/posterior.npz", **posterior)
