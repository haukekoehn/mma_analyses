import numpy as np

dyn_per_cm2_in_MeV_per_fm3 = 6.241509e-34
g_per_cm3_in_MeV_per_fm3 = 5.60959e-13

n, eps, p = np.loadtxt("H4_not_useful_format.dat", unpack=True)

p *= dyn_per_cm2_in_MeV_per_fm3
eps *= g_per_cm3_in_MeV_per_fm3

indices = np.argsort(eps)

np.savetxt("H4_useful_format.dat", np.array([eps[indices], p[indices]]).T, header="energy \t density pressure [all in nuclear units]")

