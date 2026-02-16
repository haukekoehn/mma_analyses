import numpy as np
import scipy.integrate as integrate
import os

import astropy.units as u
import astropy.constants as constants

MeV_per_fm3_to_gr_per_cm3 = (1.*u.fm**(-3)).to(u.cm**(-3)).value * (1*u.MeV/constants.c**2).to(u.g).value
MeV_per_fm3_to_dynes_per_cm2 = (1.*u.MeV*u.fm**(-3)).to(u.dyne/u.cm**2).value
per_fm3_to_per_cm3 = (1*u.fm**(-3)).to(u.cm**(-3)).value

eos_file = "./RMF3_cold_beta_lamb.d"

_, n, _, p, eps_per_n = np.loadtxt(eos_file, unpack = True)
eps = eps_per_n * n

filt = n<10*0.16
enthalpy = (constants.c**2).to(u.cm**2/u.s**2).value * integrate.cumulative_trapezoid(y= 1/(eps[filt]+p[filt]), x = p[filt], initial=0)
enthalpy[0] = 1
    
n = per_fm3_to_per_cm3*n[filt]
p = MeV_per_fm3_to_dynes_per_cm2*p[filt]
eps = MeV_per_fm3_to_gr_per_cm3*eps[filt]
    
    
np.savetxt(f"./rns_eos.dat", np.array([eps, p, enthalpy, n]).T, 
           header= str(len(eps)), comments='', fmt = "%.5e %.5e %.15e %.15e")