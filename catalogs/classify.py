import sys

import joblib
import numpy as np
import pandas as pd
import csv

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from nmma.core.conversion import BNSEjectaFitting

file = sys.argv[1]

m_val, r_val, l_val, mb_val = np.loadtxt("../eos/RMF3_MRLMb.dat", unpack=True)
mtov = m_val.max()
r16 = np.interp(1.6, m_val, r_val)
Mthr = (-3.606 * 1.477* mtov/ r16 + 2.38) * mtov
Mkep_bar = 2.918 

events = pd.read_csv(file, sep=" ")
pc = np.ones(events.shape[0])

prompt_collapse = (events["mass_1_source"] + events["mass_2_source"]) > Mthr
m1_bar = np.interp(events["mass_1_source"], m_val, mb_val)
m2_bar = np.interp(events["mass_2_source"], m_val, mb_val)
stable = (m1_bar + m2_bar) < Mkep_bar

pc[~prompt_collapse] = 2
pc[stable] = 4


events["prompt_collapse"] = pc
events.to_csv(file, sep=" ", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


