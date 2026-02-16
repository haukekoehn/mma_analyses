import numpy as np
import subprocess

import astropy.units as u
import astropy.constants as constants

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###config

epsTOV_samp = np.loadtxt("../plots_pulsars/epsTOV_samp")
MTOV_samp = np.loadtxt("../plots_pulsars/MTOV_samp")
NEOS = 100000

###

MeV_per_fm3_to_gr_per_cm3 = (1.*u.fm**(-3)).to(u.cm**(-3)).value * (1*u.MeV/constants.c**2).to(u.g).value
gr = (np.sqrt(5) + 1) / 2


def rotating_mass(EOSID, ener_dens):
    command = f"./rns -f ./converted_eos/{EOSID}.dat -t omega -e {ener_dens:.5e} -o 0.445477838 -p 2 -d 0"
    result = subprocess.run(command.split(" "), capture_output = True, text = True, timeout=30)

    if result.returncode !=0:
        raise Exception(f"rns had a problem with {EOSID} and {ener_dens:.5e}.")
       
    result = result.stdout.split("\n")
    M = float((result[-2].split(" "))[1])

    return M



def mass_limit(EOSID):

    eTOV = epsTOV_samp[EOSID-1]*MeV_per_fm3_to_gr_per_cm3

    try:
        result = maximum_search_goldenratio(EOSID, eTOV)
    
    except:

        try:
            result = maximum_search_from_above(EOSID, eTOV)

        except:
            result = -1

    return result


def maximum_search_goldenratio(EOSID, eTOV):
    x1 = 0.7* eTOV
    x4  = 1.05* epsTOV_samp[EOSID-1]*MeV_per_fm3_to_gr_per_cm3

    M1 = rotating_mass(EOSID, x1)
    M4 = rotating_mass(EOSID, x4)


    x2 = x4 - (x4-x1) / gr
    x3 = x1 + (x4-x1) / gr
    M2 = rotating_mass(EOSID, x2)
    M3 = rotating_mass(EOSID, x3)

    x = [x1, x2, x3, x4]
    y = [M1, M2, M3, M4]


    counter = 0
    while np.abs((x[3]-x[0])/x[3])>0.02:

        #print(np.array(x)/1e15, y)

        maxpos = y.index(max(y))

        if maxpos==0 or maxpos==1:
            newx = x[2] - (x[2]-x[0]) / gr
            x = [x[0], newx , x[1], x[2]]
            y = [y[0], rotating_mass(EOSID, newx) , y[1], y[2]]

        else:
            newx = x[1] + (x[3]-x[1]) / gr
            x = [x[1], x[2], newx, x[3]]
            y = [y[1], y[2], rotating_mass(EOSID, newx), y[3]]

        counter +=1

        if counter>100:
            raise Exception("Too many iterations.")


    return max(y)

def maximum_search_from_above(EOSID, eTOV):
    
    y = [0]
    x = eTOV


    while True:
       y.append(rotating_mass(EOSID, x))
       x -= 0.01*eTOV

       if y[-1]<y[-2]:
           break

    return max(y)

        




splits = np.array_split(np.arange(1, NEOS+1), size)
work = splits[rank]

save = []
comm.Barrier()
for EOSID in work:

    if MTOV_samp[EOSID-1]<1.5:
        save.append(0)

    else:
        save.append(mass_limit(EOSID))
    print(f"Did {EOSID}.")
    print("\n")


comm.Barrier()
save = comm.gather(save, root = 0)

if rank ==0:
    save = np.array([Mrot for sublist in save for Mrot in sublist])
    np.savetxt("Mrot_samp", save)
