from pathlib import Path
import sys
import shutil

import numpy as np
import pandas as pd


def rename_and_remove_old_events(dir, events, events_old):
    
    new_indices = np.zeros(events_old.shape[0])

    for j in range(events_old.shape[0]):
        row = events_old.loc[j]
        new_index = np.where( (row["redshift"]==events["redshift"]) & (row["mass_1_source"]==events["mass_1_source"]) )[0]
        assert len(new_index)<=1

        if new_index.size==0:
            new_indices[j] = -1
        else:
            new_indices[j] = new_index[0]

    for old_ind, new_ind in enumerate(new_indices):

        src_gw = dir / "gw_posteriors" / f"source_{old_ind}"
        src_em = dir / "em_posteriors" / f"source_{old_ind}"

        if new_ind==-1:
            dst_gw = dir / "trash" / "gw_posteriors"
            dst_em = dir / "trash" / "em_posteriors"
        else:
            dst_gw = dir / "gw_posteriors" / f"source_{int(new_ind)}_new_ind"
            dst_em = dir / "em_posteriors" / f"source_{int(new_ind)}_new_ind"

        shutil.move(src=src_gw, dst=dst_gw)
        shutil.move(src=src_em, dst=dst_em)


def get_list_of_missing_events(dir, events, events_old):

    counter_missing_events = 0
    missing_indices = []

    for new_ind in range(events.shape[0]):
        row = events.loc[new_ind]
        old_ind = np.where((row["redshift"]==events_old["redshift"]) &(row["mass_1_source"]==events_old["mass_1_source"]) )[0]

        if old_ind.size==0:
            counter_missing_events +=1
            missing_indices.append(new_ind)
    
    np.savetxt(dir / "missing_events.dat", missing_indices, fmt="%d")

def main(dir: str):

    dir = Path(dir)

    events = pd.read_csv(dir / "events.dat", sep=" ")
    events_old = pd.read_csv(dir / "events_old.dat", sep=" ")

    #rename_and_remove_old_events(dir, events, events_old)

    get_list_of_missing_events(dir, events, events_old)


if __name__=="__main__":
    main(sys.argv[1])