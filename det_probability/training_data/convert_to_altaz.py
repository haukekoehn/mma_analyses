import tqdm

import numpy as np
import pandas as pd


import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.utils import iers

ETL_location = EarthLocation(lat=40.5*u.deg, lon=9.35*u.deg, height=100*u.m)
ETT_location = EarthLocation(lat=50.795483*u.deg, lon=5.848956*u.deg, height=100*u.m)

def convert_df(df):
    df["alt"] = np.zeros(df.shape[0])
    df["dec"] = np.zeros(df.shape[0])

    mjd = Time(df["geocent_time"].to_numpy(), format="gps")
    target = SkyCoord(ra=df["ra"].to_numpy()*u.rad, dec=df["dec"].to_numpy()*u.rad, frame='icrs')
    altaz_frame = AltAz(obstime=mjd, location=ETL_location)
    sky_coords = target.transform_to(altaz_frame)

    df["alt"] = sky_coords.alt.rad
    df["az"] = sky_coords.az.rad

    return df

df_ETT = pd.read_csv("./train_ETL_3.dat", sep=" ")
df_ETT = convert_df(df_ETT)
df_ETT.to_csv("./train_ETL_3.dat", sep=" ")


