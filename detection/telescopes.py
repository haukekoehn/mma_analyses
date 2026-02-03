from equinox import is_inexact_array
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

from fiesta.inference.lightcurve_model import FluxModel
from fiesta.filters import Filter


class GroundTelescope:
    
    def __init__(self,
                 name: str,
                 lat: float,
                 long: float,
                 height: float,
                 fov: float,
                 thresholds: dict[str, float],
                 exposure_time: float,
                 dead_time: float) -> None: # filter exchange + slewing
        
        self.name = name
        self.loc = EarthLocation(lat=lat*u.deg, lon=long*u.deg, height=height*u.m)
        self.fov = fov
        self.thr = thresholds
        self.filters = list(thresholds.keys())
        self.exposure_time = exposure_time / (24*3600)
        self.dead_time = dead_time / (24*3600)
    
    def follow_up_campaign(self,
                           trigger_time, 
                           dec, 
                           ra, 
                           DeltaOmega, 
                           times_transient, 
                           mags_transient):
        
        time = trigger_time + 0.2
        ntiles = int(np.ceil(DeltaOmega/self.fov)) + 1
        true_tile = np.random.choice(ntiles)

        total_telescope_time = 0
        total_detections = {filt: 0 for filt in self.filters}

        for epoch, delta_t in enumerate([0., 1/3, 1/2]):
            time, telescope_time, detections = self.target_epoch(time+delta_t, ntiles, true_tile, dec, ra, times_transient, mags_transient)
            
            total_telescope_time += telescope_time
            for filt in self.filters: 
                total_detections[filt] += detections[filt]

            # if target is not observable stop algorithm 
            if telescope_time==0:
                break

            # if it takes too much time, no more epochs
            if total_telescope_time >= 0.5:
                break
            
            # third epoch only if certain criteria are met
            ambigous_detection = sum(total_detections.values()) > 0 and sum(total_detections.values()) <= len(self.filters)/2
            if epoch==0:
                continue
            elif epoch==1 and total_telescope_time<=0.25 and ambigous_detection:
                continue
            else: 
                break
        
        return total_telescope_time, total_detections
   
    def target_epoch(self,
                     start_time,
                     ntiles: int, 
                     true_tile: int,
                     dec: float,
                     ra: float,
                     times_transient: np.ndarray,
                     mags_transient: dict[str, np.ndarray])-> dict[str, bool]:
        
        detections = {filt: 0 for filt in self.filters}
        telescope_time = 0

        for filt in self.filters:
            t_obs = self.get_tiling_times(start_time, dec, ra, ntiles)
            if t_obs.size==0:
                return start_time, 0, detections
            
            mag_obs = np.interp(t_obs[true_tile], times_transient, mags_transient[filt])

            if mag_obs < self.thr[filt]:
                detections[filt] += 1

            telescope_time += ntiles*self.exposure_time + self.dead_time
            time = t_obs[-1] + self.exposure_time + self.dead_time

        return time, telescope_time, detections
    
    def get_tiling_times(self,
                         start_time: float,
                         dec: float,
                         ra: float,
                         ntiles: float) -> np.ndarray:

        tobs = np.empty(ntiles)
        time = start_time
        target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        for j in range(ntiles):

            while not self.check_night_visibility(target, time):
                time += 1/24
                if time>=start_time + 1:
                    return np.array([])

            tobs[j] = time
            time += self.exposure_time
            
        return tobs
    
    def check_night_visibility(self, target, time):

        mjd = Time(time, format="mjd")
        altaz_frame = AltAz(obstime=mjd, location=self.loc)

        sun_coords = get_sun(mjd).transform_to(altaz_frame)
        sky_coords = target.transform_to(altaz_frame)

        return sun_coords.alt.degree <= -12. and sky_coords.alt.degree > 20.
        

wavel_ultrasat = np.linspace(220, 300, 100)
trans_ultrasat = np.exp( -0.5*((wavel_ultrasat - 260)/15)**2 ) * 0.34
nus_ultrasat = 2.99792458e17 / wavel_ultrasat
ultrasat_filter = Filter("ultrasat_custom", nus=nus_ultrasat[::-1], trans=trans_ultrasat[::-1])


class ULTRASAT:

    def __init__(self,
                 fov: float = 204,
                 thresholds = {"ultrasat_custom": 22.5},
                 exposure_time = 900,
                 dead_time = 60) -> None: # filter exchange + slewing
        
        self.name = "ultrasat"
        self.fov = fov
        self.thr = thresholds
        self.filters = list(thresholds.keys())
        self.exposure_time = exposure_time / (24*3600)
        self.dead_time = dead_time / (24*3600)
        self.instant_coverage = 0.51
        self.max_coverage = 0.75 # based on sky access limitations
    
    def follow_up_campaign(self,
                           trigger_time, 
                           dec, 
                           ra, 
                           DeltaOmega, 
                           times_transient, 
                           mags_transient):
        
        ntiles = int(np.ceil(DeltaOmega/self.fov)) + 1
        true_tile = np.random.choice(ntiles)
      
        alpha = np.random.uniform()
        if alpha < self.instant_coverage:
            time, telescope_time, detections = self.target_epoch(trigger_time + 0.25/24, ntiles, true_tile, dec, ra, times_transient, mags_transient)
            return telescope_time, detections
        
        alpha = np.random.uniform()
        if alpha < self.max_coverage:
            time, telescope_time, detections = self.target_epoch(trigger_time + 3/24, ntiles, true_tile, dec, ra, times_transient, mags_transient)
            return telescope_time, detections
        else: 
            return 0, {filt: 0 for filt in self.filters}

    def target_epoch(self,
                     start_time,
                     ntiles: int, 
                     true_tile: int,
                     dec: float,
                     ra: float,
                     times_transient: np.ndarray,
                     mags_transient: dict[str, np.ndarray])-> dict[str, bool]:     



        tobs = np.linspace(start_time, start_time + (ntiles-1)*(self.exposure_time + self.dead_time), ntiles)
        
        detection = {filt: 0 for filt in self.filters}
        for filt in self.filters:
            mag_obs = np.interp(tobs[true_tile], times_transient, mags_transient[filt])
            if mag_obs < self.thr[filt]:
                detection[filt] +=1

        return tobs[-1]+ self.exposure_time + self.dead_time, ntiles*self.exposure_time, detection


#####################
# Actual telescopes #
#####################

ztf = GroundTelescope("ztf", 
                      lat=33 + 31/60 + 26/3600, 
                      long=-116-51/60-35/3600, 
                      height=1712, 
                      fov=47,
                      exposure_time=300,
                      dead_time=100,
                      thresholds=dict(ztfg=22, ztfi=22))

vr = GroundTelescope("vr", 
                      lat=-30 - 14/60 - 41/3600, 
                      long=-70-44/60-58/3600, 
                      height=2672.75, 
                      fov=9.6,
                      exposure_time=600,
                      dead_time=220,
                      thresholds=dict(lsstg=26.5, lssti=25.6))

pstarrs = GroundTelescope("pstarrs", 
                          lat=20 + 42/60 + 26/3600, 
                          long=-156 -15/60 -21/3600, 
                          height=3052, 
                          fov=7,
                          exposure_time=400,
                          dead_time=100,
                          thresholds={"ps1::g": 24, "ps1::i": 24})

ultrasat = ULTRASAT()

        
