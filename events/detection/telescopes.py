import numpy as np
import pandas as pd
from scipy import integrate, interpolate

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import warnings
from astropy.utils.exceptions import ErfaWarning
warnings.simplefilter("ignore", ErfaWarning)


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
    
    def kilonova_campaign(self,
                    start: bool,
                    trigger_time, 
                    dec, 
                    ra, 
                    DeltaOmega, 
                    times_transient, 
                    mags_transient):
        
        if not start:
            return 0, {filt: 0 for filt in self.filters}
        
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
            if epoch==0:
                continue
            elif epoch==1:
                ambigous_detection = 0 < sum(total_detections.values()) <= len(self.filters)/2
                if total_telescope_time<=0.25 and ambigous_detection:
                    continue
                else: 
                    break
        
        return total_telescope_time, total_detections

    def afterglow_campaign(self,               
                           start: bool,
                           trigger_time, 
                           dec, 
                           ra, 
                           DeltaOmega, 
                           times_transient, 
                           mags_transient):
        
        if not start:
            return 0, {filt: 0 for filt in self.filters}
        
        t_epochs = np.geomspace(10, 5*365, 10) + trigger_time
        ntiles = int(np.ceil(DeltaOmega/self.fov)) + 1
        true_tile = np.random.choice(ntiles)

        total_telescope_time = 0
        total_detections = {filt: 0 for filt in self.filters}

        for t_epoch in t_epochs:
            if t_epoch - trigger_time < 30:
                _, telescope_time, detections = self.target_epoch(t_epoch, 
                                                                  ntiles, 
                                                                  true_tile, 
                                                                  dec, 
                                                                  ra, 
                                                                  times_transient,
                                                                  mags_transient)
            else:
                epoch_tolerance = (t_epoch - trigger_time)/2 >  182
                _, telescope_time, detections = self.late_epoch(t_epoch,
                                                                epoch_tolerance,
                                                                ntiles,
                                                                true_tile,
                                                                dec,
                                                                ra,
                                                                times_transient,
                                                                mags_transient)


            total_telescope_time += telescope_time
            for filt in self.filters: 
                total_detections[filt] += detections[filt]
            if sum(total_detections.values())  > 0:
                return total_telescope_time, total_detections

        return 0, {filt: 0 for filt in self.filters}
    
    def late_epoch(self, 
                   start_time,
                   epoch_tolerance,
                   ntiles,
                   true_tile,
                   dec,
                   ra,
                   times_transient,
                   mags_transient):
        
        detections = {filt: 0 for filt in self.filters}
        telescope_time = 0

        for filt in self.filters:
            possible_times = np.arange(start_time, start_time+1, 1/24)
            visible = self.check_visibility(possible_times, ra, dec)

            if not np.any(visible) and epoch_tolerance:
                visible = self.check_visibility(possible_times+ 182, ra, dec)
            if not np.any(visible):
                break
            else: 
                t_obs = possible_times[visible][0]
            
            mag_obs = np.interp(t_obs, times_transient, mags_transient[filt])
            if mag_obs < self.thr[filt]:
                detections[filt] += 1
            telescope_time += ntiles * self.exposure_time + self.dead_time

        return np.nan, telescope_time, detections
   
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

        possible_times = np.arange(start_time, start_time+1, 1/24)
        visible = self.check_visibility(possible_times, ra, dec)

        if not np.any(visible):
            return np.array([])
        else:
            return possible_times[visible][0] + np.linspace(0, (ntiles-1)*self.exposure_time, ntiles)
    
    def check_visibility(self, time, ra, dec):
        mjd = Time(time, format="mjd")
        altaz_frame = AltAz(obstime=mjd, location=self.loc)
        
        target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        sun_coords = get_sun(mjd).transform_to(altaz_frame)
        sky_coords = target.transform_to(altaz_frame)

        return (sun_coords.alt.degree <= -18.) & (sky_coords.alt.degree > 20.)
        

wavel_ultrasat = np.linspace(200, 350, 100)
transmission_matrix = np.loadtxt("./ultrasat_filter/ULTRASAT_TR.dat", delimiter=",")
wavelengths = np.loadtxt("./ultrasat_filter/wavelength.dat")
trans_ultrasat = np.interp(wavel_ultrasat, wavelengths/10, transmission_matrix[:, 10]) # 4.6 deg offset
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
    
    def kilonova_campaign(self,
                           start: bool,
                           trigger_time, 
                           dec, 
                           ra, 
                           DeltaOmega, 
                           times_transient, 
                           mags_transient):
        
        if not start:
            return 0, {filt: 0 for filt in self.filters}
        
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
    

class RadioTelescope(GroundTelescope):

    def __init__(self,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def check_visibility(self, time, ra, dec):
        mjd = Time(time, format="mjd")
        altaz_frame = AltAz(obstime=mjd, location=self.loc)
        
        target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        sky_coords = target.transform_to(altaz_frame)

        return sky_coords.alt.degree > 20.
        
class EINSTEINPROBE():

    def __init__(self, name ="einsteinprobe"):
        self.name = name    
    
    def afterglow_campaign(self,
                           start,
                           trigger_time,
                           dec,
                           ra,
                           DeltaOmega,
                           times_transient,
                           nus_transient,
                           log10_flux):
        
        if not start: 
            return 0, {"X-ray-0.5-4keV": 0}
        
        t_epochs =  np.geomspace(10, 5*365, 10) + trigger_time
        visible = self.check_visibility(t_epochs, ra, dec)
        t_epochs[~visible] += 365/2
        t_epochs[:2] = np.minimum(trigger_time + 365/2 + 10, t_epochs[:2])

        xray_filt = Filter("X-ray-0.5-4keV")
        log10flux_xray = interpolate.interp1d(nus_transient, log10_flux, axis=0)(xray_filt.nus)
        xray_fluence = integrate.simpson(y=10**(log10flux_xray-26), x=xray_filt.nus, axis=0)
        xray_fluence = np.interp(t_epochs, times_transient, xray_fluence)

        detected = xray_fluence >= 2.6e-11

        return 5 * 300/(24*3600) , {"X-ray-0.5-4keV": np.sum(detected)}
    
    def check_visibility(self, time, ra, dec):
        mjd = Time(time, format="mjd")
        sun_coords = get_sun(mjd)
        sun_theta = np.pi/2 - sun_coords.dec.rad
        sun_phi = sun_coords.ra.rad
        theta = np.pi/2 - dec
        cos_alpha = np.cos(sun_phi) * np.sin(sun_theta) * np.cos(ra) * np.sin(theta) + np.sin(sun_phi) * np.sin(sun_theta) * np.sin(ra) * np.sin(theta) + np.cos(sun_theta) * np.cos(theta)
        return cos_alpha < 0 


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


ska = RadioTelescope("ska",
                     lat = -30.7,
                     long = 21.4,
                     height = 1086.6,
                     fov = 5,
                     exposure_time=300,
                     dead_time=100,
                     thresholds={"radio-1.4GHz": 22.1})

dsa = RadioTelescope("dsa",
                     lat = 34 + 4/60 + 43/3600,
                     long = 107 + 37/60 + 4/3600,
                     height=2124,
                     fov = 10.6,
                     exposure_time=300,
                     dead_time=100,
                     thresholds={"radio-1.4GHz": 22.5})

einsteinprobe = EINSTEINPROBE()