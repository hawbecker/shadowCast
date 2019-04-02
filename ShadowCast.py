import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import pandas as pd
from mpmath import cot
from matplotlib.colors import Normalize as Normalize
import matplotlib.patches as patches


class shadowCast():
    '''
    shadowCast will take the time, location, and specifications of the turbine
    and return the shadow of the turbine over a flat area. The dimensions of
    the area can be defined with the dx, dy, lx, and ly variables.

    '''
    def __init__(self,date_start='2019-08-15', ntimes=48, data_freq='1h',\
                 latoi=39.91, lonoi=-105.23, tower_height=70.0, \
                 tower_width=3.0, rotor_rad=40.0, tower_shadow_weight=0.9,\
                 rotor_shadow_weight=0.2, dx=1.0, dy=1.0, lx=500.0, ly=500.0):
        self.getTimesAndLocation(date_start,ntimes,data_freq,latoi,lonoi)
        self.getTurbineDetails(tower_height,tower_width,rotor_rad,tower_shadow_weight,rotor_shadow_weight)
        self.generateMesh(dx,dy,lx,ly)
        self.calculateSunPosition(ntimes)
        self.calculateShadow(ntimes)

    def getTimesAndLocation(self,date_start,ntimes,data_freq,latoi,lonoi):
        times   = pd.date_range(date_start, periods=ntimes, freq=data_freq)
        loc     = coord.EarthLocation(lon=lonoi * u.deg,
                                      lat=latoi * u.deg)
        self.times    = times
        self.location = loc

    def getTurbineDetails(self,tower_height,tower_width,rotor_rad,tower_shadow_weight,rotor_shadow_weight):
        self.tower_height = tower_height
        self.tower_width = tower_width
        self.rotor_radius = rotor_rad
        self.tower_shadow_weight = tower_shadow_weight
        self.rotor_shadow_weight = rotor_shadow_weight

    def calculateSunPosition(self,ntimes):
        azi_ang = np.zeros((ntimes))
        elv_ang = np.zeros((ntimes))
        for tt,time in enumerate(self.times):
            sun_time = Time(time) #UTC time
            sunpos = coord.AltAz(obstime=sun_time, location=self.location)
            alt    = coord.get_sun(sun_time).transform_to(sunpos).alt
            azi    = coord.get_sun(sun_time).transform_to(sunpos).az
            elv_ang[tt] = alt.degree
            azi_ang[tt] = azi.degree
        self.elevation_angle = elv_ang
        self.azimuth_angle   = azi_ang

    def generateMesh(self,dx,dy,lx,ly):
        x = np.arange(0,lx+0.1,dx) - lx/2.0
        xc = 0.5*(x[1:] + x[:-1])
        y = np.arange(0,ly+0.1,dy) - ly/2.0
        yc = 0.5*(y[1:] + y[:-1])
        xy,yx = np.meshgrid(x,y)
        xyc, yxc = np.meshgrid(xc,yc)
        self.xy = xy
        self.yx = yx
        self.xyc = xyc
        self.yxc = yxc


    def calculateShadow(self,ntimes):
        shadow_length_tower     = np.zeros((ntimes))
        shadow_length_rotor_top = np.zeros((ntimes))
        shadow_length_rotor_bot = np.zeros((ntimes))

        for tt in range(0,ntimes):
            shadow_length_tower[tt]     = self.tower_height*cot(np.radians(self.elevation_angle[tt]))
            shadow_length_rotor_top[tt] = (self.tower_height+self.rotor_radius)*cot(np.radians(self.elevation_angle[tt]))
            shadow_length_rotor_bot[tt] = (self.tower_height-self.rotor_radius)*cot(np.radians(self.elevation_angle[tt]))
        shadow_length_tower[self.elevation_angle <=0.0] = np.nan
        shadow_length_rotor_top[self.elevation_angle <=0.0] = np.nan
        shadow_length_rotor_bot[self.elevation_angle <=0.0] = np.nan
        shadow_ang = self.azimuth_angle - 180.0
        shadow_ang[self.elevation_angle <= 0.0] = np.nan
        shadow_ang[shadow_ang < 0.0] += 360.0

        shadow = np.ones((np.shape(self.xyc)[0],np.shape(self.xyc)[1],ntimes))

        distance = np.sqrt(self.xyc**2 + self.yxc**2)
        angle            = np.degrees(np.arctan(self.xyc/self.yxc))
        angle[self.yxc<0.0]   = angle[self.yxc<0.0] + 180.0
        angle[angle<0.0] = angle[angle<0.0] + 360.0

        for toi in range(0,ntimes):
            D     = shadow_length_tower[toi]
            theta = np.radians(shadow_ang[toi])
            mask = np.ones(np.shape(self.xyc))

            htw = self.tower_width/2.0
            # find the left and right edges of the tower shadow by adding/subtracting 90 degrees from the shadow angle
            tower_left_xs, tower_left_ys = htw*np.sin(np.radians(shadow_ang[toi]-90.0)), htw*np.cos(np.radians(shadow_ang[toi]-90.0))
            tower_rght_xs, tower_rght_ys = htw*np.sin(np.radians(shadow_ang[toi]+90.0)), htw*np.cos(np.radians(shadow_ang[toi]+90.0))
            tower_left_xe, tower_left_ye = D*np.sin(theta)+tower_left_xs, D*np.cos(theta)+tower_left_ys
            tower_rght_xe, tower_rght_ye = D*np.sin(theta)+tower_rght_xs, D*np.cos(theta)+tower_rght_ys
            # Find the slopes & intercepts of these lines to find the cells that are between the two
            tower_left_slope = (tower_left_ys-tower_left_ye) / (tower_left_xs-tower_left_xe)
            tower_rght_slope = (tower_rght_ys-tower_rght_ye) / (tower_rght_xs-tower_rght_xe)
            tower_left_int   = tower_left_ys - tower_left_slope*tower_left_xs
            tower_rght_int   = tower_rght_ys - tower_rght_slope*tower_rght_xs
            tower_axis_slope = (tower_left_ys-tower_rght_ys) / (tower_left_xs-tower_rght_xs)
            # Make is so that the angle is 0 in the direction of the shadow... remove 180 > A > 270
            shadow_angle = angle-shadow_ang[toi]
            shadow_angle[shadow_angle<0.0] += 360.0
            # Find points between the two lines...
            mask[((self.yxc>=tower_left_slope*self.xyc+tower_left_int) & (self.yxc<=tower_rght_slope*self.xyc+tower_rght_int)) | 
                 ((self.yxc<=tower_left_slope*self.xyc+tower_left_int) & (self.yxc>=tower_rght_slope*self.xyc+tower_rght_int))] = (1.0 - self.tower_shadow_weight)
            # Find points that are less than the shadow distance
            mask[distance>shadow_length_tower[toi]] = 1.0
            # Find points in the direction of the shadow
            mask[(shadow_angle>90.0) & (shadow_angle<270.0)] = 1.0
            
            # Define the ellipse!
            g_ell_center = (D*np.sin(theta), D*np.cos(theta))
            g_ell_height = self.rotor_radius
            g_ell_width  = shadow_length_rotor_top[toi] - shadow_length_rotor_bot[toi]
            ell_angle    = np.degrees(np.arctan2(D*np.cos(theta), D*np.sin(theta)))
            # Get the angles of the axes
            cos_angle = np.cos(np.radians(180.-ell_angle))
            sin_angle = np.sin(np.radians(180.-ell_angle))
            # Find the distance of each gridpoint from the ellipse
            exc = self.xyc - g_ell_center[0]
            eyc = self.yxc - g_ell_center[1]
            exct = exc * cos_angle - eyc * sin_angle
            eyct = exc * sin_angle + eyc * cos_angle 
            # Get the radial distance away from ellipse: cutoff at r = 1
            rad_cc = (exct**2/(g_ell_width/2.)**2) + (eyct**2/(g_ell_height/2.)**2)

            g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=ell_angle)
            mask[rad_cc<=1.0] = mask[rad_cc<=1.0]*(1.0-self.rotor_shadow_weight)

            shadow[:,:,toi] = mask
        self.shadow = shadow



