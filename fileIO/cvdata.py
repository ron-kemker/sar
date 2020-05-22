# -*- coding: utf-8 -*-
'''
File: cvdata.py
Description: Contains software to process AFRL Civilian Dome Datasets
Author: Ronald Kemker
'''
from scipy.io import loadmat
import numpy as np

class CVData(object):
    """Processes the Civilian Vechicle SAR Datadome Data
    
    @author: Ronald Kemker

    See examples/cv_example.py for example code.

    # Arguments
        data_path         : String. 
                            File path to desired data file (.mat)
        target            : String. 
                            What vehicle is present in the data
        polarization      : String.  
                            What polarization to image (HH,HV,VV)
        min_azimuth_angle : Numeric >= 0. 
                            Minimum azimuth angle (degrees)
        max_azimuth_angle : Numeric > 0. 
                            Maximum azimuth angle (degrees)
        min_frequency     : Numeric >= 0. 
                            Minimum frequency (in Hz)
        max_frequency     : Numeric > 0. 
                            Maximum frequency (in Hz)
        bandwidth         : Numeric > 0.  
                            Bandwidth to process (in Hz)
        center_freq       : Numeric > 0.  
                            Center frequency to process (in Hz)
        single_precision  : Boolean.  
                            If false, it will be double precision.
        verbose           : Boolean
                            If true, prints off statistics for the data
        altitude          : Numeric. 
                            FLight altitude (meters)

    # References
        - [Civilian Vehicle Data Dome Overview](
          https://www.sdms.afrl.af.mil/index.php?collection=cv_dome)
    """
    def __init__(self, data_path, target, polarization='vv',
                 min_azimuth_angle = 0, max_azimuth_angle=360, 
                 min_frequency = 0, max_frequency=20e9,
                 bandwidth=None, center_frequency=None,
                 verbose = True, single_precision=True,
                 altitude=10):
        
        self.target = target
        pol = polarization
        minaz = min_azimuth_angle
        maxaz = max_azimuth_angle
        minfreq = min_frequency
        maxfreq = max_frequency
        Z_a = altitude
        
        if single_precision:
            fdtype = np.float32
            cdtype = np.complex64
        else:
            fdtype = np.float64
            cdtype = np.complex128           
        
        c = fdtype(299792458) # Speed of Light
        
        # Load model data
        filename = data_path.split('\\')[-1]
        if verbose:
            print('Loading %s...\n' % filename)
        data = loadmat(data_path)['data']
        azim = fdtype(data['azim'][0][0][0])
        freq = fdtype(data['FGHz'][0,0][:,0] * 1e9)
        elev = fdtype(data['elev'][0][0][0][0]) # AKA Grazing angle
        
        # If center_frequency and bandwidth defined, override frequency range
        if bandwidth is not None and center_frequency is not None:
            half = bandwidth/2.0
            minfreq = center_frequency - half
            maxfreq = center_frequency + half
        
        # Only grab phase history over desired azimuth and frequency range 
        az_idx = np.logical_and(azim >= minaz, azim <= maxaz)
        freq_idx = np.logical_and(freq >= minfreq, freq <= maxfreq)
        
        # Complex phase history data
        self.cphd = cdtype(data[pol][0,0].T)[:, freq_idx][az_idx]
        
        # Grab the true collection geometries stored in the data
        AntAzim = fdtype(azim[az_idx])
        AntElev = fdtype(data['elev'][0,0][0,0])
        AntFreq = fdtype(freq[freq_idx])
        center_freq = (AntFreq[-1] + AntFreq[0])/2.0
        minF = np.min(AntFreq)
        deltaF = AntFreq[1] - AntFreq[0] # Pulse-Bandwidth
        [Np, K] = self.cphd.shape 
                        
        # Determine the azimuth angles of the image pulses (radians)
        AntAz = AntAzim*np.pi/180.0
        AntElev = AntElev*np.pi/180.0
        # Determine the average azimuth angle step size (radians)
        deltaAz = np.abs(np.mean(np.diff(np.sort(AntAz))))
        
        # Determine the total azimuth angle of the aperture (radians)
        totalAz = np.max(AntAz) - np.min(AntAz)
        
        # Determine the maximum wavelength (m)
        maxLambda = c / minF 
        
        # Determine the maximum scene size of the image (m)
        maxWr = c/(2*deltaF)  
        maxWx = maxLambda/(2*deltaAz)
        
        # Determine the resolution of the image (m)
        dr = c/(2*deltaF*K)
        dx = maxLambda/(2*totalAz)

        # Print off some data statictics (if verbose is on)
        f1 = np.min(AntFreq)/1e9
        f2 = np.max(AntFreq)/1e9
        az1 = np.rad2deg(np.min(AntAz))
        az2 = np.rad2deg(np.max(AntAz))
        if verbose:
            print('        Using %s model...' % target)
            print(' Elevation/Grazing Angle: %1.0f deg' % elev)
            print('  Center Frequency (GHz): %1.1f' % (center_freq/1e9))
            print('   Frequency Range (GHz): %1.2f-%1.2f'%(f1, f2))
            print('  Maximum wavelength [m]: %1.1e' % (maxLambda))
            print('        Number of Pulses: %d' % Np)
            print('Frequency Bins per Pulse: %d' % K)
            print('           Azimuth Range: %1.1f-%1.1f deg' % (az1,az2))
            print('  Max Scene Size (range): %1.2fm' % maxWr)
            print('Max Scene Size (x-range): %1.2fm' % maxWx)
            print('        Range Resolution: %1.2fm'  % dr)
            print('  Cross-Range Resolution: %1.2fm'  % dx)
            print("")
        
        self.num_pulses = Np
        self.num_samples = K
        self.elevation = AntElev*np.ones((Np, ), fdtype)
        self.azimuth = AntAz
        self.freq = AntFreq
        self.bandwidth = (f1-f2)*1e9
        self.delta_r = fdtype(c/(2.0*self.bandwidth))
        self.range_extent = maxWr
        self.cross_range_extent = maxWx
        self.range_pixels = int(self.range_extent / dr)
        self.cross_range_pixels = int(self.cross_range_extent / dx )
        self.polarization = pol
        self.f_0 = (AntFreq[0] + AntFreq[-1])/2
        self.k_r = 4*np.pi*AntFreq/c
        self.n_hat = np.array([ 0, 0 , 1]  , dtype=fdtype)
        
        Z_a = np.ones(self.elevation.shape) * Z_a
        X_a = Z_a / np.tan(self.elevation) * np.sin(self.azimuth)
        Y_a = Z_a / np.tan(self.elevation) * np.cos(self.azimuth)
        
        self.antenna_location = np.vstack((X_a[np.newaxis],Y_a[np.newaxis],
                                           Z_a[np.newaxis]))
