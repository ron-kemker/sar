# -*- coding: utf-8 -*-
'''

File: gotchadata.py
Description: Contains software to process AFRL GOTCHA Volumetric SAR data
Author: Ronald Kemker

'''
from scipy.io import loadmat
import numpy as np
import warnings
from signal_processing import hamming_window

class GOTCHA(object):
    """Processes the AFRL GOTCHA Data
    
    @author: Ronald Kemker

    This code snippet will load and display the data dome data:
    data_path = 'data\Civilian Vehicles\Domes\Camry\Camry_el40.0000.mat'
    cvdata = CVData(data_path, 'Camry')
    cvdata.imshow()

    # Arguments
        data_path: String. File path to desired data file (.mat)
        min_azimuth_angle: Numeric >= 0. Minimum azimuth angle (degrees)
        max_azimuth_angle: Numeric > 0. Maximum azimuth angle (degrees)
        min_frequency: Numeric >= 0. Minimum frequency (in Hz)
        max_frequency: Numeric > 0. Maximum frequency (in Hz)
        bandwidth: Numeric > 0.  Bandwidth to process (in Hz)
        center_freq: Numeric > 0.  Center frequency to process (in Hz)
        taper_func: func. Side-lobe reduction func from singal_processing.py
        single_precision: Boolean.  If false, it will be double precision.
    
    # References
        - [GOTCHA Volumetric SAR Dataset Overview](
          https://www.sdms.afrl.af.mil/index.php?collection=gotcha)
    """
    def __init__(self, data_path, min_azimuth_angle = 0, 
                 max_azimuth_angle=360, 
                 min_frequency = 0, max_frequency=20e9,
                 bandwidth=None, center_frequency=None,
                 taper_func = hamming_window, 
                 verbose = True, single_precision=True):
        
        pol = data_path.split('.mat')[-2][-2:].lower()
        minaz = np.deg2rad(min_azimuth_angle)
        maxaz = np.deg2rad(max_azimuth_angle)
        minfreq = min_frequency
        maxfreq = max_frequency
        
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
        mat = loadmat(data_path)['data'][0][0]
        self.cphd =cdtype( mat['fp'])
        freq = fdtype(mat['freq'][:,0])
        x = fdtype(mat['x'])[0]
        y = fdtype(mat['y'])[0]
        z = fdtype(mat['z'])[0]
        r0 = fdtype(mat['r0'][0])
        azim = fdtype(mat['th'][0])
        elev = fdtype(mat['phi'][0])
        
        try:
            af = mat['af'][0][0]
            self.af_params = np.vstack([af[0], af[1]])
        except ValueError:
            self.af_params = None
            
        try:
            self.r_correct = mat['r_correct']
        except ValueError:
            self.r_correct = None
        
        try:
            self.ph_correct = mat['ph_correct']
        except ValueError:
            self.ph_correct = None
            
        incident_angle = 90.0 - np.mean(elev)
        
        # If center_frequency and bandwidth defined, override frequency range
        if bandwidth is not None and center_frequency is not None:
            half = bandwidth/2.0
            minfreq = center_frequency - half
            maxfreq = center_frequency + half
        
        # Only grab phase history over desired azimuth and frequency range 
        az_idx = np.logical_and(azim >= minaz, azim <= maxaz)
        freq_idx = np.logical_and(freq >= minfreq, freq <= maxfreq)
        
        # Complex phase history data
        self.cphd = self.cphd[freq_idx][:, az_idx]
        
        # Grab the true collection geometries stored in the data
        AntAzim = fdtype(azim[az_idx])
        AntElev = elev[az_idx]
        AntR0 = r0[az_idx]
        AntFreq = fdtype(freq[freq_idx])
        center_freq = (AntFreq[-1] + AntFreq[0])/2.0
        minF = np.min(AntFreq)
        deltaF = AntFreq[1] - AntFreq[0] # Pulse-Bandwidth
        [K, Np] = self.cphd.shape 
        
        # Apply a 2-D hamming window to CPHD for side-lobe suppression
        if taper_func is not None:
            self.cphd = cdtype(taper_func(self.cphd))
                
        # Determine the azimuth angles of the image pulses (in radians)
        AntAz = AntAzim*np.pi/180.0
        AntElev = AntElev*np.pi/180.0
        
        # Determine the average azimuth angle step size (radians)
        deltaAz = np.abs(np.mean(np.diff(np.sort(AntAz))))
        
        # Determine the total azimuth angle of the aperture (radians)
        totalAz = np.max(AntAz) - np.min(AntAz)
        
        # Determine the maximum wavelength (m)
        # The next line was provided by AFRL, but I think it is wrong.
        # maxLambda = c / (minF[0] + (deltaF * K)) # this is minLambda
        maxLambda = c / minF # This is what maxLambda should be I think
        
        # Determine the maximum scene size of the image (m)
        maxWr = c/(2*deltaF)  
        maxWx = maxLambda/(2*deltaAz)
        
        # Determine the resolution of the image (m)
        dr = c/(2*deltaF*K)
        dx = maxLambda/(2*totalAz)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                dwell_angle = np.arccos(c / (4 * dx * center_freq))
            except RuntimeWarning:
                dwell_angle = 0.0

        # Print off some data statictics (if verbose is on)
        f1 = np.min(AntFreq)/1e9
        f2 = np.max(AntFreq)/1e9
        az1 = np.rad2deg(np.min(AntAz))
        az2 = np.rad2deg(np.max(AntAz))
        if verbose:
            print('          Incident Angle: %1.0f deg' % incident_angle)
            print(' Elevation/Grazing Angle: %1.0f deg' % np.mean(elev))
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
            print('             Dwell Angle: %1.1f degrees' % dwell_angle)
            print("")
        
        self.num_pulses = Np
        self.num_samples = K
        self.elevation = AntElev*np.pi/180.0
        self.azimuth = AntAzim*np.pi/180.0
        self.freq = AntFreq
        self.bandwidth = (f1-f2)*1e9
        self.delta_r = fdtype(c/(2.0*self.bandwidth))
        self.r0 = AntR0
        self.antenna_location = np.vstack([x, y, z])
        self.range_extent = maxWr
        self.cross_range_extent = maxWx
        self.range_pixels = int(self.range_extent / dr)
        self.cross_range_pixels = int(self.cross_range_extent / dx )
        self.polarization = pol
        
    # Return Complex Phase History Data
    def getCPHD(self):
        return self.cphd
    