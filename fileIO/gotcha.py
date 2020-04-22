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
from glob import glob

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
    def __init__(self, data_path, min_azimuth_angle = -180, 
                 max_azimuth_angle=180, 
                 min_frequency = 0, max_frequency=20e9,
                 bandwidth=None, center_frequency=None,
                 taper_func = hamming_window, 
                 verbose = True, single_precision=True):
        
        pol = data_path.split('\\')[-1].lower()
        minaz = min_azimuth_angle
        maxaz = max_azimuth_angle
        minfreq = min_frequency
        maxfreq = max_frequency
        
        if single_precision:
            fdtype = np.float32
            cdtype = np.complex64
        else:
            fdtype = np.float64
            cdtype = np.complex128           
        
        c = fdtype(299792458) # Speed of Light
        
        self.cphd = None
        self.r0 = np.zeros((0, ), dtype=fdtype)
        self.azim = np.zeros((0, ), dtype=fdtype)
        self.elev = np.zeros((0, ), dtype=fdtype)
        self.freq = None
        self.antenna_location = np.zeros((3, 0), dtype=fdtype)
        self.r_correct = None
        self.ph_correct = None
        self.af_params = np.zeros((2, 0), dtype=fdtype)

        files = glob(data_path +  '/*.mat')

        # Load model data
        for f in files: 
            self.readMATFile(f, minaz, maxaz)
                            
        # If center_frequency and bandwidth defined, override frequency range
        if bandwidth is not None and center_frequency is not None:
            half = bandwidth/2.0
            minfreq = center_frequency - half
            maxfreq = center_frequency + half
        
        # Only grab phase history over desired azimuth and frequency range 
        az_idx = np.logical_and(self.azim >= minaz, self.azim <= maxaz)
        freq_idx = np.logical_and(self.freq >= minfreq, self.freq <= maxfreq)
        
        # Complex phase history data
        self.cphd = cdtype(self.cphd[freq_idx][:, az_idx])
        [K, Np] = self.cphd.shape
        
        # Grab the true collection geometries stored in the data
        AntAzim = self.azim[az_idx]
        AntElev = self.elev[az_idx]
        AntR0 = self.r0[az_idx]
        self.antenna_location = self.antenna_location[:, az_idx]
        
        if self.af_params.shape[1] > 0:
            self.af_params = self.af_params[:, az_idx]
        
        AntFreq = self.freq[freq_idx]
        center_freq = (AntFreq[-1] + AntFreq[0])/2.0
        minF = np.min(AntFreq)
        deltaF = AntFreq[1] - AntFreq[0] # Pulse-Bandwidth
               
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
        maxLambda = c / minF 
        
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
            print(' Elevation/Grazing Angle: %1.0f deg' % np.mean(self.elev))
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
        self.bandwidth = (f2-f1)*1e9
        self.delta_r = fdtype(c/(2.0*self.bandwidth))
        self.r0 = AntR0
        # self.antenna_location = np.vstack([x, y, z])
        self.range_extent = maxWr
        self.cross_range_extent = maxWx
        self.range_pixels = int(self.range_extent / dr)
        self.cross_range_pixels = int(self.cross_range_extent / dx )
        self.polarization = pol
                
        mid = Np / 2
        if mid > 0:
            self.center_loc = self.antenna_location[:, int(mid)]
        else:
            idx = int(mid)
            self.center_loc = np.mean(self.antenna_location[:, idx:idx+2],1)
    
    def readMATFile(self, file_name, minaz, maxaz):
        mat = loadmat(file_name)['data'][0][0]
        
        azim = mat['th'][0]
        if not (np.any(azim <= maxaz) and np.any(azim >= minaz)): 
            return
        
        if self.cphd is None:
            self.cphd = mat['fp']
        else:
            self.cphd = np.append(self.cphd, mat['fp'], axis=1)

        if self.freq is None:
            self.freq = mat['freq'][:, 0]
            
        x = mat['x'][0]
        y = mat['y'][0]
        z = mat['z'][0]
        self.antenna_location = np.append(self.antenna_location, 
                                          np.vstack([x, y, z]), axis=1)
        
        self.r0 =  np.append(self.r0 , mat['r0'][0])
        self.azim = np.append(self.azim, azim)
        self.elev = np.append(self.elev, mat['phi'][0])
        
        try:
            af = mat['af'][0][0]
            self.af_params = np.append(self.af_params, 
                                       np.vstack([af[0], af[1]]), axis=1)
        except ValueError:
            pass
            
        try:
            self.r_correct = mat['r_correct'][0]
        except ValueError:
            self.r_correct = None
        
        try:
            self.ph_correct = mat['ph_correct'][0]
        except ValueError:
            self.ph_correct = None
    
    # Return Complex Phase History Data
    def getCPHD(self):
        return self.cphd
    