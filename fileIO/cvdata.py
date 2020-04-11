# -*- coding: utf-8 -*-
'''
File: cvdata.py
Description: Contains software to process AFRL Civilian Dome Datasets
Author: Ronald Kemker
'''
from scipy.io import loadmat
import numpy as np
import warnings
from signal_processing import hamming_window

class CVData(object):
    """Processes the Civilian Vechicle SAR Datadome Data
    
    @author: Ronald Kemker

    This code snippet will load and display the data dome data:
    data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'        
    cvdata = CVData(data_path, 'camry', 
                    min_azimuth_angle=44, 
                    max_azimuth_angle=46, 
                    polarization='vv',
                    center_frequency=9.6e9, 
                    bandwidth=300e6,
                    taper_flag=True,
                    )
    image = backProjection(cvdata)
    imshow(image, cvdata.x_vec, cvdata.y_vec)

    # Arguments
        data_path: String. File path to desired data file (.mat)
        target: String. What vehicle is present in the data
        polarization: String.  What polarization to image (HH,HV,VV)
        min_azimuth_angle: Numeric >= 0. Minimum azimuth angle (degrees)
        max_azimuth_angle: Numeric > 0. Maximum azimuth angle (degrees)
        min_frequency: Numeric >= 0. Minimum frequency (in Hz)
        max_frequency: Numeric > 0. Maximum frequency (in Hz)
        bandwidth: Numeric > 0.  Bandwidth to process (in Hz)
        center_freq: Numeric > 0.  Center frequency to process (in Hz)
        taper_flag: Boolean. Add a hamming taper for sidelobe control
        scene_extent_x: Numeric > 0. Scene extent x (m)
        scene_extent_y: Numeric > 0. Scene extent y (m)
        num_x_samples: Int > 0. Number of samples in x direction
        num_y_samples: Int > 0. Number of samples in y direction
        scene_center_x: Numeric. Center of image scene in x direction (m)
        scene_center_y: Numeric. Center of image scene in y direction (m)
        single_precision: Boolean.  If false, it will be double precision.
    
    # References
        - [Civilian Vehicle Data Dome Overview](
          https://www.sdms.afrl.af.mil/index.php?collection=cv_dome)
    """
    def __init__(self, data_path, target, polarization='vv',
                 min_azimuth_angle = 0, max_azimuth_angle=360, 
                 min_frequency = 0, max_frequency=20e9,
                 bandwidth=None, center_frequency=None,
                 taper_flag = True, scene_extent_x = 10, 
                 scene_extent_y=10, num_x_samples=41,
                 num_y_samples=41, scene_center_x = 0, scene_center_y=0,
                 verbose = True, n_jobs=1, single_precision=True):
        
        self.target = target
        pol = polarization
        minaz = min_azimuth_angle
        maxaz = max_azimuth_angle
        minfreq = min_frequency
        maxfreq = max_frequency
        Wx = scene_extent_x
        Wy = scene_extent_y
        Nx = num_x_samples
        Ny = num_y_samples
        x0 = scene_center_x
        y0 = scene_center_y
        
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
        incident_angle = 90 - elev
        # self.elevation = elev
        
        # If center_frequency and bandwidth defined, override frequency range
        if bandwidth is not None and center_frequency is not None:
            half = bandwidth/2.0
            minfreq = center_frequency - half
            maxfreq = center_frequency + half
        
        # Only grab phase history over desired azimuth and frequency range 
        az_idx = np.logical_and(azim >= minaz, azim <= maxaz)
        freq_idx = np.logical_and(freq >= minfreq, freq <= maxfreq)
        
        # Complex phase history data
        self.cphd = cdtype(data[pol][0,0])[freq_idx][:, az_idx]
        
        # Grab the true collection geometries stored in the data
        AntAzim = fdtype(azim[az_idx])
        AntElev = fdtype(data['elev'][0,0][0,0])
        AntFreq = fdtype(freq[freq_idx])
        center_freq = (AntFreq[-1] + AntFreq[0])/2.0
        minF = np.min(AntFreq)
        deltaF = AntFreq[1] - AntFreq[0] # Pulse-Bandwidth
        [K, Np] = self.cphd.shape 
        
        # Apply a 2-D hamming window to CPHD for side-lobe suppression
        if taper_flag:
            self.cphd = fdtype(hamming_window(self.cphd))
        
        # Define the spatial extent with MeshGrid for quicker processing
        self.x_vec = np.linspace(x0 - Wx/2, x0 + Wx/2, Nx, dtype=fdtype)
        self.y_vec = np.linspace(y0 - Wy/2, y0 + Wy/2, Ny, dtype=fdtype)
        [x_mat, y_mat] = np.meshgrid(self.x_vec, self.y_vec)
        z_mat = np.zeros(x_mat.shape, fdtype)
        
        # Determine the azimuth angles of the image pulses (radians)
        AntAz = np.sort(AntAzim*np.pi/180.0)

        # Determine the average azimuth angle step size (radians)
        deltaAz = np.abs(np.mean(np.diff(AntAz)))
        
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
        f1 = AntFreq[0]/1e9
        f2 = AntFreq[-1]/1e9
        az1 = np.rad2deg(AntAz[0])
        az2 = np.rad2deg(AntAz[-1])
        if verbose:
            print('        Using %s model...' % target)
            print('          Incident Angle: %1.0f deg' % incident_angle)
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
            print('             Dwell Angle: %1.1f degrees' % dwell_angle)
            print("")
        
        self.num_pulses = Np
        self.num_samples = K
        self.elevation = AntElev*np.pi/180.0
        self.azimuth = AntAzim*np.pi/180.0
        self.pos = [x_mat, y_mat, z_mat]
        self.freq = AntFreq
        
    # Return Complex Phase History Data
    def getCPHD(self):
        return self.cphd
        
if __name__ == "__main__": 
    from image_formation import backProjection
    from utils import imshow
    
    data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
    target = data_path.split('\\')[-2]
        
    cvdata = CVData(data_path, target, 
                    min_azimuth_angle=44, 
                    max_azimuth_angle=46, 
                    polarization='vv',
                    center_frequency=9.6e9, 
                    bandwidth=300e6,
                    taper_flag=True,
                    )
    
    image = backProjection(cvdata)
    imshow(image, cvdata.x_vec, cvdata.y_vec)
