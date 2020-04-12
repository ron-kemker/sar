# -*- coding: utf-8 -*-
'''

File: image_formation.py
Description: Contains SAR image formation algorithms, e.g., backprojection
Author: Ronald Kemker

'''

from multiprocessing import Pool
from numpy.fft import ifft, fftshift
import numpy as np

# This processes a single pulse (broken out for parallelization)
def bp_helper(ph_data, Nfft, x_mat, y_mat, z_mat, AntElev, AntAzim, 
               phCorr_exp, min_rvec, max_rvec, r_vec):
                
    # Form the range profile with zero padding added
    rc = fftshift(ifft(ph_data,Nfft))

    # Calculate differential range for each pixel in the image (m)
    dR =  x_mat * np.cos(AntElev) * np.cos(AntAzim) 
    dR += y_mat * np.cos(AntElev) * np.sin(AntAzim) 
    dR += z_mat * np.sin(AntElev)

    # Calculate phase correction for image
    phCorr = np.exp(phCorr_exp*dR)

    # Determine which pixels fall within the range swath
    idx = np.logical_and(dR > min_rvec, dR < max_rvec)

    # Update the image using linear interpolation
    return np.interp(dR[idx], r_vec, rc) * phCorr[idx], idx

def backProjection(sar_obj, fft_samples=512, n_jobs=1, single_precision=True):

    """Performs backprojection image-formation
    
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
        sar_obj: Object. One of the fileIO SAR data readers.
        fft_samples: Int > 0. Number of samples in FFT
        n_jobs: Integer > 0. Number of multiprocessor jobs.
        single_precision: Boolean.  If false, it will be double precision.
    
    # References
        - TODO
    """
    
    if single_precision:
        fdtype = np.float32
        cdtype = np.complex64
    else:
        fdtype = np.float64
        cdtype = np.complex128 
    
    # Required Paramters for Backprojection Algorithm
    cphd = sar_obj.cphd
    freq = sar_obj.freq
    Np = sar_obj.num_pulses
    az = sar_obj.azimuth
    el = sar_obj.elevation
    x_mat = sar_obj.pos[0]
    y_mat = sar_obj.pos[1]
    z_mat = sar_obj.pos[2]
    Nfft = fft_samples
    
    c =  fdtype(299792458) # Speed of Light
    deltaF = freq[1] - freq[0]
    maxWr = c/(2*deltaF)
    minF = np.min(freq)

    # Calculate the range to every bin in the range profile (m)
    r_vec = np.linspace(-Nfft/2,Nfft/2-1, Nfft, dtype=fdtype)* maxWr/Nfft
        
    # Precomputing constant values inside the loop
    min_rvec = np.min(r_vec)
    max_rvec = np.max(r_vec)
    phCorr_exp = np.complex64(1j*4.0*np.pi*minF/c)
    im_final = np.zeros(x_mat.shape, cdtype);
    
    # Multi-processing approach
    if n_jobs > 1:
        args = []
        for ii in range(Np):
            args += [(cphd[:,ii], Nfft, x_mat, y_mat, 
                     z_mat, el, az[ii], phCorr_exp, 
                     min_rvec, max_rvec, r_vec)]

        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(bp_helper, args)
                        
        for ii in range(Np):
            idx = results[ii][1]
            im_final[idx] = im_final[idx] + results[ii][0]
    
    # Single Processor approach
    else:
        print("")
        for ii in range(Np):
            print('\rProcessing: %1.1f%%' % (ii/Np *100.0), end="") 
            [img, idx] = bp_helper(cphd[:,ii], Nfft, x_mat, 
                    y_mat, z_mat, el, az[ii],  
                    phCorr_exp, min_rvec, max_rvec, r_vec)
            im_final[idx] = im_final[idx] + img
        
    print("") 
    return im_final    