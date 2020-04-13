# -*- coding: utf-8 -*-
'''

File: image_formation.py
Description: Contains SAR image formation algorithms, e.g., backprojection
Author: Ronald Kemker

'''

from multiprocessing import Pool
from numpy.fft import ifft, fftshift
import numpy as np


def image_projection(sar_obj, num_x_samples, num_y_samples, scene_extent_x,
                     scene_extent_y, scene_center_x=0, scene_center_y=0,
                     single_precision=True):
    """Defines the image plane to project the image onto
    
    @author: Ronald Kemker

    The code snippet from examples/civilian_data_dome_example.py shows how to
    how this function is used.

    # Arguments
        sar_obj: Object.  Defines the collected CPHD.
        num_x_samples: Int > 0. Number of samples in x direction
        num_y_samples: Int > 0. Number of samples in y direction
        scene_extent_x: Numeric > 0. Scene extent x (m)
        scene_extent_y: Numeric > 0. Scene extent y (m)
        scene_center_x: Numeric. Center of image scene in x direction (m)
        scene_center_y: Numeric. Center of image scene in y direction (m)
        single_precision: Boolean.  If false, it will be double precision.
    
    # References
        - [Civilian Vehicle Data Dome Overview](
          https://www.sdms.afrl.af.mil/index.php?collection=cv_dome)
    """    
    Nx = num_x_samples
    Ny = num_y_samples
    Wx = scene_extent_x
    Wy = scene_extent_y
    x0 = scene_center_x
    y0 = scene_center_y
        
    if single_precision:
        fdtype = np.float32
    else:
        fdtype = np.float64
    
    x_vec = np.linspace(x0 - Wx/2, x0 + Wx/2, Nx, dtype=fdtype)
    y_vec = np.linspace(y0 - Wy/2, y0 + Wy/2, Ny, dtype=fdtype)
    [x_mat, y_mat] = np.meshgrid(x_vec, y_vec)
    
    output_dict = {'x_vec': x_vec,
                   'y_vec': y_vec,
                   'x_mat': x_mat,
                   'y_mat': y_mat,
                   }
    
    return output_dict

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

def backProjection(sar_obj, image_plane, fft_samples=None, n_jobs=1, 
                   single_precision=True):

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
        image_plane: Object.  Defines the image plane to project onto.
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
    
    upsample = 6
    
    # Required Paramters for Backprojection Algorithm
    cphd = sar_obj.cphd
    freq = sar_obj.freq
    Np = sar_obj.num_pulses
    az = sar_obj.azimuth
    el = sar_obj.elevation
    alt = sar_obj.altitude
    x_mat = image_plane['x_mat']
    y_mat = image_plane['y_mat']
    
    if fft_samples is None:
        Nfft = 2**(int(np.log2(Np*upsample))+1)
    else:
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
    
    z_mat = np.ones(x_mat.shape, fdtype)
    
    # Multi-processing approach
    if n_jobs > 1:
        args = []
        for ii in range(Np):
            args += [(cphd[:,ii], Nfft, x_mat, y_mat, 
                     z_mat*alt[ii], el, az[ii], phCorr_exp, 
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
                    y_mat, z_mat*alt[ii], el, az[ii],  
                    phCorr_exp, min_rvec, max_rvec, r_vec)
            im_final[idx] = im_final[idx] + img
        
    print("") 
    return im_final    