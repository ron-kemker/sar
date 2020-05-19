# -*- coding: utf-8 -*-
'''

File: image_formation.py
Description: Contains SAR image formation algorithms, e.g., backprojection
Author: Ronald Kemker

'''

from multiprocessing import Pool
from numpy.fft import ifft, fftshift
import numpy as np
from signal_processing import polyphase_interp as poly_int
from utils import ft2
from numpy.linalg import norm
from signal_processing import taylor

# This processes a single pulse (broken out for parallelization)
def bp_helper(ph_data, Nfft, x_mat, y_mat, z_mat, AntElev, AntAzim, 
               phCorr_exp, min_rvec, max_rvec, r_vec, 
               interp_func=np.interp):
                
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
    return interp_func(dR[idx], r_vec, rc) * phCorr[idx], idx

def back_projection(sar_obj, fft_samples=None, n_jobs=1, 
                   single_precision=True, 
                   num_x_samples=501, num_y_samples=501, 
                   scene_extent_x=None, scene_extent_y=None, 
                   scene_center_x=0, scene_center_y=0,):

    """Performs backprojection image-formation
    
    @author: Ronald Kemker

    This code snippet will load and display the data dome data:
    data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'        
    cvdata = CVData(data_path, 'camry', 
                    polarization='vv',
                    center_frequency=9.6e9, 
                    bandwidth=300e6,
                    )
    image = backProjection(cvdata)
    imshow(image, cvdata.x_vec, cvdata.y_vec)

    # Arguments
        sar_obj: Object. One of the fileIO SAR data readers.
        fft_samples: Int > 0. Number of samples in FFT
        n_jobs: Integer > 0. Number of multiprocessor jobs.
        single_precision: Boolean.  If false, it will be double precision.
        num_x_samples: Int > 0. Number of samples in x direction
        num_y_samples: Int > 0. Number of samples in y direction
        scene_extent_x: Numeric > 0. Scene extent x (m)
        scene_extent_y: Numeric > 0. Scene extent y (m)
        scene_center_x: Numeric. Center of image scene in x direction (m)
        scene_center_y: Numeric. Center of image scene in y direction (m)
    
    # References
        - [Civilian Vehicle Data Dome Overview](
          https://www.sdms.afrl.af.mil/index.php?collection=cv_dome)
    """
    
    if single_precision:
        fdtype = np.float32
        cdtype = np.complex64
    else:
        fdtype = np.float64
        cdtype = np.complex128 
    
    upsample = 6
    c =  fdtype(299792458) # Speed of Light

    # Required Paramters for Backprojection Algorithm
    if scene_extent_x is None:
        Wx = sar_obj.range_extent
    else:
        Wx = scene_extent_x
        
    if scene_extent_y is None:
        Wy = sar_obj.cross_range_extent
    else:
        Wy = scene_extent_y
    
    x0 = scene_center_x
    y0 = scene_center_y
    cphd = sar_obj.cphd
    freq = sar_obj.freq
    Np = sar_obj.num_pulses
    az = sar_obj.azimuth
    el = sar_obj.elevation

    x_vec = np.linspace(x0 - Wx/2, x0 + Wx/2, num_x_samples, dtype=fdtype)
    y_vec = np.linspace(y0 - Wy/2, y0 + Wy/2, num_y_samples, dtype=fdtype)
    [x_mat, y_mat] = np.meshgrid(x_vec, y_vec)
    
    if fft_samples is None:
        Nfft = 2**(int(np.log2(Np*upsample))+1)
    else:
        Nfft = fft_samples

    
    deltaF = freq[1] - freq[0]
    maxWr = c/(2*deltaF)
    minF = np.min(freq)

    # Calculate the range to every bin in the range profile (m)
    r_vec = np.linspace(-Nfft/2,Nfft/2-1, Nfft, dtype=fdtype)* maxWr/Nfft
        
    # Precomputing constant values inside the loop
    min_rvec = np.min(r_vec)
    max_rvec = np.max(r_vec)
    phCorr_exp = cdtype(1j*4.0*np.pi*minF/c)
    im_final = np.zeros(x_mat.shape, cdtype);
    
    # TODO: This currently doesn't account for non-flat terrain.
    z_mat = np.zeros(x_mat.shape, fdtype)
    
    # Multi-processing approach
    if n_jobs > 1:
        args = []
        for ii in range(Np):
            args += [(cphd[ii], Nfft, x_mat, y_mat, 
                     z_mat, el[ii], az[ii], phCorr_exp, 
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
            [img, idx] = bp_helper(cphd[ii], Nfft, x_mat, 
                    y_mat, z_mat, el[ii], az[ii],  
                    phCorr_exp, min_rvec, max_rvec, r_vec)
            im_final[idx] = im_final[idx] + img
        
    print("") 
    return im_final    

def polar_format_algorithm(sar_obj, single_precision=True,
                            interp_func = poly_int, upsample=True,
                            taylor_weighting=0):
    """Performs polar format algorithm for image-formation
    
    @author: Ronald Kemker

    See examples/sandia_example.py for example.

    # Arguments
                 sar_obj: Object. 
                          One of the fileIO SAR data readers.
        single_precision: Boolean.  Default=True  
                          If false, it will be double precision.
             interp_func: Function.  Default=polyphase_interpolation
                          The type of interpolation used for polar formatting
                upsample: Boolean. 
                          Upsample to the nearest power of 2.
        taylor_weighting: Numeric.  Default=0 (No Taylor Weighting)
                          Taylor weighting factor (in dB) for sidelobe 
                          mitigation 

    # References
        - Carrera, Goodman, and Majewski (1995).
    """    
    #Retrieve relevent parameters
    pi          =   np.pi
    c           =   299792458.0
    K           =   sar_obj.num_samples
    Np          =   sar_obj.num_pulses
    f_0         =   sar_obj.f_0
    pos         =   sar_obj.antenna_location.T
    k           =   sar_obj.k_r
    n_hat       =   sar_obj.n_hat
    cphd        =   sar_obj.cphd
    delta_r     =   sar_obj.delta_r

    # Precision of the output data
    if single_precision:
        fdtype = np.float32
        cdtype = np.complex64
    else:
        fdtype = np.float64
        cdtype = np.complex128

    # Find the center pulse
    if np.mod(Np,2)>0:
        R_c = pos[int(Np/2)]
    else:
        R_c = np.mean(pos[int(Np/2-1):int(Np/2+1)], axis = 0)

    # Define the shape of the output image
    if upsample:
        nu = 2**int(np.log2(K)+bool(np.mod(np.log2(K),1)))
        nv = 2**int(np.log2(Np)+bool(np.mod(np.log2(Np),1)))
    else:
        nu = K
        nv = Np

    #Define resolution.  This should be less than the system resolution limits
    du = delta_r*K/nu
    dv = du # TODO: Could consider custom aspect ratios
            
    #Derive image plane spatial frequencies
    k_ui = 2*pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nu)
    k_vi = 2*pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), nv)

    #Compute k_xi offset
    psi = pi/2-np.arccos(np.dot(R_c,n_hat)/norm(R_c))
    k_ui = k_ui + 4*pi*f_0/c*np.cos(psi)
        
    #Compute x and y unit vectors. x defined to lie along R_c.
    u_hat = (R_c-np.dot(R_c,n_hat)*n_hat)/norm((R_c-np.dot(R_c,n_hat)*n_hat))
    v_hat = np.cross(u_hat,n_hat)
    
    #Compute r_hat, the diretion of k_r, for each pulse
    r_hat = (pos.T/norm(pos,axis=1)).T
        
    #Compute kx and ky meshgrid
    k_matrix = np.tile(k,(Np,1))
    ku = np.matmul(r_hat,u_hat.T)[:,np.newaxis] * k_matrix
    kv = np.matmul(r_hat,v_hat.T)[:,np.newaxis] * k_matrix
    
    # Taylor Weighting (none is default)
    win1 = taylor(cphd.shape[1], taylor_weighting)
    win2 = taylor(cphd.shape[0], taylor_weighting)

    #Radially interpolate (i.e. range) kx and ky data from polar raster
    #onto evenly spaced kx_i and ky_i grid for each pulse
    rad_interp = np.zeros([Np,nu], dtype=cdtype)
    ky_new = np.zeros([Np,nu], dtype=fdtype)
    for i in range(Np):
        rad_interp.real[i,:] = interp_func(k_ui, ku[i,:], 
            cphd.real[i,:]*win1, left=0, right=0)
        rad_interp.imag[i,:] = interp_func(k_ui, ku[i,:], 
            cphd.imag[i,:]*win1, left=0, right=0)
        ky_new[i,:] = np.interp(k_ui, ku[i,:], kv[i,:])  
    
    #Interpolate in along track direction to obtain polar formatted data
    polar_interp = np.zeros([nv,nu], dtype=cdtype)
    
    # Force ky_new vertices to be in ascending order to match k_vi
    if (ky_new[Np//2, nu//2] > ky_new[Np//2+1, nu//2]):
        ky_new = ky_new[::-1]
        rad_interp = rad_interp[::-1]
    
    # Azimuth Interpolation
    for i in range(nu):
        polar_interp.real[:,i] = interp_func(k_vi, ky_new[:,i], 
            rad_interp.real[:,i]*win2, left=0, right=0)
        polar_interp.imag[:,i] = interp_func(k_vi, ky_new[:,i], 
            rad_interp.imag[:,i]*win2, left=0, right=0)

    return ft2(np.nan_to_num(polar_interp))  