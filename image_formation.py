# -*- coding: utf-8 -*-
'''

File: image_formation.py
Description: Contains SAR image formation algorithms, e.g., backprojection
Author: Ronald Kemker

'''

from multiprocessing import Pool
from numpy.fft import ifft, fftshift, fft2
from numpy.linalg import norm
import numpy as np
from utils import polyphase_interp as poly_int
from signal_processing import hamming_window
from scipy.stats import linregress

def image_projection(sar_obj, num_x_samples, num_y_samples, scene_extent_x,
                     scene_extent_y, scene_center_x=0, scene_center_y=0,
                     single_precision=True, upsample=True, res_factor=1.0):
    """Defines the image plane to project the image onto based on the desired
        extent of the scene.
    
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
    
    num_samples = sar_obj.num_samples
    num_pulses = sar_obj.num_pulses
    delta_r = sar_obj.delta_r
    aspect = Wy/Wx
        
    #Define image plane parameters
    if upsample:
        nu= 2**int(np.log2(num_samples)+bool(np.mod(np.log2(num_samples),1)))
        nv= 2**int(np.log2(num_pulses)+bool(np.mod(np.log2(num_pulses),1)))
    else:
        nu= num_samples
        nv= num_pulses

    # Define range and cross-range locations
    x_vec = np.linspace(x0 - Wx/2, x0 + Wx/2, nu, dtype=fdtype)
    y_vec = np.linspace(y0 - Wy/2, y0 + Wy/2, nv, dtype=fdtype)
    [x_mat, y_mat] = np.meshgrid(x_vec, y_vec)

    #Define resolution.  This should be less than the system resolution limits
    du = delta_r*res_factor*num_samples/nu
    dv = aspect*du
        
    #Derive image plane spatial frequencies
    k_u = 2*np.pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nu, dtype=fdtype)
    k_v = 2*np.pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), nv, dtype=fdtype)   
    
    output_dict = {'x_vec': x_vec,
                   'y_vec': y_vec,
                   'x_mat': x_mat,
                   'y_mat': y_mat,
                   'k_u' : k_u,
                   'k_v' : k_v,
                   }
    
    return output_dict

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
    phCorr_exp = cdtype(1j*4.0*np.pi*minF/c)
    im_final = np.zeros(x_mat.shape, cdtype);
    
    # TODO: This currently doesn't account for non-flat terrain.
    z_mat = np.zeros(x_mat.shape, fdtype)
    
    # Multi-processing approach
    if n_jobs > 1:
        args = []
        for ii in range(Np):
            args += [(cphd[:,ii], Nfft, x_mat, y_mat, 
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
            [img, idx] = bp_helper(cphd[:,ii], Nfft, x_mat, 
                    y_mat, z_mat, el[ii], az[ii],  
                    phCorr_exp, min_rvec, max_rvec, r_vec)
            im_final[idx] = im_final[idx] + img
        
    print("") 
    return im_final    



def polar_format_algorithm(sar_obj, single_precision=True, upsample=True):

    """Performs polar format algorithm for image-formation
    
    @author: Ronald Kemker

    See examples/basic_example.py for example.

    # Arguments
        sar_obj: Object. One of the fileIO SAR data readers.
        single_precision: Boolean.  If false, it will be double precision.
        upsample : Boolean. Should we upsample to the nearest power of 2.
            
    # References
        - Carrera, Goodman, and Majewski (1995).
    """
    
    if single_precision:
        fdtype = np.float32
        cdtype = np.complex64
    else:
        fdtype = np.float64
        cdtype = np.complex128
        
    #Retrieve relevent parameters
    c           =   299792458.0
    Np          =   sar_obj.num_pulses
    K           =   sar_obj.num_samples
    pos         =   sar_obj.antenna_location
    cphd        =   sar_obj.cphd
    f           =   sar_obj.freq

    #Define image plane parameters
    if upsample:
        NPHr= 2**int(np.log2(K)+bool(np.mod(np.log2(K),1)))
        NPHa= 2**int(np.log2(Np)+bool(np.mod(np.log2(Np),1)))
    else:
        NPHr = K
        NPHa = Np  
    
    # Computer other useful variables
    center_pulse = int(Np/2)
    R0 = np.sqrt(np.sum(pos**2, 0))
    
    # Rotate the XY-plane
    theta = np.arctan2(pos[1, center_pulse], pos[0, center_pulse])
    A = np.array([np.cos(theta), np.sin(theta), -np.sin(theta), 
                  np.cos(theta)]).reshape(2,2)
    pos[0:2] = np.matmul(A , pos[0:2])
    
    # Define the ouput Keystone
    kx = 4*np.pi*f/c*pos[0,0]/R0[0]
    ky= 4*np.pi*f/c*pos[1,0]/R0[0]
    kx_min = np.min(kx)
    kx = 4*np.pi*f/c * pos[0,center_pulse]/R0[center_pulse]
    Kx = np.linspace(kx_min, np.max(kx), NPHr, dtype=fdtype)
    Ky = np.linspace(-np.max(np.abs(ky)), np.max(np.abs(ky)), NPHa, 
                     dtype=fdtype)
    
    # Range Interpolation
    range_interp_real = np.zeros((Np, NPHr), fdtype)
    range_interp_imag = np.zeros((Np, NPHr), fdtype)
    for i in range(Np):
        kx = 4*np.pi*f/c*pos[0,i]/R0[i] 
        range_interp_real[i] = np.interp(Kx, kx, cphd.real[:,i])
        range_interp_imag[i] = np.interp(Kx, kx, cphd.imag[:,i])
 
    # Azimuth Interpolation
    az_interp_real = np.zeros((NPHa, NPHr), fdtype)
    az_interp_imag = np.zeros((NPHa, NPHr), fdtype)              
    for i in range(NPHr):
        Ky_keystone = Kx[i] * pos[1]/pos[0]
        az_interp_real[:,i] = np.interp(Ky, Ky_keystone, range_interp_real[:,i])
        az_interp_imag[:,i] = np.interp(Ky, Ky_keystone, range_interp_imag[:,i])        
 
    real_polar = np.nan_to_num(az_interp_real)
    imag_polar = np.nan_to_num(az_interp_imag)    
    phs_polar = np.nan_to_num(real_polar+1j*imag_polar)
    
    # 2-D FFT
    im_final = fftshift(fft2(fftshift(phs_polar)))
    return cdtype(im_final)