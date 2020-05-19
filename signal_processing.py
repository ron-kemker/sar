# -*- coding: utf-8 -*-
'''
File: signal_processing.py
Description: Contains SAR signal processing tools, mostly sidelobe management
Author: Ronald Kemker

'''

import numpy as np
from scipy.signal import convolve2d as conv2d
from utils import ft, ift
from scipy.stats import linregress
import scipy.signal as sp

# This applies a Hamming window to the CPHD file (side-lobe suppression)
def hamming_window(cphd):
    [K, Np] = cphd.shape
    hamming1 = np.hamming(K)[np.newaxis].T
    hamming2 = np.hamming(Np)[np.newaxis]
    taper_window = np.matmul(hamming1, hamming2)
    return cphd * taper_window

# This applies a Hamming window to the CPHD file (side-lobe suppression)
def hanning_window(cphd):
    [K, Np] = cphd.shape
    hamming1 = np.hanning(K)[np.newaxis].T
    hamming2 = np.hanning(Np)[np.newaxis]
    taper_window = np.matmul(hamming1, hamming2)
    return cphd * 2.0 * taper_window

# This applies a Taylor window to the CPHD file (side-lobe suppression)
def taylor_window(cphd , sidelobe=30, n_bar=4):
    [K, Np] = cphd.shape
    taylor1 = taylor(K, sidelobe, n_bar)[np.newaxis].T
    taylor2 = taylor(Np, sidelobe)[np.newaxis]
    taper_window = np.matmul(taylor1, taylor2)   
    return cphd * taper_window

def taylor(N, sidelobe, n_bar=None):
    """
    
    Parameters
    ----------
    N : Int > 0, This is the size of the taylor window on one side
    sidelobe : float >= 0.0,  This is the magnitude of the sidelobe in dB.

    Reference
    -------
    [RIT-SAR](https://github.com/dm6718/RITSAR/blob/master/ritsar/phsRead.py)

    """
    xi = np.linspace(-0.5, 0.5, N)
    A = np.arccosh(10.0**(sidelobe/20.0))/np.pi
    
    if n_bar is None:
        n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar**2 / (A**2 + (n_bar-0.5)**2)
    
    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*(-1)**(i+1)*(1-i**2*1.0/sigma_p/(A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)
            
        F_m[i-1] = num/den
    
    w = np.ones(N)
    for i in m:
        w += F_m[i-1]*np.cos(2*np.pi*i*xi)
    
    w = w/w.max()          
    return w

def spatial_variant_apodization(X, N=1):
    '''
    This runs the spatial variant apodization algorithm. 

    Parameters
    ----------
    X : complex-valued, numeric.  This is the input image.
    N : integer >=1.  This is the sampling frequency.
        DESCRIPTION. The default is 1.

    Returns
    -------
    output : numeric.  This is the output image.
    
    References
    ----------
    Stankwitz, Dallaire, and Fienup (1995).  "Nonlinear Apodization for 
    Sidelobe Control in SAR Imagery."  
    '''
    X1 = np.copy(X)
    X2 = np.rot90(X)

    filt = np.zeros((1, 2*N + 1), np.float32)
    filt[0,0] = 1
    filt[0,-1] = 1

    # Horizontal Apodization
    sum_r = conv2d(X.real, filt, mode='same')
    sum_i = conv2d(X.imag, filt, mode='same')
    
    w_u = np.zeros(sum_r.shape, X.dtype)
    w_u.real = -X.real/sum_r
    w_u.imag = -X.imag/sum_i
    
    w_u.real[w_u.real < 0] = 0.0
    w_u.real[w_u.real > 0.5] = 0.5
    w_u.imag[w_u.imag < 0] = 0.0
    w_u.imag[w_u.imag > 0.5] = 0.5
    
    X1.real = X.real + w_u.real * sum_r
    X1.imag = X.imag + w_u.imag * sum_i           
    det1 = X1 * np.conj(X1)

    # Vertical Apodization
    sum_r = conv2d(X2.real, filt, mode='same')
    sum_i = conv2d(X2.imag, filt, mode='same')
    
    w_u = np.zeros(sum_r.shape, X.dtype)
    w_u.real = -X2.real/sum_r
    w_u.imag = -X2.imag/sum_i
    
    w_u.real[w_u.real < 0] = 0.0
    w_u.real[w_u.real > 0.5] = 0.5
    w_u.imag[w_u.imag < 0] = 0.0
    w_u.imag[w_u.imag > 0.5] = 0.5
    
    X2.real = X2.real + w_u.real * sum_r
    X2.imag = X2.imag + w_u.imag * sum_i           
    det2 = X2 * np.conj(X2)
    
    X2 = np.rot90(X2, 3)
    det2 = np.rot90(det2, 3)
    
    output = np.zeros(X1.shape, X1.dtype)
    idx = det1 < det2
    output[idx] = X1[idx]
    output[~idx] = X2[~idx]
    return output

def residual_video_phase_compensation(cphd, frequency, gamma=None, 
                                      delta_r=None):
    '''
    This performs RVP Compensation.

    Parameters
    ----------
    cphd : complex-valued 2-D numpy array
        The complex phase history data to be compensated.
    frequency : numpy array of floats
        The frequency range of the input cphd.
    gamma : float
        The chirprate in Hz
    Returns
    -------
    output : same datatype as cphd
        This is the RVP compensated phase history data.

    '''    
    c = 3e8 #299792458 # Speed of Light
    [Np, K] = cphd.shape

    bandwidth = np.max(frequency) - np.min(frequency)
    
    if delta_r is None:
        delta_r = c/(2.0*bandwidth)  
                                     
    t = np.linspace(-K/2, K/2, K)
    
    if gamma is None:
        delta_t = 1 / bandwidth
        gamma,_,_,_,_ = linregress(t*delta_t, frequency)
    
    f_t = t*2*gamma/c*delta_r
    S_c = np.exp(-1j*np.pi*f_t**2/gamma)
    S_c = np.tile(S_c, [Np, 1])
    return ift(ft(cphd)*S_c)

# Polyphase interpolation as described in "Spotlight Synthetic Aperture Radar"
# written by Carrara, Goodman, and Majewski
# Code written by Doug MacDonald
def polyphase_interp (x, xp, yp, n_taps=15, n_phases=10000, cutoff = 0.95,
                      left=None, right=None):
        
    # Compute input and output sample spacing
    dxp = np.diff(xp).min()
    dx = np.diff(x).min()
    
    # Assume uniformly spaced input
    if dx > (1.001*np.diff(x)).max() or dx < (0.999*np.diff(x)).min():
            raise ValueError('Output sample spacing not uniform')
    
    # Input centered convolution - scale output sample spacing to 1
    offset = x.min()
    G = ((x-offset)/dx).astype(int)
    Gp = ((xp-offset)/dx)
    
    # Create prototype polyphase filter
    if not n_taps%2:
        raise ValueError('Filter should have odd number of taps')
    
    if dx > dxp:                    # Downsampling
        f_cutoff = cutoff           # Align filter nulls with output which has a sample spacing of 1 by definition
    else:                           # Upsampling
        f_cutoff = cutoff * dx/dxp  # Align filter nulls with input which has a normalized sample spacing of dxp/dx
    
    filt_proto = sp.firwin(n_taps, f_cutoff, fs=2)
    
    # Create polyphase filter
    filt_poly = sp.resample(filt_proto, n_taps*n_phases)
    mid_point = (n_taps-1)/2
    # Pad input for convolution
    pad_left = max(G[0] - int(np.floor(Gp[0] - mid_point)), 0)
    pad_right = max(int(np.ceil(Gp[-1] + mid_point)) - G[-1], 0)
    
    # Calculate output
    y_pad = np.zeros(x.size + pad_left + pad_right)
    
    for i in range(xp.size):
        V_current = yp[i]
        G_current = Gp[i] + pad_left
        G_left = G_current - mid_point
        G_start = int(np.ceil(G_left))
        G_right = G_current + mid_point
        G_end = int(np.floor(G_right))
        
        # Input samples may not be evenly spaced so comput a local scale factor
        if i < xp.size - 1:
            local_scale = Gp[i+1] - Gp[i]
        
        filt = filt_poly[int((G_start-G_left)*n_phases): \
                         int((G_end-G_left)*n_phases)+1:n_phases]*local_scale
        y_pad[G_start:G_end+1] += V_current*filt
      
    if pad_right > 0:
        return y_pad[pad_left:-pad_right]
    else:
        return y_pad[pad_left:]
    