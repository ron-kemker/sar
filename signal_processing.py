# -*- coding: utf-8 -*-
'''
File: signal_processing.py
Description: Contains SAR signal processing tools, mostly sidelobe management
Author: Ronald Kemker

'''

import numpy as np
from scipy.signal import convolve2d as conv2d

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
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)
            
        F_m[i-1] = num/den
    
    w = np.ones(N)
    for i in m:
        w += F_m[i-1]*np.cos(2*np.pi*i*xi)
    
    w = w/w.max()          
    return w

def spatial_variant_apodization(X, N=2):
    '''
    This runs the spatial variant apodization algorithm 
    This matches a version shared by a co-worker.  

    Parameters
    ----------
    X : complex-valued, numeric.  This is the input image.
    N : integer >=1.  This is the sampling frequency.
        DESCRIPTION. The default is 2.

    Returns
    -------
    output : numeric.  This is the output image.
    
    References
    ----------
        - Carrera, Goodman, and Majewski (1995), Appendix D
    '''
    # Define functions to clean up space    
    min2D = np.minimum
    max2D = np.maximum
    
    X2 = np.rot90(X)

    filt = np.zeros((1, 2*N + 1), np.float32)
    filt[0,0] = 1
    filt[0,-1] = 1

    # Horizontal Apodization
    sum_r = conv2d(X.real, filt, mode='same')
    sum_i = conv2d(X.imag, filt, mode='same')
    
    weights = np.zeros(sum_r.shape, X.dtype)
    idx = sum_r != 0
    weights.real[idx] = min2D(max2D(-X.real[idx]/sum_r[idx], 
                                    0.0), 0.5)
    idx = sum_i != 0
    weights.imag[idx] = min2D(max2D(-X.imag[idx]/sum_i[idx], 
                                    0.0), 0.5)
    
    X1 = X.real+sum_r*weights.real + (X.imag+sum_i*weights.imag) * 1j 
             
    det1 = X1 * np.conj(X1)

    # Vertical Apodization
    sum_r = conv2d(X2.real, filt, mode='same')
    sum_i = conv2d(X2.imag, filt, mode='same')
    
    weights = np.zeros(sum_r.shape, X.dtype)
    idx = sum_r != 0
    weights.real[idx] = min2D(max2D(-X2.real[idx]/sum_r[idx], 
                                    0.0), 0.5)
    idx = sum_i != 0
    weights.imag[idx] = min2D(max2D(-X2.imag[idx]/sum_i[idx], 
                                    0.0), 0.5)
    
    X2 = X2.real+sum_r*weights.real + (X2.imag+sum_i*weights.imag) * 1j 
             
    det2 = X2 * np.conj(X2)
    
    X2 = np.rot90(X2, 3)
    det2 = np.rot90(det2, 3)
    
    output = np.zeros(X1.shape, X1.dtype)
    idx = det1 < det2
    output[idx] = X1[idx]
    output[~idx] = X2[~idx]
    
    return output

def spatial_variant_apodization2(X, N=1):
    '''
    This runs the spatial variant apodization algorithm 
    This is more similar to the book implementation.

    Parameters
    ----------
    X : complex-valued, numeric.  This is the input image.
    N : integer >=1.  This is the sampling frequency.
        DESCRIPTION. The default is 2.

    Returns
    -------
    output : numeric.  This is the output image.
    
    References
    ----------
        - Carrera, Goodman, and Majewski (1995), Appendix D
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
    
    X1 = np.copy(X)
    idx_r = np.logical_and(w_u.real >= 0.0, w_u.real<= 0.5) 
    idx_i = np.logical_and(w_u.imag >= 0.0, w_u.imag<= 0.5)
    X1.real[idx_r] = 0.0
    X1.imag[idx_i] = 0.0
    
    idx_r = w_u.real > 0.5
    idx_i = w_u.imag > 0.5
    X1.real[idx_r] = X.real[idx_r]+sum_r[idx_r]*0.5
    X1.imag[idx_i] = X.imag[idx_i]+sum_i[idx_i]*0.5            
    det1 = X1 * np.conj(X1)

    # Vertical Apodization
    sum_r = conv2d(X2.real, filt, mode='same')
    sum_i = conv2d(X2.imag, filt, mode='same')
    
    w_u = np.zeros(sum_r.shape, X.dtype)
    w_u.real = -X2.real/sum_r
    w_u.imag = -X2.imag/sum_i
    
    idx_r = np.logical_and(w_u.real >= 0.0, w_u.real<= 0.5) 
    idx_i = np.logical_and(w_u.imag >= 0.0, w_u.imag<= 0.5)
    X2.real[idx_r] = 0.0
    X2.imag[idx_i] = 0.0 
    
    idx_r = w_u.real > 0.5
    idx_i = w_u.imag > 0.5
    X2.real[idx_r] = X2.real[idx_r]+sum_r[idx_r]*0.5
    X2.imag[idx_i] = X2.imag[idx_i]+sum_i[idx_i]*0.5         
    det2 = X2 * np.conj(X2)
    
    X2 = np.rot90(X2, 3)
    det2 = np.rot90(det2, 3)
    
    output = np.zeros(X1.shape, X1.dtype)
    idx = det1 < det2
    output[idx] = X1[idx]
    output[~idx] = X2[~idx]
    return output