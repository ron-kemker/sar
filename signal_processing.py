# -*- coding: utf-8 -*-
'''
File: signal_processing.py
Description: Contains SAR signal processing tools
Author: Ronald Kemker

'''

import numpy as np

# This applies a Hamming window to the CPHD file (side-lobe suppression)
def hamming_window(cphd):
    [K, Np] = cphd.shape
    hamming1 = np.hamming(K)[np.newaxis].T
    hamming2 = np.hamming(Np)[np.newaxis]
    taper_window = np.matmul(hamming1, hamming2)
    return cphd * taper_window

# This applies a Taylor window to the CPHD file (side-lobe suppression)
def taylor_window(cphd , sidelobe=43):
    [K, Np] = cphd.shape
    taylor1 = taylor(K, sidelobe)[np.newaxis].T
    taylor2 = taylor(Np, sidelobe)[np.newaxis]
    taper_window = np.matmul(taylor1, taylor2)   
    return cphd * taper_window

def taylor(N, sidelobe=43):
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
    A = 1.0/np.pi*np.arccosh(10**(sidelobe*1.0/20))
    n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar/np.sqrt(A**2+(n_bar-0.5)**2)
    
    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p**2/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)
            
        F_m[i-1] = num/den
    
    w = np.ones(N)
    for i in m:
        w += F_m[i-1]*np.cos(2*np.pi*i*xi)
    
    w = w/w.max()          
    return w