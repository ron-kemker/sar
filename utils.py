# -*- coding: utf-8 -*-
'''
File: utils.py
Description: Contains various SAR utilities
Author: Ronald Kemker

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import time
import scipy.signal as sp

# Display log-scaled image
def imshow(im, dynamic_range=70):
        
    plt.figure(1, figsize=[10,10])
    img = np.abs(im)/np.max(np.abs(im))
    img = 20.0 * np.log10(img)
    img[img > 0] = 0.0
    img[img < -dynamic_range] = -dynamic_range
    plt.imshow(img, cmap='gray')
    
    
# Return magnitide-only image from complex-valued image
def getAmplitudeOnly(img):
    return np.abs(img)

# Return phase-only image from complex-valued image
def getPhaseOnly(img):
    return np.angle(img)

# Convert complex-valued image into two channel image (for deep learning)
def getAmplitudePhase(img):
    amp = getAmplitudeOnly(img)[:,:,np.newaxis]
    return np.append(amp, getPhaseOnly(img)[:,:,np.newaxis], 2)

# Two-side t-test for statistical significance testing
def ttest(x1 , x2, confidence=0.99):
    
    ttest, pval = ttest_ind(x1, x2, equal_var=False)
    if pval < 1-confidence:
        return True
    else:
        return False

# Timer object
class Timer(object):
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tic = time.time() 
    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        self.delta = time.time() - self.tic
        print('Elapsed: %1.3f seconds' % self.delta)
        
# Polyphase interpolation as described in "Spotlight Synthetic Aperture Radar"
# written by Carrara, Goodman, and Majewski
# Code written by Doug MacDonald
# TODO: Implement left/right to match np.interp format        
def polyphase_interp(x, xp , yp, n_taps=15, n_phases=1000, cutoff=0.9):
    
    dtype = yp.dtype
    
    # Compute input and output sample spacing
    dxp = np.diff(xp).min()
    dx = np.diff(x).min()
    
    # Assume uniformly spaced output
    if dx > (1.001 * np.diff(x)).max() or dx < (0.999*np.diff(x)).min():
        raise ValueError('Output sample spacing not uniform')
        
    # Input centered convolution - scale output sample spacing to 1
    offset = x.min()
    G = ((x-offset)/dx).astype(int)
    Gp = ((xp-offset)/dx)
    
    # Create prototpe filter
    if not n_taps%2:
        raise ValueError('Filter should have odd number of taps')
        
    if dx > dxp:                    # Downsampling
        f_cutoff = cutoff           # Align filter nulls w/ output which has
                                    # sample spacing = 1 by definition
    else:                           # Upsampling
        f_cutoff = cutoff * dx/dxp  # Align filter nulls w/ input which has a
                                    # normalized sample spacing of dxp/dx
    filt_proto = sp.firwin(n_taps, f_cutoff, fs=2)
    
    # Create polyphase filter
    filt_poly = sp.resample(filt_proto, n_taps * n_phases)
    
    # Pad for convolution
    pad_left = max(G[0] - int(np.floor(Gp[0] - (n_taps - 1)/2)), 0)
    pad_right = max(G[-1] - int(np.ceil(Gp[0] - (n_taps - 1)/2) - G[-1]), 0)
    
    # Calculate output
    y_pad = np.zeros(x.size + pad_left + pad_right, dtype=dtype)
    
    for i in range(xp.size):
        V_current = yp[i]
        G_current = Gp[i] + pad_left
        G_left = G_current - (n_taps-1)/2
        G_start = int(np.ceil(G_left))
        G_right = G_current + (n_taps-1)/2
        G_end = int(np.floor(G_right))
        
        if i < xp.size-1:
            local_scale=Gp[i+1] - Gp[i]
            
        filt = filt_poly[int((G_start - G_left))*n_phases : \
                        int((G_end-G_left)*n_phases)+1 : n_phases]*local_scale
        y_pad[G_start:G_end+1]  += V_current*filt
        
    return y_pad[pad_left : -pad_right]