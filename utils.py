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
from scipy.fftpack import fftshift, fft2, fft, ifft, ifft2
from matplotlib import cm

# 2-D FFT Helper function to clean up code
def ft2(x):
    return fftshift(fft2(fftshift(x)))

def ft(x, ax=-1):
    return fftshift(fft(fftshift(x), axis = ax))
        
def ift(x, ax = -1):
    return fftshift(ifft(fftshift(x), axis = ax))

def ift2(x):
    return fftshift(ifft2(fftshift(x)))

def histogram_equalization_rgb(img):
    
    for i in range(3):
        img[...,i] = historgram_equalization(img[...,i])
        
    return img

def historgram_equalization(img):
    
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]
    
    
def pauli_decomposition(S_hh, S_vv, S_hv):
       
    red = np.abs(S_hh - S_vv)
    green = np.abs(S_hv) 
    blue = np.abs(S_hh + S_vv) 

    red = np.log10(red / np.max(red))
    green = np.log10(green / np.max(green))
    blue = np.log10(blue / np.max(blue))
    
    red = red - red.min()
    green = green - green.min()
    blue = blue - blue.min()
    
    red = red / np.max(red) * 255.0
    green = green / np.max(green) * 255.0
    blue = blue / np.max(blue) * 255.0

    return np.vstack((red[np.newaxis],green[np.newaxis],
                      blue[np.newaxis])).transpose(1,2,0).astype('uint8')

def krogager_decomposition(S_hh, S_vv, S_hv, dynamic_range = 30):
    pass

# Display log-scaled image
def imshow(im, dynamic_range=70, ax=plt):
        
    img_abs = np.abs(im)
    img = img_abs / img_abs.max()
    img = 10.0 * np.log10(img)
    img[img == -np.inf] = -dynamic_range
    ax.imshow(img, cmap=cm.Greys_r, vmin = -dynamic_range, vmax = 0.0)
    
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