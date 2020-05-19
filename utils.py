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