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

# Display log-scaled image
def imshow(im, x_vec=None, y_vec=None, dynamic_range=70):
        
    plt.figure(1, figsize=[10,10])
    img = np.abs(im)/np.max(np.abs(im))
    img = 20.0 * np.log10(img)
    img[img > 0] = 0.0
    img[img < -dynamic_range] = -dynamic_range
    plt.imshow(img, cmap='gray')
    if x_vec is None:
        x_tic = np.linspace(x_vec[0], x_vec[-1], 11)
        x_loc = np.linspace(0, img.shape[0], 11, dtype=np.int32)
        plt.xticks(x_loc, x_tic)
    if y_vec is None:
        y_tic = np.linspace(y_vec[0], y_vec[-1], 11)
        y_loc = np.linspace(0, img.shape[1], 11, dtype=np.int32)
        plt.yticks(y_loc, y_tic)
    
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