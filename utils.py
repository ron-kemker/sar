# -*- coding: utf-8 -*-
'''
File: utils.py
Description: Contains various SAR utilities
Author: Ronald Kemker

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Display log-scaled image
def imshow(im, x_vec, y_vec, dynamic_range=70):
    plt.figure(1, figsize=[10,10])
    img = np.abs(im)/np.max(np.abs(im))
    img = 20.0 * np.log10(img)
    img[img > 0] = 0.0
    img[img < -dynamic_range] = -dynamic_range
    plt.imshow(img, cmap='gray')
    plt.xlabel('x [meters]')
    plt.ylabel('y [meters]')
    x_tic = np.linspace(x_vec[0], x_vec[-1], 11)
    y_tic = np.linspace(y_vec[0], y_vec[-1], 11)
    x_loc = np.linspace(0, img.shape[0], 11, dtype=np.int32)
    y_loc = np.linspace(0, img.shape[1], 11, dtype=np.int32)
    plt.xticks(x_loc, x_tic)
    plt.yticks(y_loc, y_tic)
    
def getAmplitudeOnly(img):
    return np.abs(img)

def getPhaseOnly(img):
    return np.angle(img)

def getAmplitudePhase(img):
    amp = getAmplitudeOnly(img)[:,:,np.newaxis]
    return np.append(amp, getPhaseOnly(img)[:,:,np.newaxis], 2)

def ttest(x1 , x2, confidence=0.99):
    
    ttest, pval = ttest_ind(x1, x2, equal_var=False)
    if pval < 1-confidence:
        return True
    else:
        return False
