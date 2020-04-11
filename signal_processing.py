# -*- coding: utf-8 -*-
'''
File: signal_processing.py
Description: Contains SAR signal processing tools
Author: Ronald Kemker

'''

import numpy as np

# This applies a hamming window to the CPHD file (side-lobe suppression)
def hamming_window(cphd):
    [K, Np] = cphd.shape
    hamming1 = np.hamming(K)[np.newaxis].T
    hamming2 = np.hamming(Np)[np.newaxis]
    taper_window = np.matmul(hamming1, hamming2)
    return cphd * taper_window