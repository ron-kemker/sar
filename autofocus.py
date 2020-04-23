# -*- coding: utf-8 -*-
'''

File: autofocus.py
Description: Contains SAR autofocus algorithms
Author: Ronald Kemker

'''

import numpy as np
from warnings import warn
from utils import ft2

# Helper function for multi_aperture_map_drift_algorithm
def corr_help(X, Y):
    delta = np.zeros((X.shape[1], ), dtype=np.float32)
    for i in range(X.shape[1]):
        corr = np.correlate(X[:,i], Y[:, i], mode='same')
        delta[i] = np.argmax(corr)
        
    return delta - X.shape[0]/2
    
def multi_aperture_map_drift_algorithm(range_data, N=2, num_iter=6):
    """Performs multi-aperture map-drift algorithm to correct Nth Order Phase
       Errors.
    
    @author: Ronald Kemker

    # Arguments
        range_data: Complex data. This is the compressed range data from PFA.
        N: integer >=2.  This is the number of subapertures for map-drift.
        num_iter: integer >=1.  The number of iterations to run MAM
    # References
        - Carrera, Goodman, and Majewski (1995).
    """    
    
    if N > 5:
        warn('MAM works best with N<=5')
    
    (Ta, K) = range_data.shape
    t_i = ((np.arange(1,N+1)/N)-((N+1)/(2*N))) * Ta 
    range_bins = np.max([int(0.05*K), 32])
    idx = np.argsort(np.abs(range_data.ravel()))[:range_bins]
    _, idx = np.unravel_index(idx, range_data.shape)
    range_mask = np.zeros((K, ), dtype=np.bool)
    range_mask[np.unique(idx)] = True
    phi = 0
    
    for _ in range(num_iter):
        
        range_bins = np.sum(range_mask)
        data = range_data * np.exp(1j * phi)
        map_arr = []
        
        ti_pix = np.int32(t_i + Ta/2)
        h = int(Ta/(2*N))
        for i in range(N):
            map_arr += [np.abs(ft2(data[ti_pix[i]-h:ti_pix[i]+h]))]
        
        num_pairs = int(N*(N-1)/2)
        rel_shift = np.zeros((num_pairs, range_bins ) , dtype=np.float32)
        delta = np.zeros ((num_pairs, N-1) , dtype=np.float32)
        r=0
        for i in range(N):
            for j in range(i+1, N):
                rel_shift[r] = corr_help(map_arr[i][:,range_mask],
                                 map_arr[j][:,range_mask])
                for k in range(2,N+1):
                    delta[r,k-2] = k/(2*np.pi)*(t_i[j]**(k-1) - t_i[i]**(k-1))
                r+=1
    
        delta_inv = np.linalg.pinv(delta)
        a_vec = np.matmul(delta_inv, np.mean(rel_shift, 1))
        
        # Build new phi value off of computed a_vec
        # phi = 0.0
        for i in range(num_pairs):
            phi += a_vec[0] * 2 * t_i[i]
            
        # Discard range bins 1 std away from mean
        mean_err = np.mean(rel_shift, 1)
        one_std_from_mean = np.std(rel_shift, 1)/2.0
        mean_range_err = np.mean(rel_shift, 0)
        range_mask_idx = np.argwhere(range_mask)
        less_than = mean_range_err < mean_err - one_std_from_mean
        greater_than = mean_range_err > mean_err + one_std_from_mean
        discard = less_than + greater_than
        range_mask[range_mask_idx[discard]] = False
        
               
    return range_data * np.exp(1j * phi)