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
    
def multi_aperture_map_drift_algorithm(range_data, N=3, num_iter=6):
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
        
        if range_bins < 1:
            break
        
        map_arr = []
        
        ti_pix = np.int32(t_i + Ta/2)
        h = int(Ta/(2*N))
        for i in range(N):
            map_arr += [np.abs(ft2(range_data[ti_pix[i]-h:ti_pix[i]+h]))]
        
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
        
        # Compute quadratic error for phi
        # TODO: I need to apply the entire N-th order phase change, my bad 
        phi = 0.0
        for i in range(num_pairs):
            phi += a_vec[0] * 2 * t_i[i]
            
        # Discard range bins 1 std away from mean
        mean_err = np.mean(rel_shift)
        one_std_from_mean = np.std(rel_shift)#/2.0
        mean_range_err = np.mean(rel_shift, 0)
        range_mask_idx = np.argwhere(range_mask)
        less_than = mean_range_err < (mean_err - one_std_from_mean)
        greater_than = mean_range_err > (mean_err + one_std_from_mean)
        discard = less_than + greater_than
        range_mask[range_mask_idx[discard]] = False
        range_data = range_data * np.exp(1j * phi)

    return range_data

def phase_gradient_autofocus(range_data):
    
    return
    
# def autoFocus(img, win = 'auto', win_params = [100,0.5]):
# ##############################################################################
# #                                                                            #
# #  This program autofocuses an image using the Phase Gradient Algorithm.     #
# #  If the parameter win is set to auto, an adaptive window is used.          #
# #  Otherwise, the user sets win to 0 and defines win_params.  The first      #
# #  element of win_params is the starting windows size.  The second element   #
# #  is the factor by which to reduce it by for each iteration.  This version  #
# #  is more suited for an image that is mostly focused.  Below is the paper   #
# #  this algorithm is based off of.                                           #
# #                                                                            #
# #  D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    #
# #  autofocus-a robust tool for high resolution sar phase correction,"        #
# #  Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          #
# #  pp. 827{835, Jul 1994.                                                    #
# #                                                                            #
# ##############################################################################
    
#     #Derive parameters
#     npulses = int(img.shape[0])
#     nsamples = int(img.shape[1])
    
#     #Initialize loop variables
#     img_af = 1.0*img
#     max_iter = 30
#     af_ph = 0
#     rms = []
    
#     #Compute phase error and apply correction
#     for iii in range(max_iter):
        
#         #Find brightest azimuth sample in each range bin
#         index = np.argsort(np.abs(img_af), axis=0)[-1]
        
#         #Circularly shift image so max values line up   
#         f = np.zeros(img.shape)+0j
#         for i in range(nsamples):
#             f[:,i] = np.roll(img_af[:,i], npulses/2-index[i])
        
#         if win == 'auto':
#             #Compute window width    
#             s = np.sum(f*np.conj(f), axis = -1)
#             s = 10*np.log10(s/s.max())
#             width = np.sum(s>-30)
#             window = np.arange(npulses/2-width/2,npulses/2+width/2)
#         else:
#             #Compute window width using win_params if win not set to 'auto'    
#             width = int(win_params[0]*win_params[1]**iii)
#             window = np.arange(npulses/2-width/2,npulses/2+width/2)
#             if width<5:
#                 break
        
#         #Window image
#         g = np.zeros(img.shape)+0j
#         g[window] = f[window]
        
#         #Fourier Transform
#         G = sig.ift(g, ax=0)
        
#         #take derivative
#         G_dot = np.diff(G, axis=0)
#         a = np.array([G_dot[-1,:]])
#         G_dot = np.append(G_dot,a,axis = 0)
        
#         #Estimate Spectrum for the derivative of the phase error
#         phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
#                   np.sum(np.abs(G)**2, axis = -1)
                
#         #Integrate to obtain estimate of phase error(Jak)
#         phi = np.cumsum(phi_dot)
        
#         #Remove linear trend
#         t = np.arange(0,nsamples)
#         slope, intercept, r_value, p_value, std_err = linregress(t,phi)
#         line = slope*t+intercept
#         phi = phi-line
#         rms.append(np.sqrt(np.mean(phi**2)))
        
#         if win == 'auto':
#             if rms[iii]<0.1:
#                 break
        
#         #Apply correction
#         phi2 = np.tile(np.array([phi]).T,(1,nsamples))
#         IMG_af = sig.ift(img_af, ax=0)
#         IMG_af = IMG_af*np.exp(-1j*phi2)
#         img_af = sig.ft(IMG_af, ax=0)
        
#         #Store phase
#         af_ph += phi  