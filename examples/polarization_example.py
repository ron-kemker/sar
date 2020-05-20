# -*- coding: utf-8 -*-
'''
File: polarization_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

import numpy as np
from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow
import matplotlib.pyplot as plt
from signal_processing import residual_video_phase_compensation as RVP_Comp

if __name__ == '__main__':

    data_path ='..\..\data\GOTCHA\DATA\pass1\\'
    dynamic_range = 30
    plt.close('all')
        
    fig, ax = plt.subplots(2,2 , figsize=[8,8])
    polarization = ['VV','HH','VH']
    image_arr = []
    for i, pol in enumerate(polarization):
        
        sar_obj = GOTCHA(data_path+pol,
                    center_frequency=9.6e9, 
                    bandwidth=650e6,
                    min_azimuth_angle = 178,
                    max_azimuth_angle = 182,
                    verbose=False,
                    )      
        
        sar_obj.cphd = RVP_Comp(sar_obj.cphd, sar_obj.freq)
        
        image = PFA(sar_obj, 
                single_precision=True,
                n_jobs=8,
                taylor_weighting = 30,
                )
        imshow(image, dynamic_range, ax=ax[int(i//2)][int(i%2)])
        ax[int(i//2)][int(i%2)].axis('off')
        ax[int(i//2)][int(i%2)].title.set_text(pol)
        image_arr += [image]

    rgb_img = np.zeros(image_arr[0].shape+(3, ), dtype=np.complex64 )
    sh = rgb_img.shape
    rgb_img[...,0] = image_arr[0]
    rgb_img[...,1] = image_arr[1]
    rgb_img[...,2] = image_arr[2]
    
    rgb_img = np.abs(rgb_img)
    rgb_img = rgb_img.reshape(-1, 3)
    rgb_img = rgb_img / rgb_img.max(0)
    rgb_img = 10.0 * np.log10(rgb_img)
    X_std = (rgb_img - rgb_img.min(axis=0)) / \
        (rgb_img.max(axis=0) - rgb_img.min(axis=0))
    X_scaled = np.uint8(X_std * 255).reshape(sh)
    ax[1][1].imshow(X_scaled)    
    ax[1][1].axis('off')
    ax[1][1].title.set_text('Dual-Pol False Color')