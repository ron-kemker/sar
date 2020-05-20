# -*- coding: utf-8 -*-
'''
File: re_mocomp_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

import numpy as np
from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow
from signal_processing import re_mocomp
import matplotlib.pyplot as plt
from signal_processing import residual_video_phase_compensation as RVP_Comp

if __name__ == '__main__':

    data_path ='..\..\data\GOTCHA\DATA\pass1\VV'
    dynamic_range = 30
    taylor_weighting = 30
    plt.close('all')
    
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=650e6,
                min_azimuth_angle = 0,
                max_azimuth_angle = 4,
                )
    
    fig, ax = plt.subplots(1,2 , figsize=[8,8])
    
    sar_obj.cphd = RVP_Comp(sar_obj.cphd, sar_obj.freq)

    image = PFA(sar_obj, 
            single_precision=True,
            n_jobs=8,
            taylor_weighting=taylor_weighting,
            )
    imshow(image, dynamic_range, ax=ax[0])
    ax[0].axis('off')
    ax[0].title.set_text('Before Re-Mocomp')

    # Re-Mocomp to Toyota Camry (B-LR)
    sar_obj.cphd = re_mocomp(sar_obj.cphd, sar_obj.antenna_location,
                              sar_obj.k_r, np.array([21.65, -16.40, 0.03]))    
    
    image = PFA(sar_obj, 
            single_precision=True,
            n_jobs=8,
            taylor_weighting=taylor_weighting,
            )
    imshow(image, dynamic_range, ax=ax[1])
    ax[1].axis('off')
    ax[1].title.set_text('After Re-Mocomp')
    


