# -*- coding: utf-8 -*-
'''
File: interpolation_comparison.py
Description: Compare interpolation methods on Sandia Dataset 
Author: Ronald Kemker

'''

import numpy as np
from fileIO.sandia import SANDIA
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer
import matplotlib.pyplot as plt
from signal_processing import residual_video_phase_compensation as RVP_Comp
from scipy import interpolate

def cubic_interp(x, xp, yp, left=0, right=0):
    f = interpolate.interp1d(xp, yp, kind='cubic', assume_sorted=True,
                             fill_value=(left,right), bounds_error=False)
    return f(x)

if __name__ == '__main__':
    
    data_path ='..\..\data\Sandia\\'
    plt.close('all')
    
    timer = Timer('FileIO')
    with timer as _:
        sar_obj = SANDIA(data_path)
        
    fig, ax = plt.subplots(1,2, figsize=[8,8])
    plt.tight_layout()
    
    sar_obj.cphd = RVP_Comp(sar_obj.cphd, sar_obj.freq, sar_obj.chirprate,
                            sar_obj.delta_r)
    
    timer = Timer('Cubic Interpolation')
    with timer as _:
        image = PFA(sar_obj, 
                interp_func=cubic_interp,
                single_precision=True,
                taylor_weighting=30,
                )
    
        imshow(image, ax=ax[0], dynamic_range=45)
        ax[0].axis('off')
        ax[0].title.set_text('Cubic Interpolation')
    
    timer = Timer('Polyphase Interpolation')
    with timer as _:
        image = PFA(sar_obj, 
                single_precision=True,
                taylor_weighting=30,
                n_jobs=8,
                )
        imshow(image, ax=ax[1], dynamic_range=45)
        ax[1].axis('off')
        ax[1].title.set_text('Polyphase Interpolation')

    



