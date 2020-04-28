# -*- coding: utf-8 -*-
'''
File: gotcha_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer, polyphase_interp
from signal_processing import spatial_variant_apodization as SVA
import numpy as np
import matplotlib.pyplot as plt
from autofocus import phase_gradient_autofocus as PGA
from signal_processing import residual_video_phase_compensation as RVP_Comp

data_path ='..\..\data\GOTCHA\DATA\pass1\HH'
plt.close('all')


timer = Timer('FileIO')
with timer as _:
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=650e6,
                min_azimuth_angle=-50,
                max_azimuth_angle=-46,
                )
fig, ax = plt.subplots(1,3 , figsize=[8,8])

sar_obj.cphd = RVP_Comp(sar_obj.cphd, sar_obj.freq)


timer = Timer('Polar Format Algorithm')
with timer as _:
    image = PFA(sar_obj, 
            upsample=True, 
            # interp_func = np.interp,
            single_precision=True,
            )    
    imshow(image.T, ax=ax[0])
    ax[0].axis('off')
    ax[0].title.set_text('Polar Format Algorithm')


timer = Timer('Spatially Variant Apodization')
with timer as _:
    image_sva = SVA(image, N=1)
    imshow(image_sva.T, ax=ax[1])
    ax[1].axis('off')
    ax[1].title.set_text('Spatially Variant Apodization')

timer = Timer('Phase Gradient Autofocus')
with timer as _:
    image_af = PGA(image_sva, range_subset=100)
    imshow(image_af.T, ax=ax[2])
    ax[2].axis('off')
    ax[2].title.set_text('Phase Gradient Autofocus')

