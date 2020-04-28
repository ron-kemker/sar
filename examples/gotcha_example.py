# -*- coding: utf-8 -*-
'''
File: gotcha_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer, polyphase_interp
from signal_processing import spatial_variant_apodization2 as SVA
import numpy as np
import matplotlib.pyplot as plt
from autofocus import multi_aperture_map_drift_algorithm as MAM
from autofocus import phase_gradient_autofocus as PGA

data_path ='..\..\data\GOTCHA\DATA\pass1\HH'

timer = Timer('FileIO')
with timer as _:
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=650e6,
                min_azimuth_angle=-50,
                max_azimuth_angle=-46,
                )
fig, ax = plt.subplots(1,3, figsize=[8,12])

timer = Timer('Polar Format Algorithm')
with timer as _:
    image = PFA(sar_obj, 
            upsample=True, 
            interp_func = np.interp,
            single_precision=True,
            )    
    imshow(image.T, ax=ax[0])
    ax[0].axis('off')
    ax[0].title.set_text('Polar Format Algorithm')

timer = Timer('Spatially Variant Apodization')
with timer as _:
    image_sva = SVA(image)
    imshow(image_sva.T, ax=ax[1])
    ax[1].axis('off')
    ax[1].title.set_text('Spatially Variant Apodization')

timer = Timer('Phase Gradient Autofocus')
with timer as _:
    image_af = PGA(image_sva)
    imshow(image_af.T, ax=ax[2])
    ax[2].axis('off')
    ax[2].title.set_text('Phase Gradient Autofocus')

