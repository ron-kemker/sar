# -*- coding: utf-8 -*-
'''
File: gotcha_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer, polyphase_interp
from signal_processing import taylor_window
import numpy as np
import matplotlib.pyplot as plt
from autofocus import multi_aperture_map_drift_algorithm as MAM

data_path ='..\..\data\GOTCHA\DATA\pass8\HH'

timer = Timer('FileIO')
with timer as _:
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=taylor_window,
                min_azimuth_angle=40,
                max_azimuth_angle=45
                )

plt.figure(1, figsize=[10,10])

timer = Timer("PFA")
with timer as _:
    image = PFA(sar_obj, 
                upsample=False, 
                interp_func = np.interp,
                # num_range_samples=128,
                # num_crossrange_samples=256
                )

plt.subplot(2,1,1)
imshow(image.T)

timer = Timer("Multi-Aperture Map-Drift")
with timer as _:
    image_mam = MAM(image)

plt.subplot(2,1,2)
imshow(image_mam.T)

